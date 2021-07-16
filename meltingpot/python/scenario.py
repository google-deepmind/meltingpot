# Copyright 2020 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Scenario factory."""

import concurrent
import random
from typing import Callable, List, Mapping, Sequence, Tuple

from absl import logging
import dm_env
from ml_collections import config_dict
import numpy as np
from meltingpot.python import bot as bot_factory
from meltingpot.python import substrate as substrate_factory
from meltingpot.python.configs import scenarios as scenario_config
from meltingpot.python.utils.scenarios.wrappers import agent_slot_wrapper
from meltingpot.python.utils.scenarios.wrappers import all_observations_wrapper
from meltingpot.python.utils.scenarios.wrappers import base
from meltingpot.python.utils.scenarios.wrappers import default_observation_wrapper
from meltingpot.python.utils.scenarios.wrappers import restricted_observations_wrapper

AVAILABLE_SCENARIOS = frozenset(scenario_config.SCENARIOS)
PERMITTED_OBSERVATIONS = frozenset({
    'INVENTORY',
    'READY_TO_SHOOT',
    'RGB',
})


def _step_fn(policy: bot_factory.Policy) -> Callable[[dm_env.TimeStep], int]:
  """Returns a stateful step function where the state is encapsulated.

  Args:
    policy: the underlying policy to use.

  Returns:
    A step function that returns an action in response to a timestep.
  """
  state = policy.initial_state()

  def step(timestep: dm_env.TimeStep) -> int:
    nonlocal state
    action, state = policy.step(timestep=timestep, prev_state=state)
    return action

  return step


class Scenario(base.Wrapper):
  """An substrate where a number of player slots are filled by bots."""

  def __init__(
      self,
      substrate,
      bots: Mapping[str, bot_factory.Policy],
      num_bots: int,
  ) -> None:
    """Initializes the scenario.

    Args:
      substrate: the substrate to add bots to.
      bots: the bots to sample from (with replacement) each episode.
      num_bots: the number of to sample each episode.
    """
    super().__init__(substrate)
    self._bots = dict(bots)
    self._num_bots = num_bots
    self._executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=self._num_bots)
    self._bot_step_fns: List[Callable[[dm_env.TimeStep], int]] = []
    self._action_futures: List[concurrent.futures.Future] = []

  def close(self):
    """See base class."""
    for bot in self._bots.values():
      bot.close()
    self._executor.shutdown(wait=False)
    super().close()

  def _resample_bots(self):
    """Resamples the currently active bots."""
    sampled_names = random.choices(tuple(self._bots), k=self._num_bots)
    logging.info('Resampled bots: {sampled_bot_names}')
    self._bot_step_fns = [_step_fn(self._bots[name]) for name in sampled_names]
    for future in self._action_futures:
      future.cancel()
    self._action_futures.clear()

  def _send_timesteps(self, timesteps: Sequence[dm_env.TimeStep]) -> None:
    """Sends timesteps to bots for asynchronous processing."""
    assert not self._action_futures
    for bot_step, timestep in zip(self._bot_step_fns, timesteps):
      future = self._executor.submit(bot_step, timestep=timestep)
      self._action_futures.append(future)

  def _await_actions(self) -> Sequence[int]:
    """Waits for the bots actions form the last timestep sent."""
    assert self._action_futures
    actions = [future.result() for future in self._action_futures]
    self._action_futures.clear()
    return actions

  def _split_timestep(
      self, timestep: dm_env.TimeStep
  ) -> Tuple[dm_env.TimeStep, Sequence[dm_env.TimeStep]]:
    """Splits multiplayer timestep as needed by agents and bots."""
    agent_timestep = timestep._replace(
        reward=timestep.reward[:-self._num_bots],
        observation=timestep.observation[:-self._num_bots])
    bot_timesteps = [
        timestep._replace(
            observation=timestep.observation[-i], reward=timestep.reward[-i])
        for i in range(-self._num_bots, 0)
    ]
    return agent_timestep, bot_timesteps

  def reset(self) -> Sequence[dm_env.TimeStep]:
    """See base class."""
    self._resample_bots()
    timestep = super().reset()
    agent_timestep, bot_timesteps = self._split_timestep(timestep)
    self._send_timesteps(bot_timesteps)
    return agent_timestep

  def step(self, action: Sequence[int]) -> Sequence[dm_env.TimeStep]:
    """See base class."""
    agent_actions = action
    bot_actions = self._await_actions()
    timestep = super().step(list(agent_actions) + list(bot_actions))
    agent_timestep, bot_timesteps = self._split_timestep(timestep)
    self._send_timesteps(bot_timesteps)
    return agent_timestep

  def action_spec(self) -> Sequence[dm_env.specs.DiscreteArray]:
    """See base class."""
    return super().action_spec()[:-self._num_bots]

  def observation_spec(self) -> Sequence[Mapping[str, dm_env.specs.Array]]:
    """See base class."""
    return super().observation_spec()[:-self._num_bots]

  def reward_spec(self) -> Sequence[dm_env.specs.Array]:
    """See base class."""
    # TODO(b/192925212): better typing to avoid pytype disables.
    reward_spec: Sequence[dm_env.specs.Array] = super().reward_spec()  # pytype: disable=annotation-type-mismatch
    return reward_spec[:-self._num_bots]


def get_config(scenario_name: str) -> config_dict.ConfigDict:
  """Returns a config for the specified scenario.

  Args:
    scenario_name: Name of the scenario. Must be in AVAILABLE_SCENARIOS.
  """
  if scenario_name not in AVAILABLE_SCENARIOS:
    raise ValueError(f'Unknown scenario {scenario_name!r}')
  scenario = scenario_config.SCENARIOS[scenario_name]
  substrate = substrate_factory.get_config(scenario.substrate)
  bots = {name: bot_factory.get_config(name) for name in scenario.bots}
  config = config_dict.create(
      substrate=substrate,
      num_players=scenario.num_focal_agents,
      bots=bots,
      num_bots=scenario.num_background_bots,
  )
  return config.lock()


def build(config: config_dict.ConfigDict) -> dm_env.Environment:
  """Builds a scenario for the given config.

  Args:
    config: config resulting from `get_config`.

  Returns:
    The test scenario.
  """
  substrate = substrate_factory.build(config.substrate)
  bots = {
      bot_name: bot_factory.build(bot_config)
      for bot_name, bot_config in config.bots.items()
  }

  # Add observations needed by the bots.
  substrate = all_observations_wrapper.Wrapper(
      substrate, observations_to_share=['POSITION'], share_actions=True)
  substrate = agent_slot_wrapper.Wrapper(substrate)
  add_inventory = 'INVENTORY' not in substrate.observation_spec()[0]
  if add_inventory:
    substrate = default_observation_wrapper.Wrapper(
        substrate, key='INVENTORY', default_value=np.zeros([1]))

  scenario = Scenario(substrate=substrate, bots=bots, num_bots=config.num_bots)
  scenario = restricted_observations_wrapper.Wrapper(
      scenario, permitted_observations=PERMITTED_OBSERVATIONS)
  return scenario
