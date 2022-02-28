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
import dataclasses
import random
from typing import Any, Callable, Collection, Iterable, List, Mapping, Sequence, Tuple, TypeVar

import dm_env
import immutabledict
from ml_collections import config_dict
import numpy as np
import rx
from rx import subject

from meltingpot.python import bot as bot_factory
from meltingpot.python import substrate as substrate_factory
from meltingpot.python.configs import scenarios as scenario_config
from meltingpot.python.utils.scenarios.wrappers import agent_slot_wrapper
from meltingpot.python.utils.scenarios.wrappers import all_observations_wrapper
from meltingpot.python.utils.scenarios.wrappers import base
from meltingpot.python.utils.scenarios.wrappers import default_observation_wrapper

AVAILABLE_SCENARIOS = frozenset(scenario_config.SCENARIOS)

SCENARIOS_BY_SUBSTRATE: Mapping[
    str, Collection[str]] = scenario_config.scenarios_by_substrate(
        scenario_config.SCENARIOS)

PERMITTED_OBSERVATIONS = frozenset({
    'INVENTORY',
    'READY_TO_SHOOT',
    'RGB',
})

T = TypeVar('T')


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


class Population:
  """A population of policies to use in a scenario."""

  def __init__(self, policies: Mapping[str, bot_factory.Policy],
               population_size: int) -> None:
    """Initializes the population.

    Args:
      policies: the policies to sample from (with replacement) each episode.
      population_size: the number of policies to sample on each reset.
    """
    self._policies = dict(policies)
    self._population_size = population_size
    self._executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=self._population_size)
    self._step_fns: List[Callable[[dm_env.TimeStep], int]] = []
    self._action_futures: List[concurrent.futures.Future] = []

  def close(self):
    """Closes the population."""
    for future in self._action_futures:
      future.cancel()
    self._executor.shutdown(wait=False)
    for policy in self._policies.values():
      policy.close()

  def _sample_names(self) -> Sequence[str]:
    """Returns a sample of policy names for the population."""
    return random.choices(tuple(self._policies), k=self._population_size)

  def reset(self) -> None:
    """Resamples the population."""
    names = self._sample_names()
    self._step_fns = [_step_fn(self._policies[name]) for name in names]
    for future in self._action_futures:
      future.cancel()
    self._action_futures.clear()

  def send_timestep(self, timestep: dm_env.TimeStep) -> None:
    """Sends timestep to population for asynchronous processing.

    Args:
      timestep: The substrate timestep for the popualtion.

    Raises:
      RuntimeError: previous action has not been awaited.
    """
    if self._action_futures:
      raise RuntimeError('Previous action not retrieved.')
    for n, step_fn in enumerate(self._step_fns):
      bot_timestep = timestep._replace(
          observation=timestep.observation[n], reward=timestep.reward[n])
      future = self._executor.submit(step_fn, bot_timestep)
      self._action_futures.append(future)

  def await_action(self) -> Sequence[int]:
    """Waits for the population action in response to last timestep.

    Returns:
      The action for the population.

    Raises:
      RuntimeError: no timestep has been sent.
    """
    if not self._action_futures:
      raise RuntimeError('No timestep sent.')
    actions = tuple(future.result() for future in self._action_futures)
    self._action_futures.clear()
    return actions


def _restrict_observation(
    observation: Mapping[str, T],
    permitted_observations: Collection[str],
) -> Mapping[str, T]:
  """Restricts an observation to only the permitted keys."""
  return immutabledict.immutabledict({
      key: observation[key]
      for key in observation if key in permitted_observations
  })


def _restrict_observations(
    observations: Iterable[Mapping[str, T]],
    permitted_observations: Collection[str],
) -> Sequence[Mapping[str, T]]:
  """Restricts multiple observations to only the permitted keys."""
  return tuple(
      _restrict_observation(observation, permitted_observations)
      for observation in observations
  )


def _partition(
    values: Sequence[T],
    is_focal: Sequence[bool],
) -> Tuple[Sequence[T], Sequence[T]]:
  """Partitions a sequence into focal and background sequences."""
  focal_values = []
  background_values = []
  for focal, value in zip(is_focal, values):
    if focal:
      focal_values.append(value)
    else:
      background_values.append(value)
  return tuple(focal_values), tuple(background_values)


def _merge(
    focal_values: Sequence[T],
    background_values: Sequence[T],
    is_focal: Sequence[bool],
) -> Sequence[T]:
  """Merges focal and background sequences into one."""
  focal_values = iter(focal_values)
  background_values = iter(background_values)
  return tuple(
      next(focal_values if focal else background_values) for focal in is_focal
  )


@dataclasses.dataclass(frozen=True)
class PopulationObservables:
  """Observables for a population.

  Attributes:
    action: emits actions sent to the substrate by the poulation.
    timestep: emits timesteps sent from the substrate to the population.
  """
  action: rx.typing.Observable[Sequence[int]]
  timestep: rx.typing.Observable[dm_env.TimeStep]


@dataclasses.dataclass(frozen=True)
class ScenarioObservables(substrate_factory.SubstrateObservables):
  """Observables for a Scenario.

  Attributes:
    action: emits actions sent to the scenario from (focal) players.
    timestep: emits timesteps sent from the scenario to (focal) players.
    events: emits environment-specific events resulting from any interactions
      with the scenario.
    focal: observables from the perspective of the focal players.
    background: observables from the perspective of the background players.
    substrate: observables for the underlying substrate.
  """
  focal: PopulationObservables
  background: PopulationObservables
  substrate: substrate_factory.SubstrateObservables


class Scenario(base.Wrapper):
  """An substrate where a number of player slots are filled by bots."""

  def __init__(
      self,
      substrate,
      bots: Mapping[str, bot_factory.Policy],
      is_focal: Sequence[bool],
      permitted_observations: Collection[str] = PERMITTED_OBSERVATIONS,
  ) -> None:
    """Initializes the scenario.

    Args:
      substrate: the substrate to add bots to.
      bots: the bots to sample from (with replacement) each episode.
      is_focal: which player slots are allocated to focal players.
      permitted_observations: the observations exposed by the scenario to focal
        agents.
    """
    num_players = len(substrate.action_spec())
    if len(is_focal) != num_players:
      raise ValueError(f'is_focal is length {len(is_focal)} but substrate is '
                       f'{num_players}-player.')
    super().__init__(substrate)
    self._is_focal = is_focal
    self._background_population = Population(
        policies=bots, population_size=num_players - sum(is_focal))
    self._permitted_observations = frozenset(permitted_observations)

    self._focal_action_subject = subject.Subject()
    self._focal_timestep_subject = subject.Subject()
    self._background_action_subject = subject.Subject()
    self._background_timestep_subject = subject.Subject()
    self._events_subject = subject.Subject()
    focal_observables = PopulationObservables(
        action=self._focal_action_subject,
        timestep=self._focal_timestep_subject,
    )
    background_observables = PopulationObservables(
        action=self._background_action_subject,
        timestep=self._background_timestep_subject,
    )
    self._observables = ScenarioObservables(
        action=self._focal_action_subject,
        events=self._events_subject,
        timestep=self._focal_timestep_subject,
        focal=focal_observables,
        background=background_observables,
        substrate=super().observables(),
    )

  def close(self) -> None:
    """See base class."""
    self._background_population.close()
    super().close()
    self._focal_action_subject.on_completed()
    self._background_action_subject.on_completed()
    self._focal_timestep_subject.on_completed()
    self._background_timestep_subject.on_completed()
    self._events_subject.on_completed()

  def _await_full_action(self, focal_action: Sequence[int]) -> Sequence[int]:
    """Returns full action after awaiting bot actions."""
    self._focal_action_subject.on_next(focal_action)
    background_action = self._background_population.await_action()
    self._background_action_subject.on_next(background_action)
    return _merge(focal_action, background_action, self._is_focal)

  def _split_timestep(
      self, timestep: dm_env.TimeStep
  ) -> Tuple[dm_env.TimeStep, dm_env.TimeStep]:
    """Splits multiplayer timestep as needed by agents and bots."""
    focal_rewards, background_rewards = _partition(timestep.reward,
                                                   self._is_focal)
    focal_observations, background_observations = _partition(
        timestep.observation, self._is_focal)
    focal_observations = _restrict_observations(focal_observations,
                                                self._permitted_observations)
    focal_timestep = timestep._replace(
        reward=focal_rewards, observation=focal_observations)
    background_timestep = timestep._replace(
        reward=background_rewards, observation=background_observations)
    return focal_timestep, background_timestep

  def _send_full_timestep(self, timestep: dm_env.TimeStep) -> dm_env.TimeStep:
    """Returns focal timestep and sends background timestep to bots."""
    focal_timestep, background_timestep = self._split_timestep(timestep)
    self._background_timestep_subject.on_next(background_timestep)
    self._background_population.send_timestep(background_timestep)
    self._focal_timestep_subject.on_next(focal_timestep)
    return focal_timestep

  def reset(self) -> dm_env.TimeStep:
    """See base class."""
    self._background_population.reset()
    timestep = super().reset()
    focal_timestep = self._send_full_timestep(timestep)
    for event in self.events():
      self._events_subject.on_next(event)
    return focal_timestep

  def step(self, action: Sequence[int]) -> dm_env.TimeStep:
    """See base class."""
    action = self._await_full_action(focal_action=action)
    timestep = super().step(action)
    focal_timestep = self._send_full_timestep(timestep)
    for event in self.events():
      self._events_subject.on_next(event)
    return focal_timestep

  def events(self) -> Sequence[Tuple[str, Any]]:
    """See base class."""
    # Do not emit substrate events as these may not make sense in the context
    # of a scenario (e.g. player indices may have changed).
    return ()

  def action_spec(self) -> Sequence[dm_env.specs.DiscreteArray]:
    """See base class."""
    focal_action_spec, _ = _partition(super().action_spec(), self._is_focal)
    return focal_action_spec

  def observation_spec(self) -> Sequence[Mapping[str, dm_env.specs.Array]]:
    """See base class."""
    focal_observation_spec, _ = _partition(super().observation_spec(),
                                           self._is_focal)
    return _restrict_observations(focal_observation_spec,
                                  self._permitted_observations)

  def reward_spec(self) -> Sequence[dm_env.specs.Array]:
    """See base class."""
    # TODO(b/192925212): better typing to avoid pytype disables.
    reward_spec: Sequence[dm_env.specs.Array] = super().reward_spec()  # pytype: disable=annotation-type-mismatch
    focal_reward_spec, _ = _partition(reward_spec, self._is_focal)
    return focal_reward_spec

  def observables(self) -> ScenarioObservables:
    """Returns the observables for the scenario."""
    return self._observables


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
      bots=bots,
      is_focal=scenario.is_focal,
      num_players=sum(scenario.is_focal),
      num_bots=len(scenario.is_focal) - sum(scenario.is_focal),
  )
  return config.lock()


def build(config: config_dict.ConfigDict) -> Scenario:
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

  # Add observations needed by some bots. These are removed for focal players.
  substrate_observations = set(substrate.observation_spec()[0])
  substrate = all_observations_wrapper.Wrapper(
      substrate, observations_to_share=['POSITION'], share_actions=True)
  substrate = agent_slot_wrapper.Wrapper(substrate)
  if 'INVENTORY' not in substrate_observations:
    substrate = default_observation_wrapper.Wrapper(
        substrate, key='INVENTORY', default_value=np.zeros([1]))

  return Scenario(
      substrate=substrate,
      bots=bots,
      is_focal=config.is_focal,
      permitted_observations=PERMITTED_OBSERVATIONS & substrate_observations)
