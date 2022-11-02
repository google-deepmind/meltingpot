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
"""Let agents observe the actions, observations and rewards of others.

Requirements on the environment that is wrapped:
  - The original environment must provide action and observation specs as
    dictionaries.
  - The action spec is the same for all players. The action spec returned by the
    environment is the spec for a single player. Actions provided to the
    environment should be a list of length num_players with this format.

After wrapping:
  - the action spec will remain unchanged
  - all of the original observations and observation names will remain unchanged
  - new observations corresponding to the actions, observations and rewards of
    all players will be present under the observation key 'global'

Keys used for the additional observations:
  - rewards of all players will appear at observations['global']['rewards']
  - actions of all players will appear at
    observations['global']['actions'][<name>],
    where <name> is the name of the action in the original action spec.
    Note: if the action spec defines more than one action, then each will be
    shared under its own name.
  - observations of all players will appear at
    observations['global']['observations'][<name>]
    where <name> is the name of the observation in the original observation
    spec.
    Note: if the observation spec defines more than one observation, then each
    will be shared under its own name.

Note: shared actions, rewards and observations are provided in the same timestep
as the original, single-player versions:
- An agent's individual reward at a given timestep is included in all_rewards at
  that same timestep.
- An agent's individual observation at a given timestep is included in
  all_observations_ at that same timestep.
- The actions an agent provides to step() are included in the observations
  immediately returned from step().
"""

from typing import Any, Collection, Mapping, Sequence, Union

import dm_env
import immutabledict
import numpy as np

from meltingpot.python.utils.scenarios.wrappers import base
from meltingpot.python.utils.substrates import substrate

GLOBAL_KEY = 'global'
OBSERVATIONS_KEY = 'observations'
REWARDS_KEY = 'rewards'
ACTIONS_KEY = 'actions'


def _immutable_ndarray(value: np.ndarray) -> np.ndarray:
  value.setflags(write=False)
  return value


class Wrapper(base.SubstrateWrapper):
  """Exposes actions/observations/rewards from all players to all players."""

  def __init__(self, env: substrate.Substrate,
               observations_to_share: Collection[str] = (),
               share_actions: bool = False,
               share_rewards: bool = False) -> None:
    """Wraps an environment.

    Args:
      env: environment to wrap. When this wrapper closes env will also be
        closed.
      observations_to_share: observation keys to share with other players.
      share_actions: whether to show other players actions.
      share_rewards: whether to show other players rewards.
    """
    super().__init__(env)
    self._observations_to_share = observations_to_share
    self._share_actions = share_actions
    self._share_rewards = share_rewards

    action_spec = super().action_spec()
    self._num_players = len(action_spec)
    self._missing_actions = [spec.generate_value() for spec in action_spec]
    self._action_dtype = action_spec[0].dtype

  def _shared_observation(
      self,
      observations: Sequence[Mapping[str, Any]],
      rewards: Sequence[Union[float, np.ndarray]],
      actions: Sequence[int]):
    """Returns shared observations."""
    # We assume that this comes from this wrapper and so all shared observations
    # are the same for all players.
    shared_observation = dict(observations[0].get(GLOBAL_KEY, {}))

    additional_observations = immutabledict.immutabledict({
        name: _immutable_ndarray(np.stack([obs[name] for obs in observations]))
        for name in self._observations_to_share
    })
    if additional_observations:
      shared_observation[OBSERVATIONS_KEY] = immutabledict.immutabledict(
          shared_observation.get(OBSERVATIONS_KEY, {}),
          **additional_observations)

    if self._share_rewards:
      shared_observation[REWARDS_KEY] = _immutable_ndarray(np.stack(rewards))

    if self._share_actions:
      shared_observation[ACTIONS_KEY] = _immutable_ndarray(
          np.array(actions, dtype=self._action_dtype))

    return immutabledict.immutabledict(shared_observation)

  def _adjusted_timestep(self, timestep: dm_env.TimeStep,
                         actions: Sequence[int]) -> dm_env.TimeStep:
    """Returns timestep with shared observations."""
    shared_observation = self._shared_observation(
        observations=timestep.observation,
        rewards=timestep.reward,
        actions=actions)
    if not shared_observation:
      return timestep
    observations = tuple(
        immutabledict.immutabledict(obs, **{GLOBAL_KEY: shared_observation})
        for obs in timestep.observation)
    return timestep._replace(observation=observations)

  def reset(self) -> dm_env.TimeStep:
    """See base class."""
    timestep = super().reset()
    return self._adjusted_timestep(timestep, self._missing_actions)

  def step(self, actions: Sequence[int]) -> dm_env.TimeStep:
    """See base class."""
    timestep = super().step(actions)
    return self._adjusted_timestep(timestep, actions)

  def _shared_observation_spec(
      self,
      observation_spec: Mapping[str, Any],
      reward_spec: dm_env.specs.Array,
      action_spec: dm_env.specs.DiscreteArray):
    """Returns spec of shared observations."""
    shared_observation_spec = dict(observation_spec.get(GLOBAL_KEY, {}))

    additional_spec = {}
    for name in self._observations_to_share:
      spec = observation_spec[name]
      additional_spec[name] = spec.replace(
          shape=(self._num_players,) + spec.shape, name=name)
    if additional_spec:
      shared_observation_spec[OBSERVATIONS_KEY] = immutabledict.immutabledict(
          shared_observation_spec.get(OBSERVATIONS_KEY, {}), **additional_spec)

    if self._share_rewards:
      shared_observation_spec[REWARDS_KEY] = reward_spec.replace(
          shape=(self._num_players,), name=REWARDS_KEY)

    if self._share_actions:
      shared_observation_spec[ACTIONS_KEY] = dm_env.specs.BoundedArray(
          shape=(self._num_players,),
          dtype=action_spec.dtype,
          minimum=action_spec.minimum,
          maximum=action_spec.maximum,
          name=ACTIONS_KEY)

    return immutabledict.immutabledict(shared_observation_spec)

  def observation_spec(self):
    """See base class."""
    observation_spec = super().observation_spec()
    assert all(spec == observation_spec[0] for spec in observation_spec)
    observation_spec = observation_spec[0]

    action_spec = super().action_spec()
    assert all(spec == action_spec[0] for spec in action_spec)
    action_spec = action_spec[0]

    reward_spec = super().reward_spec()
    assert all(spec == reward_spec[0] for spec in reward_spec)
    reward_spec = reward_spec[0]

    shared_observation_spec = self._shared_observation_spec(
        observation_spec=observation_spec,
        reward_spec=reward_spec,
        action_spec=action_spec)
    observation_spec = immutabledict.immutabledict(
        observation_spec, **{GLOBAL_KEY: shared_observation_spec})
    return (observation_spec,) * self._num_players
