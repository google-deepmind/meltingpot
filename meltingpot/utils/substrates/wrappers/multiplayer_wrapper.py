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
"""Wrapper that converts the DMLab2D specs into lists of action/observation."""

from collections.abc import Collection, Iterator, Mapping, Sequence
from typing import TypeVar

import dm_env
from meltingpot.utils.substrates.wrappers import observables
import numpy as np

T = TypeVar("T")


def _player_observations(observations: Mapping[str, T], suffix: str,
                         num_players: int) -> Iterator[T]:
  """Yields observations for each player.

  Args:
    observations: dmlab2d observations source to check.
    suffix: suffix of player key to return.
    num_players: the number of players.
  """
  for player_index in range(num_players):
    try:
      value = observations[f"{player_index + 1}.{suffix}"]
    except KeyError:
      pass
    else:
      if isinstance(value, dm_env.specs.Array):
        value = value.replace(name=suffix)
      yield player_index, value


class Wrapper(observables.ObservableLab2dWrapper):
  """Wrapper that converts the environment to multiplayer lists.

  Ensures:
  -   observations are returned as lists of dictionary observations
  -   rewards are returned as lists of scalars
  -   actions are received as lists of dictionary observations
  -   discounts are never None
  """

  def __init__(self, env,
               individual_observation_names: Collection[str],
               global_observation_names: Collection[str]):
    """Constructor.

    Args:
      env: environment to wrap. When this wrapper closes env will also be
        closed.
      individual_observation_names: the per-player observations to make
        available to the players.
      global_observation_names: the observations that are available to all
        players and analytics.
    """
    super().__init__(env)
    self._num_players = self._get_num_players()
    self._individual_observation_suffixes = set(individual_observation_names)
    self._global_observation_names = set(global_observation_names)

  def _get_num_players(self) -> int:
    """Returns maximum player index in dmlab2d action spec."""
    action_spec_keys = super().action_spec().keys()
    lua_player_indices = (int(key.split(".", 1)[0]) for key in action_spec_keys)
    return max(lua_player_indices)

  def _get_observations(
      self, source: Mapping[str, T]) -> Sequence[Mapping[str, T]]:
    """Returns multiplayer observations from dmlab2d observations.

    Args:
      source: dmlab2d observations source to check.
    """
    player_observations = [{} for i in range(self._num_players)]
    for suffix in self._individual_observation_suffixes:
      for i, value in _player_observations(source, suffix, self._num_players):
        player_observations[i][suffix] = value
    for name in self._global_observation_names:
      value = source[name]
      for i in range(self._num_players):
        player_observations[i][name] = value
    return player_observations

  def _get_rewards(self, source: Mapping[str, T]) -> Sequence[T]:
    """Returns multiplayer rewards from dmlab2d observations.

    Args:
      source: dmlab2d observations source to check.
    """
    rewards = [None] * self._num_players
    for i, value in _player_observations(source, "REWARD", self._num_players):
      rewards[i] = value
    return rewards

  def _get_timestep(self, source: dm_env.TimeStep) -> dm_env.TimeStep:
    """Returns multiplayer timestep from dmlab2d observations.

    Args:
      source: dmlab2d observations source to check.
    """
    return dm_env.TimeStep(
        step_type=source.step_type,
        reward=self._get_rewards(source.observation),
        discount=0. if source.discount is None else source.discount,
        observation=self._get_observations(source.observation))

  def _get_action(self, source: Sequence[Mapping[str, T]]) -> Mapping[str, T]:
    """Returns dmlab2 action from multiplayer actions.

    Args:
      source: multiplayer actions.
    """
    dmlab2d_actions = {}
    for player_index, action in enumerate(source):
      for key, value in action.items():
        dmlab2d_actions[f"{player_index + 1}.{key}"] = value
    return dmlab2d_actions

  def reset(self) -> dm_env.TimeStep:
    """See base class."""
    timestep = super().reset()
    return self._get_timestep(timestep)

  def step(
      self, actions: Sequence[Mapping[str, np.ndarray]]) -> dm_env.TimeStep:
    """See base class."""
    action = self._get_action(actions)
    timestep = super().step(action)
    return self._get_timestep(timestep)

  def observation(self) -> Sequence[Mapping[str, np.ndarray]]:
    """See base class."""
    observation = super().observation()
    return self._get_observations(observation)

  def action_spec(self) -> Sequence[Mapping[str, dm_env.specs.Array]]:
    """See base class."""
    source = super().action_spec()
    action_spec = [{} for _ in range(self._num_players)]
    for key, spec in source.items():
      lua_player_index, suffix = key.split(".", 1)
      player_index = int(lua_player_index) - 1
      action_spec[player_index][suffix] = spec.replace(name=suffix)
    return action_spec

  def observation_spec(self) -> Sequence[Mapping[str, dm_env.specs.Array]]:
    """See base class."""
    source = super().observation_spec()
    return self._get_observations(source)

  def reward_spec(self) -> Sequence[dm_env.specs.Array]:
    """See base class."""
    source = super().observation_spec()
    return self._get_rewards(source)
