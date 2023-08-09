# Copyright 2022 DeepMind Technologies Limited.
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
"""Substrate factory."""

from collections.abc import Collection, Mapping, Sequence, Set
from typing import Callable

import dm_env
from meltingpot.utils.substrates import builder
from meltingpot.utils.substrates import substrate


class SubstrateFactory:
  """Factory for building specific substrates."""

  def __init__(
      self,
      *,
      lab2d_settings_builder: Callable[[Sequence[str]], builder.Settings],
      individual_observations: Collection[str],
      global_observations: Collection[str],
      action_table: Sequence[Mapping[str, int]],
      timestep_spec: dm_env.TimeStep,
      action_spec: dm_env.specs.DiscreteArray,
      valid_roles: Collection[str],
      default_player_roles: Sequence[str],
  ) -> None:
    """Initializes the factory.

    Args:
      lab2d_settings_builder: callable that takes a sequence of player roles and
        returns the lab2d settings for the substrate.
      individual_observations: names of the player-specific observations to make
        available to each player.
      global_observations: names of the dmlab2d observations to make available
        to all players.
      action_table: the possible actions. action_table[i] defines the dmlab2d
        action that will be forwarded to the wrapped dmlab2d environment for the
        discrete Melting Pot action i.
      timestep_spec: spec of timestep sent to a single player.
      action_spec: spec of action expected from a single player.
      valid_roles: player roles the substrate supports.
      default_player_roles: the default player roles vector that should be used
        for training.
    """
    self._lab2d_settings_builder = lab2d_settings_builder
    self._individual_observations = frozenset(individual_observations)
    self._global_observations = frozenset(global_observations)
    self._action_table = tuple(dict(row) for row in action_table)
    self._timestep_spec = timestep_spec
    self._action_spec = action_spec
    self._valid_roles = frozenset(valid_roles)
    self._default_player_roles = tuple(default_player_roles)

  def valid_roles(self) -> Set[str]:
    """Returns the roles the substrate supports."""
    return self._valid_roles

  def default_player_roles(self) -> Sequence[str]:
    """Returns the player roles used by scenarios."""
    return self._default_player_roles

  def timestep_spec(self) -> dm_env.TimeStep:
    """Returns spec of timestep sent to a single player."""
    return self._timestep_spec

  def action_spec(self) -> dm_env.specs.DiscreteArray:
    """Returns spec of action expected from a single player."""
    return self._action_spec

  def build(self, roles: Sequence[str]) -> substrate.Substrate:
    """Builds the substrate.

    Args:
      roles: the role each player will take.

    Returns:
      The constructed substrate.
    """
    return substrate.build_substrate(
        lab2d_settings=self._lab2d_settings_builder(roles),
        individual_observations=self._individual_observations,
        global_observations=self._global_observations,
        action_table=self._action_table)
