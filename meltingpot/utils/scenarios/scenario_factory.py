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
"""Factory class for building scenarios."""

from collections.abc import Collection, Mapping, Sequence
from typing import Callable, Optional

import dm_env
import immutabledict
from meltingpot.utils.policies import policy_factory
from meltingpot.utils.scenarios import scenario as scenario_lib
from meltingpot.utils.substrates import substrate as substrate_lib
from meltingpot.utils.substrates import substrate_factory

SubstrateTransform = Callable[[substrate_lib.Substrate],
                              substrate_lib.Substrate]


class ScenarioFactory:
  """Constructs populations of bots."""

  def __init__(
      self,
      *,
      substrate: substrate_factory.SubstrateFactory,
      bots: Mapping[str, policy_factory.PolicyFactory],
      bots_by_role: Mapping[str, Collection[str]],
      roles: Sequence[str],
      is_focal: Sequence[bool],
      permitted_observations: Collection[str],
  ) -> None:
    """Initializes the instance.

    Args:
      substrate: the factory for the substrate underlying the scenario.
      bots: the factory for the policies underlying the background population.
      bots_by_role: dict mapping role to bot names that can fill it.
      roles: specifies which role should fill the corresponding player slot.
      is_focal: which player slots are allocated to focal players.
      permitted_observations: the substrate observation keys permitted to be
        exposed by the scenario to focal agents.
    """
    if len(roles) != len(is_focal):
      raise ValueError('roles and is_focal must be the same length')
    self._substrate = substrate
    self._bots = immutabledict.immutabledict(bots)
    self._roles = tuple(roles)
    self._is_focal = tuple(is_focal)
    self._bots_by_role = immutabledict.immutabledict({
        role: tuple(bots) for role, bots in bots_by_role.items()
    })
    self._permitted_observations = frozenset(permitted_observations)

  def num_focal_players(self) -> int:
    """Returns the number of players the scenario supports."""
    return sum(self._is_focal)

  def focal_player_roles(self) -> Sequence[str]:
    """Returns the roles of the focal players."""
    return tuple(
        role for n, role in enumerate(self._roles) if self._is_focal[n])

  def timestep_spec(self) -> dm_env.TimeStep:
    """Returns spec of timestep sent to a single focal player."""
    substrate_timestep_spec = self._substrate.timestep_spec()
    substrate_observation_spec = substrate_timestep_spec.observation
    focal_observation_spec = immutabledict.immutabledict({
        key: spec for key, spec in substrate_observation_spec.items()
        if key in self._permitted_observations
    })
    return substrate_timestep_spec._replace(observation=focal_observation_spec)

  def action_spec(self) -> dm_env.specs.DiscreteArray:
    """Returns spec of action expected from a single focal player."""
    return self._substrate.action_spec()

  def build(self) -> scenario_lib.Scenario:
    """Builds the scenario.

    Returns:
      The constructed scenario.
    """
    return scenario_lib.build_scenario(
        substrate=self._substrate.build(self._roles),
        bots={name: factory.build() for name, factory in self._bots.items()},
        bots_by_role=self._bots_by_role,
        roles=self._roles,
        is_focal=self._is_focal,
        permitted_observations=self._permitted_observations)

  def build_transformed(
      self, substrate_transform: Optional[SubstrateTransform] = None
  ) -> scenario_lib.Scenario:
    """Builds the scenario with a transformed substrate.

    This method is designed to allow the addition of a wrapper to the underlying
    substrate for training purposes. It should not be used during evaluation.

    The observations will be unrestricted, and the timestep spec of the returned
    scenario will not match self.timestep_spec().

    Args:
      substrate_transform: transform to apply to underlying substrate.

    Returns:
      The constructed scenario.
    """
    substrate = self._substrate.build(self._roles)
    if substrate_transform:
      substrate = substrate_transform(substrate)
    all_observations = frozenset().union(*substrate.observation_spec())
    return scenario_lib.build_scenario(
        substrate=substrate,
        bots={name: factory.build() for name, factory in self._bots.items()},
        bots_by_role=self._bots_by_role,
        roles=self._roles,
        is_focal=self._is_focal,
        permitted_observations=all_observations)
