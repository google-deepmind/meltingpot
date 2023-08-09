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
"""Mocks of various Melting Pot classes for use in testing."""

from collections.abc import Mapping, Sequence
from typing import Optional, Type, TypeVar
from unittest import mock

import dm_env
import immutabledict
import meltingpot
from meltingpot.utils.scenarios import scenario
from meltingpot.utils.substrates import specs as meltingpot_specs
from meltingpot.utils.substrates import substrate
import numpy as np
import tree

SUBSTRATE_OBSERVATION_SPEC = immutabledict.immutabledict({
    # Observations present in all substrates. Sizes may vary.
    'RGB': meltingpot_specs.OBSERVATION['RGB'],
    'WORLD.RGB': meltingpot_specs.rgb(128, 256, name='WORLD.RGB'),
})
SCENARIO_OBSERVATION_SPEC = immutabledict.immutabledict({
    # Observations present in all scenarios.
    'RGB': meltingpot_specs.OBSERVATION['RGB'],
})


def _values_from_specs(
    specs: Sequence[tree.Structure[dm_env.specs.Array]]
) -> tree.Structure[np.ndarray]:
  values = tree.map_structure(lambda spec: spec.generate_value(), specs)
  return tuple(
      tree.map_structure(lambda v, n=n: np.full_like(v, n), value)
      for n, value in enumerate(values))


_AnySubstrate = TypeVar('_AnySubstrate', bound=substrate.Substrate)


def _build_mock_substrate(
    *,
    spec: Type[_AnySubstrate],
    num_players: int,
    timestep_spec: dm_env.TimeStep,
    action_spec: dm_env.specs.DiscreteArray,
) -> ...:
  """Returns a mock Substrate for use in testing.

  Args:
    spec: the Substrate class to use as a spec.
    num_players: the number of players in the substrate.
    timestep_spec: the timestep spec for a single player.
    action_spec: the action spec for a single player.
  """
  mock_substrate = mock.create_autospec(spec=spec, instance=True, spec_set=True)
  mock_substrate.__enter__.return_value = mock_substrate
  mock_substrate.__exit__.return_value = None

  mock_substrate.observation_spec.return_value = (
      timestep_spec.observation,) * num_players
  mock_substrate.reward_spec.return_value = (
      timestep_spec.reward,) * num_players
  mock_substrate.discount_spec.return_value = timestep_spec.discount
  mock_substrate.action_spec.return_value = (action_spec,) * num_players

  mock_substrate.events.return_value = ()

  observation = _values_from_specs(
      (timestep_spec.observation,) * num_players)
  mock_substrate.observation.return_value = observation
  mock_substrate.reset.return_value = dm_env.TimeStep(
      step_type=dm_env.StepType.FIRST,
      reward=(timestep_spec.reward.generate_value(),) * num_players,
      discount=0.,
      observation=observation,
  )
  mock_substrate.step.return_value = dm_env.transition(
      reward=tuple(float(i) for i in range(num_players)),
      observation=observation,
  )
  return mock_substrate


def build_mock_substrate(
    *,
    num_players: int = 8,
    num_actions: int = 8,
    observation_spec: Mapping[str,
                              dm_env.specs.Array] = SUBSTRATE_OBSERVATION_SPEC,
) -> ...:
  """Returns a mock Substrate for use in testing.

  Args:
    num_players: the number of players in the substrate.
    num_actions: the number of actions supported by the substrate.
    observation_spec: the observation spec for a single player.
  """
  return _build_mock_substrate(
      spec=substrate.Substrate,
      num_players=num_players,
      action_spec=meltingpot_specs.action(num_actions),
      timestep_spec=meltingpot_specs.timestep(observation_spec),
  )


def build_mock_substrate_like(name: str, *,
                              num_players: Optional[int] = None) -> ...:
  """Returns a mock of a specific Substrate for use in testing.

  Args:
    name: substrate to mock.
    num_players: number of players to support.
  """
  factory = meltingpot.substrate.get_factory(name)
  if num_players is None:
    num_players = len(factory.default_player_roles())
  return _build_mock_substrate(
      spec=substrate.Substrate,
      num_players=num_players,
      action_spec=factory.action_spec(),
      timestep_spec=factory.timestep_spec(),
  )


def build_mock_scenario(
    *,
    num_players: int = 8,
    num_actions: int = 8,
    observation_spec: Mapping[str,
                              dm_env.specs.Array] = SCENARIO_OBSERVATION_SPEC,
) -> ...:
  """Returns a mock Scenario for use in testing.

  Args:
    num_players: the number of focal players in the scenario.
    num_actions: the number of actions supported by the scenario.
    observation_spec: the observation spec for a single focal player.
  """
  return _build_mock_substrate(
      spec=scenario.Scenario,
      num_players=num_players,
      action_spec=meltingpot_specs.action(num_actions),
      timestep_spec=meltingpot_specs.timestep(observation_spec),
  )


def build_mock_scenario_like(name: str) -> ...:
  """Returns a mock of a specific Scenario for use in testing.

  Args:
    name: scenario to mock.
  """
  factory = meltingpot.scenario.get_factory(name)
  return _build_mock_substrate(
      spec=scenario.Scenario,
      num_players=factory.num_focal_players(),
      action_spec=factory.action_spec(),
      timestep_spec=factory.timestep_spec(),
  )
