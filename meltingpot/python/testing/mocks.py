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
from typing import Type, TypeVar
from unittest import mock

import dm_env
import immutabledict
import numpy as np
import tree

from meltingpot.python.utils.scenarios import scenario
from meltingpot.python.utils.substrates import specs as meltingpot_specs
from meltingpot.python.utils.substrates import substrate

SUBSTRATE_OBSERVATION_SPEC = immutabledict.immutabledict({
    # Observations present in all substrates. Sizes may vary.
    'RGB': meltingpot_specs.OBSERVATION['RGB'],
    'WORLD.RGB': meltingpot_specs.rgb(128, 256, name='WORLD.RGB'),
})
SCENARIO_OBSERVATION_SPEC = immutabledict.immutabledict({
    # Observations present in all scenarios.
    'RGB': dm_env.specs.Array(shape=(72, 96, 3), dtype=np.uint8),
})


def _values_from_specs(
    specs: Sequence[tree.Structure[dm_env.specs.Array]]
) -> tree.Structure[np.ndarray]:
  values = tree.map_structure(lambda spec: spec.generate_value(), specs)
  return tuple(
      tree.map_structure(lambda v, n=n: v + n, value)
      for n, value in enumerate(values))


_AnySubstrate = TypeVar('_AnySubstrate', bound=substrate.Substrate)


def _build_mock_substrate(
    *,
    spec: Type[_AnySubstrate],
    num_players: int,
    num_actions: int,
    observation_spec: Mapping[str, dm_env.specs.Array],
) -> ...:
  """Returns a mock Substrate for use in testing.

  Args:
    spec: the Substrate class to use as a spec.
    num_players: the number of players in the substrate.
    num_actions: the number of actions supported by the substrate.
    observation_spec: the observation spec for a single player.
  """
  mock_substrate = mock.create_autospec(spec=spec, instance=True, spec_set=True)
  mock_substrate.__enter__.return_value = mock_substrate
  mock_substrate.__exit__.return_value = None

  mock_substrate.observation_spec.return_value = (
      observation_spec,) * num_players
  mock_substrate.reward_spec.return_value = (
      meltingpot_specs.REWARD,) * num_players
  mock_substrate.discount_spec.return_value = meltingpot_specs.DISCOUNT
  mock_substrate.action_spec.return_value = (
      meltingpot_specs.action(num_actions),) * num_players

  mock_substrate.events.return_value = ()

  observation = _values_from_specs((observation_spec,) * num_players)
  mock_substrate.observation.return_value = observation
  mock_substrate.reset.return_value = dm_env.TimeStep(
      step_type=dm_env.StepType.FIRST,
      reward=(meltingpot_specs.REWARD.generate_value(),) * num_players,
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
      num_actions=num_actions,
      observation_spec=observation_spec)


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
      num_actions=num_actions,
      observation_spec=observation_spec)
