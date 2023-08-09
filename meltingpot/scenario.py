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
"""Scenario factory."""

import collections
from collections.abc import Collection, Mapping
from typing import Callable, Optional

import immutabledict
from meltingpot import bot as mp_bot
from meltingpot import substrate as mp_substrate
from meltingpot.configs import scenarios as scenario_configs
from meltingpot.utils.scenarios import scenario
from meltingpot.utils.scenarios import scenario_factory
from meltingpot.utils.substrates import substrate as substrate_lib

SCENARIOS = frozenset(scenario_configs.SCENARIO_CONFIGS)

SubstrateTransform = Callable[[substrate_lib.Substrate],
                              substrate_lib.Substrate]


def _scenarios_by_substrate() -> Mapping[str, Collection[str]]:
  """Returns a mapping from substrates to their scenarios."""
  scenarios_by_substrate = collections.defaultdict(list)
  for name, config in scenario_configs.SCENARIO_CONFIGS.items():
    scenarios_by_substrate[config.substrate].append(name)
  return immutabledict.immutabledict({
      substrate: frozenset(scenarios)
      for substrate, scenarios in scenarios_by_substrate.items()
  })


SCENARIOS_BY_SUBSTRATE = _scenarios_by_substrate()

PERMITTED_OBSERVATIONS = frozenset({
    # The primary visual input.
    'RGB',
    # Extra observations used in some substrates.
    'HUNGER',
    'INVENTORY',
    'MY_OFFER',
    'OFFERS',
    'READY_TO_SHOOT',
    'STAMINA',
    'VOTING',
    # An extra observation that is never necessary but could perhaps help.
    'COLLECTIVE_REWARD'
})


def get_config(name: str) -> scenario_configs.ScenarioConfig:
  """Returns the config for the specified scenario."""
  return scenario_configs.SCENARIO_CONFIGS[name]


def build(
    name: str,
    *,
    substrate_transform: Optional[SubstrateTransform] = None,
) -> scenario.Scenario:
  """Builds an instance of the specified scenario.

  Args:
    name: the scenario.
    substrate_transform: optional transform to apply to underlying substrate.
      This is intended for training purposes and should not be used during
      evaluation. If applied, the observations will not be restricted to
      PERMITTED_OBSERVATIONS.

  Returns:
    The test scenario.
  """
  config = get_config(name)
  return build_from_config(config, substrate_transform=substrate_transform)


def build_from_config(
    config: scenario_configs.ScenarioConfig,
    *,
    substrate_transform: Optional[SubstrateTransform] = None,
) -> scenario.Scenario:
  """Builds a scenario from the provided config.

  Args:
    config: bot config
    substrate_transform: optional transform to apply to underlying substrate.
      This is intended for training purposes and should not be used during
      evaluation. If applied, the observations will not be restricted to
      PERMITTED_OBSERVATIONS.

  Returns:
    The test scenario.
  """
  factory = get_factory_from_config(config)
  if substrate_transform is None:
    return factory.build()
  else:
    return factory.build_transformed(substrate_transform)


def get_factory(name: str) -> scenario_factory.ScenarioFactory:
  """Returns the factory for the specified scenario."""
  config = scenario_configs.SCENARIO_CONFIGS[name]
  return get_factory_from_config(config)


def get_factory_from_config(
    config: scenario_configs.ScenarioConfig,
) -> scenario_factory.ScenarioFactory:
  """Returns a factory from the provided config."""
  substrate = mp_substrate.get_factory(config.substrate)
  bots = {
      name: mp_bot.get_factory(name)
      for name in set().union(*config.bots_by_role.values())
  }
  return scenario_factory.ScenarioFactory(
      substrate=substrate,
      roles=config.roles,
      bots=bots,
      bots_by_role=config.bots_by_role,
      is_focal=config.is_focal,
      permitted_observations=PERMITTED_OBSERVATIONS)
