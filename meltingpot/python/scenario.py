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

from typing import Collection, Mapping

from ml_collections import config_dict

from meltingpot.python import bot as bot_factory
from meltingpot.python import substrate as substrate_factory
from meltingpot.python.configs import scenarios as scenario_config
from meltingpot.python.utils.scenarios import population
from meltingpot.python.utils.scenarios import scenario as scenario_lib
from meltingpot.python.utils.scenarios import substrate_transforms

AVAILABLE_SCENARIOS = frozenset(scenario_config.SCENARIO_CONFIGS)

SCENARIOS_BY_SUBSTRATE: Mapping[
    str, Collection[str]] = scenario_config.scenarios_by_substrate(
        scenario_config.SCENARIO_CONFIGS)

PERMITTED_OBSERVATIONS = frozenset({
    'INVENTORY',
    'READY_TO_SHOOT',
    'RGB',
})

# TODO(b/227143834): Remove aliases once internal deps have been removed.
Scenario = scenario_lib.Scenario
PopulationObservables = population.PopulationObservables
ScenarioObservables = scenario_lib.ScenarioObservables
Population = population.Population


def get_config(scenario_name: str) -> config_dict.ConfigDict:
  """Returns a config for the specified scenario.

  Args:
    scenario_name: Name of the scenario. Must be in AVAILABLE_SCENARIOS.
  """
  if scenario_name not in AVAILABLE_SCENARIOS:
    raise ValueError(f'Unknown scenario {scenario_name!r}')
  scenario = scenario_config.SCENARIO_CONFIGS[scenario_name]
  substrate = substrate_factory.get_config(scenario.substrate)
  bots = {name: bot_factory.get_config(name) for name in scenario.bots}
  config = config_dict.create(
      substrate=substrate,
      bots=bots,
      is_focal=scenario.is_focal,
      num_players=sum(scenario.is_focal),
      num_bots=len(scenario.is_focal) - sum(scenario.is_focal),
      substrate_transform=None,
      permitted_observations=set(PERMITTED_OBSERVATIONS),
  )
  return config.lock()


def build(config: config_dict.ConfigDict) -> scenario_lib.Scenario:
  """Builds a scenario for the given config.

  Args:
    config: config resulting from `get_config`.

  Returns:
    The test scenario.
  """
  substrate = substrate_factory.build(config.substrate)
  if config.substrate_transform:
    substrate = config.substrate_transform(substrate)
  permitted_observations = set(substrate.observation_spec()[0])
  if not config.substrate_transform:
    permitted_observations &= config.permitted_observations
  # Add observations needed by some bots. These are removed for focal players.
  substrate = substrate_transforms.with_tf1_bot_required_observations(substrate)

  background_population = population.Population(
      policies={
          bot_name: bot_factory.build(bot_config)
          for bot_name, bot_config in config.bots.items()
      },
      population_size=config.num_bots,
  )

  return scenario_lib.Scenario(
      substrate=substrate,
      background_population=background_population,
      is_focal=config.is_focal,
      permitted_observations=permitted_observations)
