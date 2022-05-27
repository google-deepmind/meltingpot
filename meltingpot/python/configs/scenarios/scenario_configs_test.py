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
"""Tests of the scenario configs."""

import collections

from absl.testing import absltest
from absl.testing import parameterized

from meltingpot.python.configs import bots
from meltingpot.python.configs import scenarios
from meltingpot.python.configs import substrates

SCENARIO_CONFIGS = scenarios.SCENARIO_CONFIGS
AVAILABLE_BOTS = frozenset(bots.BOT_CONFIGS)
AVAILABLE_SUBSTRATES = frozenset(substrates.SUBSTRATES)


def _is_compatible(bot_name, substrate):
  bot_config = bots.BOT_CONFIGS[bot_name]
  return substrate == bot_config.substrate


class ScenarioConfigTest(parameterized.TestCase):

  @parameterized.named_parameters(SCENARIO_CONFIGS.items())
  def test_has_description(self, scenario):
    self.assertNotEmpty(scenario.description)

  @parameterized.named_parameters(SCENARIO_CONFIGS.items())
  def test_has_tags(self, scenario):
    self.assertNotEmpty(scenario.tags)

  @parameterized.named_parameters(SCENARIO_CONFIGS.items())
  def test_has_valid_substrate(self, scenario):
    self.assertIn(scenario.substrate, AVAILABLE_SUBSTRATES)

  @parameterized.named_parameters(
      (name, name, scenario) for name, scenario in SCENARIO_CONFIGS.items())
  def test_name_starts_with_substrate_name(self, name, scenario):
    self.assertStartsWith(name, scenario.substrate)

  @parameterized.named_parameters(SCENARIO_CONFIGS.items())
  def test_has_focal_players(self, scenario):
    self.assertTrue(any(scenario.is_focal))

  @parameterized.named_parameters(SCENARIO_CONFIGS.items())
  def test_has_valid_sizes(self, scenario):
    substrate = substrates.get_config(scenario.substrate)
    self.assertLen(scenario.is_focal, substrate.num_players)

  @parameterized.named_parameters(SCENARIO_CONFIGS.items())
  def test_has_valid_bots(self, scenario):
    self.assertContainsSubset(scenario.bots, AVAILABLE_BOTS)

  @parameterized.named_parameters(SCENARIO_CONFIGS.items())
  def test_bots_compatible(self, scenario):
    incompatible = {
        bot_name for bot_name in scenario.bots
        if not _is_compatible(bot_name, scenario.substrate)
    }
    self.assertEmpty(
        incompatible,
        f'Substrate {scenario.substrate!r} not supported by: {incompatible!r}.')

  def test_no_duplicates(self):
    seen = collections.defaultdict(set)
    for name, config in SCENARIO_CONFIGS.items():
      seen[config].add(name)
    duplicates = {names for _, names in seen.items() if len(names) > 1}
    self.assertEmpty(duplicates, f'Duplicate configs found: {duplicates!r}.')

  def test_all_substrates_used_by_scenarios(self):
    used = {scenario.substrate for scenario in SCENARIO_CONFIGS.values()}
    unused = AVAILABLE_SUBSTRATES - used
    self.assertEmpty(unused, f'Substrates not used by any scenario: {unused!r}')

  def test_all_bots_used_by_scenarios(self):
    used = set().union(
        *(scenario.bots for scenario in SCENARIO_CONFIGS.values()))
    unused = AVAILABLE_BOTS - used
    self.assertEmpty(unused, f'Bots not used by any scenario: {unused!r}')

if __name__ == '__main__':
  absltest.main()
