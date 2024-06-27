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
"""Tests of the scenario configs."""

import collections

from absl.testing import absltest
from absl.testing import parameterized
from meltingpot import bot as bot_factory
from meltingpot.configs import bots
from meltingpot.configs import scenarios
from meltingpot.configs import substrates

SCENARIO_CONFIGS = scenarios.SCENARIO_CONFIGS
AVAILABLE_BOTS = bot_factory.BOTS
AVAILABLE_SUBSTRATES = frozenset(substrates.SUBSTRATES)


def _is_compatible(bot_name, substrate, role):
  if bot_name == bot_factory.NOOP_BOT_NAME:
    return True
  bot_config = bots.BOT_CONFIGS[bot_name]
  return substrate == bot_config.substrate and role in bot_config.roles


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
  def test_has_matching_sizes(self, scenario):
    self.assertLen(scenario.is_focal, len(scenario.roles))

  @parameterized.named_parameters(SCENARIO_CONFIGS.items())
  def test_has_valid_roles(self, scenario):
    valid_roles = substrates.get_config(scenario.substrate).valid_roles
    self.assertContainsSubset(scenario.roles, valid_roles)

  @parameterized.named_parameters(SCENARIO_CONFIGS.items())
  def test_has_valid_bots(self, scenario):
    scenario_bots = set().union(*scenario.bots_by_role.values())
    self.assertContainsSubset(scenario_bots, AVAILABLE_BOTS)

  @parameterized.named_parameters(SCENARIO_CONFIGS.items())
  def test_bots_compatible(self, scenario):
    for role, bot_names in scenario.bots_by_role.items():
      incompatible = {
          bot_name for bot_name in bot_names
          if not _is_compatible(bot_name, scenario.substrate, role)
      }
      with self.subTest(role):
        self.assertEmpty(
            incompatible,
            f'Substrate {scenario.substrate!r}, role {role!r} not supported '
            f'by: {incompatible!r}.')

  @parameterized.named_parameters(SCENARIO_CONFIGS.items())
  def test_no_missing_role_assigments(self, scenario):
    background_roles = set(role for n, role in enumerate(scenario.roles)
                           if not scenario.is_focal[n])
    supported_roles = {
        role for role, bots in scenario.bots_by_role.items() if bots}
    unsupported_roles = background_roles - supported_roles
    self.assertEmpty(unsupported_roles,
                     f'Background roles {unsupported_roles!r} have not been '
                     f'assigned bots.')

  @parameterized.named_parameters(SCENARIO_CONFIGS.items())
  def test_no_unused_role_assignments(self, scenario):
    background_roles = set(role for n, role in enumerate(scenario.roles)
                           if not scenario.is_focal[n])
    redundant_roles = set(scenario.bots_by_role) - background_roles
    self.assertEmpty(redundant_roles,
                     f'Bots assigned to {redundant_roles!r} are unused.')

  def test_no_duplicates(self):
    seen = collections.defaultdict(set)
    for name, config in SCENARIO_CONFIGS.items():
      seen[config].add(name)
    duplicates = tuple(names for _, names in seen.items() if len(names) > 1)
    self.assertEqual(duplicates,
                     tuple([{'territory__rooms_5', 'territory__rooms_6'}]),
                     'The only acceptable duplicates are territory__rooms_5 '
                     f'and territory__rooms_6. Found: {duplicates!r}.')
    # self.assertEmpty(duplicates, f'Duplicate configs found: {duplicates!r}.')

  def test_all_substrates_used_by_scenarios(self):
    used = {scenario.substrate for scenario in SCENARIO_CONFIGS.values()}
    unused = AVAILABLE_SUBSTRATES - used
    self.assertEmpty(unused, f'Substrates not used by any scenario: {unused!r}')

  def test_all_bots_used_by_scenarios(self):
    used = set()
    for scenario in SCENARIO_CONFIGS.values():
      used.update(*scenario.bots_by_role.values())
    unused = AVAILABLE_BOTS - used
    self.assertEmpty(unused, f'Bots not used by any scenario: {unused!r}')


if __name__ == '__main__':
  absltest.main()
