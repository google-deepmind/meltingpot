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
"""Tests of the configs."""

import os
import re

from absl.testing import absltest
from absl.testing import parameterized

from meltingpot.python.configs import bots as bot_configs
from meltingpot.python.configs import scenarios as scenario_configs
from meltingpot.python.configs import substrates as substrate_configs

_MODELS_ROOT = re.sub('meltingpot/python/.*', 'meltingpot/assets/saved_models',
                      __file__)


def _subdirs(root):
  for file in os.listdir(root):
    if os.path.isdir(os.path.join(root, file)):
      yield file


def _models(models_root=_MODELS_ROOT):
  for substrate in _subdirs(models_root):
    for model in _subdirs(os.path.join(models_root, substrate)):
      yield models_root, substrate, model


class ConfigTest(parameterized.TestCase):

  @parameterized.named_parameters(scenario_configs.SCENARIO_CONFIGS.items())
  def test_scenario_has_valid_bots(self, scenario):
    self.assertContainsSubset(scenario.bots, bot_configs.BOT_CONFIGS)

  @parameterized.named_parameters(scenario_configs.SCENARIO_CONFIGS.items())
  def test_scenario_has_valid_substrate(self, scenario):
    self.assertIn(scenario.substrate, substrate_configs.SUBSTRATES)

  @parameterized.named_parameters(scenario_configs.SCENARIO_CONFIGS.items())
  def test_scenario_has_focal_players(self, scenario):
    self.assertTrue(any(scenario.is_focal))

  @parameterized.named_parameters(scenario_configs.SCENARIO_CONFIGS.items())
  def test_scenario_has_valid_sizes(self, scenario):
    substrate = substrate_configs.get_config(scenario.substrate)
    self.assertLen(scenario.is_focal, substrate.num_players)

  @parameterized.named_parameters(
      # TODO(b/220870875): remove disable when fixed.
      (name, name, scenario)  # pylint: disable=undefined-variable
      for name, scenario in scenario_configs.SCENARIO_CONFIGS.items())
  def test_scenario_name_starts_with_substrate_name(self, name, scenario):
    self.assertStartsWith(name, scenario.substrate)

  @parameterized.named_parameters(
      # TODO(b/220870875): remove disable when fixed.
      (substrate, substrate)  # pylint: disable=undefined-variable
      for substrate in substrate_configs.SUBSTRATES)
  def test_substrate_used_by_scenario(self, substrate):
    used = any(substrate == scenario.substrate
               for scenario in scenario_configs.SCENARIO_CONFIGS.values())
    self.assertTrue(used, f'{substrate} is not covered by any scenario.')

  @parameterized.named_parameters(
      # TODO(b/220870875): remove disable when fixed.
      (bot, bot)   # pylint: disable=undefined-variable
      for bot in bot_configs.BOT_CONFIGS)
  def test_bot_used_by_scenario(self, bot):
    used = any(bot in scenario.bots
               for scenario in scenario_configs.SCENARIO_CONFIGS.values())
    self.assertTrue(used, f'{bot} is not used in any scenario.')

  @parameterized.named_parameters(bot_configs.BOT_CONFIGS.items())
  def test_bot_model_exists(self, bot):
    self.assertTrue(
        os.path.isdir(bot.model_path), f'Missing model {bot.model_path!r}.')

  @parameterized.named_parameters(bot_configs.BOT_CONFIGS.items())
  def test_bot_substrate_matches_model(self, bot):
    substrate = os.path.basename(os.path.dirname(bot.model_path))
    self.assertEqual(bot.substrate, substrate,
                     f'{bot} substrate does not match model path.')

  @parameterized.named_parameters(
      # TODO(b/220870875): remove disable when fixed.
      (f'{substrate}_{model}', os.path.join(models_root, substrate, model))  # pylint: disable=undefined-variable
      for models_root, substrate, model in _models())
  def test_model_used_by_bot(self, model_path):
    used = any(bot.model_path == model_path
               for bot in bot_configs.BOT_CONFIGS.values())
    self.assertTrue(used, f'Model {model_path} not used by any bot.')


if __name__ == '__main__':
  absltest.main()
