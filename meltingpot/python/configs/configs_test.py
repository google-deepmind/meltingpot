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


def _model_path(bot):
  return os.path.join(_MODELS_ROOT, bot.substrate, bot.model)


def _subdirs(root):
  for file in os.listdir(root):
    if os.path.isdir(os.path.join(root, file)):
      yield file


def _models():
  for substrate in _subdirs(_MODELS_ROOT):
    for model in _subdirs(os.path.join(_MODELS_ROOT, substrate)):
      yield substrate, model


class ConfigTest(parameterized.TestCase):

  @parameterized.named_parameters(scenario_configs.SCENARIOS.items())
  def test_scenario_has_valid_bots(self, scenario):
    self.assertContainsSubset(scenario.bots, bot_configs.BOTS)

  @parameterized.named_parameters(scenario_configs.SCENARIOS.items())
  def test_scenario_has_valid_substrate(self, scenario):
    self.assertIn(scenario.substrate, substrate_configs.SUBSTRATES)

  @parameterized.named_parameters(scenario_configs.SCENARIOS.items())
  def test_scenario_has_focal_players(self, scenario):
    self.assertTrue(any(scenario.is_focal))

  @parameterized.named_parameters(scenario_configs.SCENARIOS.items())
  def test_scenario_has_valid_sizes(self, scenario):
    substrate = substrate_configs.get_config(scenario.substrate)
    self.assertLen(scenario.is_focal, substrate.num_players)

  @parameterized.named_parameters(
      # TODO(b/220870875): remove disable when fixed.
      (name, name, scenario)  # pylint: disable=undefined-variable
      for name, scenario in scenario_configs.SCENARIOS.items())
  def test_scenario_name_starts_with_substrate_name(self, name, scenario):
    self.assertStartsWith(name, scenario.substrate)

  @parameterized.named_parameters(
      # TODO(b/220870875): remove disable when fixed.
      (substrate, substrate)  # pylint: disable=undefined-variable
      for substrate in substrate_configs.SUBSTRATES)
  def test_substrate_used_by_scenario(self, substrate):
    used = any(substrate == scenario.substrate
               for scenario in scenario_configs.SCENARIOS.values())
    self.assertTrue(used, f'{substrate} is not covered by any scenario.')

  @parameterized.named_parameters(
      # TODO(b/220870875): remove disable when fixed.
      (bot, bot)   # pylint: disable=undefined-variable
      for bot in bot_configs.BOTS)
  def test_bot_used_by_scenario(self, bot):
    used = any(bot in scenario.bots
               for scenario in scenario_configs.SCENARIOS.values())
    self.assertTrue(used, f'{bot} is not used in any scenario.')

  @parameterized.named_parameters(bot_configs.BOTS.items())
  def test_bot_model_exists(self, bot):
    model_path = _model_path(bot)
    self.assertTrue(
        os.path.isdir(model_path), 'Missing model {bot.substrate}/{bot.model}.')

  @parameterized.named_parameters(
      # TODO(b/220870875): remove disable when fixed.
      (f'{substrate}_{model}', substrate, model)  # pylint: disable=undefined-variable
      for substrate, model in _models())
  def test_asset_used_by_bot(self, substrate, model):
    used = any(bot.substrate == substrate and bot.model == model
               for bot in bot_configs.BOTS.values())
    self.assertTrue(used, f'Model {substrate}/{model} is not used by any bot.')


if __name__ == '__main__':
  absltest.main()
