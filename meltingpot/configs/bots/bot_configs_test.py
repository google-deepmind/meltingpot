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
"""Tests of the bot configs."""

import collections
import os

from absl.testing import absltest
from absl.testing import parameterized
from meltingpot.configs import bots
from meltingpot.configs import substrates


def _subdirs(root):
  for file in os.listdir(root):
    if os.path.isdir(os.path.join(root, file)):
      yield file


def _models(models_root=bots.MODELS_ROOT):
  for substrate in _subdirs(models_root):
    for model in _subdirs(os.path.join(models_root, substrate)):
      yield os.path.join(models_root, substrate, model)


BOT_CONFIGS = bots.BOT_CONFIGS
AVAILABLE_MODELS = frozenset(_models())
AVAILABLE_SUBSTRATES = frozenset(substrates.SUBSTRATES)


class BotConfigTest(parameterized.TestCase):

  @parameterized.named_parameters(BOT_CONFIGS.items())
  def test_has_valid_substrate(self, bot):
    self.assertIn(bot.substrate, AVAILABLE_SUBSTRATES)

  @parameterized.named_parameters(BOT_CONFIGS.items())
  def test_model_exists(self, bot):
    self.assertTrue(
        os.path.isdir(bot.model_path), f'Missing model {bot.model_path!r}.')

  @parameterized.named_parameters(BOT_CONFIGS.items())
  def test_substrate_matches_model(self, bot):
    substrate = os.path.basename(os.path.dirname(bot.model_path))
    self.assertEqual(bot.substrate, substrate,
                     f'{bot} substrate does not match model path.')

  def test_no_duplicates(self):
    seen = collections.defaultdict(set)
    for name, config in BOT_CONFIGS.items():
      seen[config].add(name)
    duplicates = {names for _, names in seen.items() if len(names) > 1}
    self.assertEmpty(duplicates, f'Duplicate configs found: {duplicates!r}.')

  def test_models_used_by_bots(self):
    used = {bot.model_path for bot in BOT_CONFIGS.values()}
    unused = AVAILABLE_MODELS - used
    self.assertEmpty(unused, f'Models not used by any bot: {unused!r}')

if __name__ == '__main__':
  absltest.main()
