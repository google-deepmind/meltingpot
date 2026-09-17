# Copyright 2026 DeepMind Technologies Limited.
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
"""Tests for the collaborative cooking human player."""

from absl.testing import absltest
from meltingpot.configs.substrates import collaborative_cooking as base_config
from meltingpot.configs.substrates import collaborative_cooking__cramped
from meltingpot.human_players import play_collaborative_cooking


class PlayCollaborativeCookingTest(absltest.TestCase):

  def test_verbose_build_enables_debug_metrics_without_changing_default(self):
    previous_debug_setting = getattr(base_config, '_ENABLE_DEBUG_OBSERVATIONS')

    env_config = play_collaborative_cooking._build_environment_config(  # pylint: disable=protected-access
        collaborative_cooking__cramped, verbose=True)

    self.assertEqual(
        getattr(base_config, '_ENABLE_DEBUG_OBSERVATIONS'),
        previous_debug_setting)
    settings = repr(env_config.lab2d_settings)
    self.assertIn('ADDED_INGREDIENT_TO_COOKING_POT', settings)
    self.assertIn('COLLECTED_SOUP_FROM_COOKING_POT', settings)

  def test_normal_build_omits_debug_metrics(self):
    env_config = play_collaborative_cooking._build_environment_config(  # pylint: disable=protected-access
        collaborative_cooking__cramped, verbose=False)

    settings = repr(env_config.lab2d_settings)
    self.assertNotIn('ADDED_INGREDIENT_TO_COOKING_POT', settings)
    self.assertNotIn('COLLECTED_SOUP_FROM_COOKING_POT', settings)


if __name__ == '__main__':
  absltest.main()
