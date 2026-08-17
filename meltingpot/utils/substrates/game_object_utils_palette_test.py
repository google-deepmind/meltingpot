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
"""Regression tests for avatar palette count validation."""

from absl.testing import absltest
from meltingpot.utils.substrates import colors
from meltingpot.utils.substrates import game_object_utils
from meltingpot.utils.substrates import shapes


class PaletteCountValidationTest(absltest.TestCase):

  def test_rejects_too_few_player_palettes(self):
    palette = shapes.get_palette(colors.palette[0])

    with self.assertRaisesRegex(ValueError, 'player palettes'):
      game_object_utils.build_avatar_objects(
          num_players=2,
          prefabs={'avatar': {}},
          player_palettes=[palette],
      )

  def test_rejects_too_few_badge_palettes(self):
    palette = shapes.get_palette(colors.palette[0])

    with self.assertRaisesRegex(ValueError, 'badge palettes'):
      game_object_utils.build_avatar_badges(
          num_players=2,
          prefabs={'avatar_badge': {}},
          badge_palettes=[palette],
      )

  def test_rejects_too_many_players_for_default_palette(self):
    with self.assertRaisesRegex(ValueError, 'default player palettes'):
      game_object_utils.build_avatar_objects(
          num_players=len(colors.palette) + 1,
          prefabs={'avatar': {}},
      )


if __name__ == '__main__':
  absltest.main()
