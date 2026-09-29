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
"""Regression tests for ASCII maps without a leading newline."""

from absl.testing import absltest
from meltingpot.utils.substrates import game_object_utils


class ParseMapWithoutLeadingNewlineTest(absltest.TestCase):

  def test_first_row_is_not_skipped(self):
    transforms = game_object_utils.get_game_object_positions_from_map(
        'A..\n...', 'A')

    self.assertEqual(
        transforms,
        [game_object_utils.Transform(
            position=game_object_utils.Position(0, 0),
            orientation=game_object_utils.Orientation.NORTH,
        )],
    )

  def test_existing_leading_newline_behavior_is_preserved(self):
    transforms = game_object_utils.get_game_object_positions_from_map(
        '\nA..\n...', 'A')

    self.assertEqual(
        transforms,
        [game_object_utils.Transform(
            position=game_object_utils.Position(0, 0),
            orientation=game_object_utils.Orientation.NORTH,
        )],
    )


if __name__ == '__main__':
  absltest.main()
