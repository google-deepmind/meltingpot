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
"""Tests for map_helpers."""

from absl.testing import absltest
from absl.testing import parameterized
from meltingpot.utils.substrates import map_helpers


class MapHelpersTest(parameterized.TestCase):

  def test_a_or_b_with_odds(self):
    self.assertEqual(
        map_helpers.a_or_b_with_odds('a', 'b', (2, 1)),
        {'type': 'choice', 'list': ['a', 'a', 'b']})

  @parameterized.parameters(
      (),
      (1,),
      (1, 2, 3),
      (-1, 1),
      (1, -1),
      (0, 0),
      (1.5, 1),
  )
  def test_a_or_b_with_odds_rejects_invalid_odds(self, odds):
    with self.assertRaises(ValueError):
      map_helpers.a_or_b_with_odds('a', 'b', odds)


if __name__ == '__main__':
  absltest.main()
