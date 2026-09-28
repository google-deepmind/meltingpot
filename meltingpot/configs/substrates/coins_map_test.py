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
"""Regression tests for the procedural Coins map."""

from absl.testing import absltest
from meltingpot.configs.substrates import coins


class CoinsMapTest(absltest.TestCase):

  def test_height_padding_preserves_row_width(self):
    ascii_map = coins.get_ascii_map(
        min_width=10,
        max_width=10,
        min_height=8,
        max_height=10,
    )

    rows = ascii_map.splitlines()
    self.assertLen(rows, 12)
    self.assertEqual({len(row) for row in rows}, {12})


if __name__ == '__main__':
  absltest.main()
