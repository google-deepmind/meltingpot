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
"""Tests for shapes."""

from absl.testing import absltest
from absl.testing import parameterized
from meltingpot.utils.substrates import shapes


class ShapesTest(parameterized.TestCase):

  @parameterized.parameters([
      ["""
a
b
c
""", """
c
b
a
"""], ["""
""", """
"""], ["""
abc
def
ghi
""", """
ghi
def
abc
"""],
  ])
  def test_flip_vertical(self, original, expected):
    actual = shapes.flip_vertical(original)
    self.assertEqual(actual, expected)

  @parameterized.parameters([
      ["""
a
b
c
""", """
a
b
c
"""], ["""
""", """
"""], ["""
abc
def
ghi
""", """
cba
fed
ihg
"""],
  ])
  def test_flip_horizontal(self, original, expected):
    actual = shapes.flip_horizontal(original)
    self.assertEqual(actual, expected)


if __name__ == '__main__':
  absltest.main()
