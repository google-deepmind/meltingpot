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

  def test_scale_color_clamps_negative_channels_to_zero(self):
    self.assertEqual(
        shapes.scale_color((80, 120, 200, 64), -0.5), (0, 0, 0, 64)
    )
    self.assertEqual(shapes.scale_color((80, 120, 200), -2), (0, 0, 0, 255))

  def test_scale_color_clamps_high_values_and_preserves_alpha(self):
    self.assertEqual(
        shapes.scale_color((200, 100, 50, 14), 2), (255, 200, 100, 14)
    )
    self.assertEqual(
        shapes.scale_color((100, 80, 60, 14), 0.5, 23), (50, 40, 30, 23)
    )

  def test_brightness_does_not_exceed_rgb_range(self):
    result = shapes.adjust_color_brightness((200, 100, 50, 14), 2)
    self.assertEqual(result[3], 14)
    self.assertEqual(result[0], 255)
    self.assertTrue(all(0 <= value <= 255 for value in result))

  def test_negative_brightness_becomes_black_without_changing_alpha(self):
    self.assertEqual(
        shapes.adjust_color_brightness((200, 100, 50), -0.5), (0, 0, 0, 255)
    )
    self.assertEqual(
        shapes.adjust_color_brightness((200, 100, 50, 20), -0.5), (0, 0, 0, 20)
    )

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
