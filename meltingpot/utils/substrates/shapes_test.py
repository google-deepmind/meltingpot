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

import collections

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


class VerticalFlipFramingTest(parameterized.TestCase):

  @parameterized.parameters(
      ('ab\ncd', 'cd\nab'),
      ('ab\ncd\n', 'cd\nab\n'),
      ('\nab\ncd', '\ncd\nab'),
      ('\nab\ncd\n', '\ncd\nab\n'),
      ('abc', 'abc'),
      ('abc\n', 'abc\n'),
      ('\nabc', '\nabc'),
      ('', ''),
      ('\n', '\n'),
      (' a \nb c', 'b c\n a '),
      ('\nab\n\ncd\n', '\ncd\n\nab\n'),
  )
  def test_vertical_flip_preserves_every_pixel_and_boundary(
      self, sprite, expected
  ):
    actual = shapes.flip_vertical(sprite)
    self.assertEqual(actual, expected)
    self.assertEqual(collections.Counter(actual), collections.Counter(sprite))
    self.assertEqual(actual.count('\n'), sprite.count('\n'))

  @parameterized.product(
      leading=('', '\n'),
      trailing=('', '\n'),
      rows=(('ab', 'cd'), ('a', 'b', 'c'), ('  a', ' b ', 'c  '), ('xyz',)),
  )
  def test_flipping_twice_restores_the_exact_sprite(
      self, leading, trailing, rows
  ):
    sprite = leading + '\n'.join(rows) + trailing
    self.assertEqual(shapes.flip_vertical(shapes.flip_vertical(sprite)), sprite)
    self.assertEqual(
        shapes.flip_vertical(sprite),
        leading + '\n'.join(reversed(rows)) + trailing,
    )

  @parameterized.product(leading=('', '\n'), trailing=('', '\n'))
  def test_horizontal_and_vertical_flips_commute(self, leading, trailing):
    sprite = leading + 'abc\ndef\nghi' + trailing
    vertical_then_horizontal = shapes.flip_horizontal(
        shapes.flip_vertical(sprite)
    )
    horizontal_then_vertical = shapes.flip_vertical(
        shapes.flip_horizontal(sprite)
    )
    self.assertEqual(vertical_then_horizontal, horizontal_then_vertical)
    self.assertEqual(
        vertical_then_horizontal, leading + 'ihg\nfed\ncba' + trailing
    )

  def test_existing_framed_sprite_constants_keep_identical_output(self):
    checked = 0
    for name, sprite in vars(shapes).items():
      if not (
          isinstance(sprite, str)
          and sprite.startswith('\n')
          and sprite.endswith('\n')
      ):
        continue
      with self.subTest(sprite=name):
        # The previous routine's output for its supported framed inputs.
        expected = ''.join(
            row + '\n' for row in reversed(sprite[1:].split('\n'))
        )
        self.assertEqual(shapes.flip_vertical(sprite), expected)
        self.assertEqual(
            shapes.flip_vertical(shapes.flip_vertical(sprite)), sprite
        )
        checked += 1
    self.assertGreater(checked, 0)

  def test_unframed_real_sprite_matches_the_framed_pixel_grid(self):
    sprite = shapes.HD_AVATAR_N
    bare = sprite.strip('\n')
    actual = shapes.flip_vertical(bare)
    expected_rows = sprite.strip('\n').split('\n')[::-1]
    self.assertEqual(actual.split('\n'), expected_rows)
    self.assertEqual(shapes.flip_vertical(sprite).strip('\n'), actual)
    self.assertTrue(all(len(row) == 16 for row in actual.split('\n')))


if __name__ == '__main__':
  absltest.main()
