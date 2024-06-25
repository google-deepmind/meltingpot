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
"""Tests for allelopathic_harvest puppeteers."""

from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from meltingpot.testing import puppeteers
from meltingpot.utils.puppeteers import allelopathic_harvest

_CONSUME_ANY = mock.sentinel.consume
_PREFER_R = mock.sentinel.pref_red
_PREFER_G = mock.sentinel.pref_green
_PREFER_B = mock.sentinel.pref_blue
_PREFER = (_PREFER_R, _PREFER_G, _PREFER_B)

_RGB_KEY = 'RGB'
_NEUTRAL_COLOR = (((100, 100, 100),),)
_HIGH_R = (((255, 55, 55),),)
_HIGH_G = (((55, 255, 55),),)
_HIGH_B = (((55, 55, 255),),)


def _goals(puppeteer, rgbs, state=None):
  observations = [{_RGB_KEY: rgb} for rgb in rgbs]
  goals, state = puppeteers.goals_from_observations(
      puppeteer, observations, state
  )
  return goals, state


class ConventionFollowerTest(parameterized.TestCase):

  def test_starts_with_consume(self):
    puppeteer = allelopathic_harvest.ConventionFollower(
        initial_goal=_CONSUME_ANY,
        preference_goals=_PREFER,
        color_threshold=200,
        recency_window=5,
    )
    expected = [_CONSUME_ANY]
    actual, _ = _goals(puppeteer, [_NEUTRAL_COLOR])
    self.assertEqual(actual, expected)

  @parameterized.parameters(
      (_HIGH_R, _PREFER_R),
      (_HIGH_G, _PREFER_G),
      (_HIGH_B, _PREFER_B),
  )
  def test_change_on_immediate_recency(self, image, preference):
    puppeteer = allelopathic_harvest.ConventionFollower(
        initial_goal=_CONSUME_ANY,
        preference_goals=_PREFER,
        color_threshold=200,
        recency_window=1,
    )
    expected = [_CONSUME_ANY, preference]
    actual, _ = _goals(puppeteer, [_NEUTRAL_COLOR, image])
    self.assertEqual(actual, expected)

  @parameterized.parameters(
      (_HIGH_R,),
      (_HIGH_G,),
      (_HIGH_B,),
  )
  def test_no_change_with_long_recency(self, image):
    puppeteer = allelopathic_harvest.ConventionFollower(
        initial_goal=_CONSUME_ANY,
        preference_goals=_PREFER,
        color_threshold=250,
        recency_window=5,
    )
    expected = [_CONSUME_ANY, _CONSUME_ANY]
    actual, _ = _goals(puppeteer, [_NEUTRAL_COLOR, image])
    self.assertEqual(actual, expected)

  @parameterized.parameters(5, 10, 25)
  def test_change_over_long_recency(self, recency):
    puppeteer = allelopathic_harvest.ConventionFollower(
        initial_goal=_CONSUME_ANY,
        preference_goals=_PREFER,
        color_threshold=254,
        recency_window=recency,
    )
    expected = [_CONSUME_ANY] * recency + [_PREFER_R]
    actual, _ = _goals(puppeteer, [_NEUTRAL_COLOR] + [_HIGH_R] * recency)
    self.assertEqual(actual, expected)

if __name__ == '__main__':
  absltest.main()
