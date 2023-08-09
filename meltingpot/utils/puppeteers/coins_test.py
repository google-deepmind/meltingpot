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
"""Tests for coins puppeteers."""

from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from meltingpot.testing import puppeteers
from meltingpot.utils.puppeteers import coins

_COOPERATE = mock.sentinel.cooperate
_DEFECT = mock.sentinel.defect
_SPITE = mock.sentinel.spite
_NUM_DEFECTIONS_KEY = 'DEFECTIONS'


def _goals(puppeteer, num_defections, state=None):
  observations = [{_NUM_DEFECTIONS_KEY: n} for n in num_defections]
  goals, state = puppeteers.goals_from_observations(
      puppeteer, observations, state
  )
  return goals, state


class ReciprocatorTest(parameterized.TestCase):

  def test_threshold(self):
    puppeteer = coins.Reciprocator(
        cooperate_goal=_COOPERATE,
        defect_goal=_DEFECT,
        spite_goal=_SPITE,
        partner_defection_signal=_NUM_DEFECTIONS_KEY,
        recency_window=1,
        threshold=3,
        frames_to_punish=1,
        spiteful_punishment_window=0,
    )
    num_defections = [0, 1, 2, 3]
    expected = [_COOPERATE, _COOPERATE, _COOPERATE, _DEFECT]
    actual, _ = _goals(puppeteer, num_defections)
    self.assertSequenceEqual(actual, expected)

  @parameterized.parameters(
      [(1, 0, 0, 1), (_COOPERATE, _COOPERATE, _COOPERATE, _COOPERATE)],
      [(1, 0, 1), (_COOPERATE, _COOPERATE, _DEFECT)],
      [(1, 1), (_COOPERATE, _DEFECT)],
  )
  def test_recency_window(self, num_defections, expected):
    puppeteer = coins.Reciprocator(
        cooperate_goal=_COOPERATE,
        defect_goal=_DEFECT,
        spite_goal=_SPITE,
        partner_defection_signal=_NUM_DEFECTIONS_KEY,
        recency_window=3,
        threshold=2,
        frames_to_punish=1,
        spiteful_punishment_window=0,
    )
    actual, _ = _goals(puppeteer, num_defections)
    self.assertSequenceEqual(actual, expected)

  @parameterized.parameters(1, 2, 3)
  def test_defect_duration(self, duration):
    puppeteer = coins.Reciprocator(
        cooperate_goal=_COOPERATE,
        defect_goal=_DEFECT,
        spite_goal=_SPITE,
        partner_defection_signal=_NUM_DEFECTIONS_KEY,
        recency_window=1,
        threshold=1,
        frames_to_punish=duration,
        spiteful_punishment_window=1,
    )
    num_defections = [1, 0, 0, 0]
    expected = (
        [_SPITE] + [_DEFECT] * (duration - 1) + [_COOPERATE] * (4 - duration))
    actual, _ = _goals(puppeteer, num_defections)
    self.assertSequenceEqual(actual, expected)

  @parameterized.parameters(1, 2, 4)
  def test_spite_duration(self, duration):
    puppeteer = coins.Reciprocator(
        cooperate_goal=_COOPERATE,
        defect_goal=_DEFECT,
        spite_goal=_SPITE,
        partner_defection_signal=_NUM_DEFECTIONS_KEY,
        recency_window=1,
        threshold=1,
        frames_to_punish=4,
        spiteful_punishment_window=duration,
    )
    num_defections = [1, 0, 0, 0]
    expected = [_SPITE] * duration + [_DEFECT] * (4 - duration)
    actual, _ = _goals(puppeteer, num_defections)
    self.assertSequenceEqual(actual, expected)

  def test_resets_on_first(self):
    puppeteer = coins.Reciprocator(
        cooperate_goal=_COOPERATE,
        defect_goal=_DEFECT,
        spite_goal=_SPITE,
        partner_defection_signal=_NUM_DEFECTIONS_KEY,
        recency_window=8,
        threshold=1,
        frames_to_punish=8,
        spiteful_punishment_window=8,
    )
    _, state = _goals(puppeteer, [0, 0, 1, 0])
    num_defections = [0, 0, 0, 0]
    expected = [_COOPERATE, _COOPERATE, _COOPERATE, _COOPERATE]
    actual, _ = _goals(puppeteer, num_defections, state)
    self.assertSequenceEqual(actual, expected)

  def test_defection_during_defect_resets_spite(self):
    puppeteer = coins.Reciprocator(
        cooperate_goal=_COOPERATE,
        defect_goal=_DEFECT,
        spite_goal=_SPITE,
        partner_defection_signal=_NUM_DEFECTIONS_KEY,
        recency_window=1,
        threshold=1,
        frames_to_punish=3,
        spiteful_punishment_window=1,
    )
    num_defections = [1, 0, 1, 0, 0, 0]
    expected = [_SPITE, _DEFECT, _SPITE, _DEFECT, _DEFECT, _COOPERATE]
    actual, _ = _goals(puppeteer, num_defections)
    self.assertSequenceEqual(actual, expected)

  def test_defection_during_spite_extends_spite(self):
    puppeteer = coins.Reciprocator(
        cooperate_goal=_COOPERATE,
        defect_goal=_DEFECT,
        spite_goal=_SPITE,
        partner_defection_signal=_NUM_DEFECTIONS_KEY,
        recency_window=1,
        threshold=1,
        frames_to_punish=3,
        spiteful_punishment_window=2,
    )
    num_defections = [1, 0, 1, 0, 0, 0]
    expected = [_SPITE, _SPITE, _SPITE, _SPITE, _DEFECT, _COOPERATE]
    actual, _ = _goals(puppeteer, num_defections)
    self.assertSequenceEqual(actual, expected)

  def test_impulse_response(self):
    puppeteer = coins.Reciprocator(
        cooperate_goal=_COOPERATE,
        defect_goal=_DEFECT,
        spite_goal=_SPITE,
        partner_defection_signal=_NUM_DEFECTIONS_KEY,
        recency_window=4,
        threshold=1,
        frames_to_punish=2,
        spiteful_punishment_window=1,
    )
    num_defections = [1, 0, 0, 0]
    expected = [_SPITE, _DEFECT, _COOPERATE, _COOPERATE]
    actual, _ = _goals(puppeteer, num_defections)
    self.assertSequenceEqual(actual, expected)

  def test_boxcar_response(self):
    puppeteer = coins.Reciprocator(
        cooperate_goal=_COOPERATE,
        defect_goal=_DEFECT,
        spite_goal=_SPITE,
        partner_defection_signal=_NUM_DEFECTIONS_KEY,
        recency_window=4,
        threshold=1,
        frames_to_punish=2,
        spiteful_punishment_window=1,
    )
    num_defections = [1, 1, 1, 0, 0, 0]
    expected = [_SPITE, _SPITE, _SPITE, _DEFECT, _COOPERATE, _COOPERATE]
    actual, _ = _goals(puppeteer, num_defections)
    self.assertSequenceEqual(actual, expected)


if __name__ == '__main__':
  absltest.main()
