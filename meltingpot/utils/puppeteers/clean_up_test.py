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
"""Tests for clean_up puppeteers."""

from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from meltingpot.testing import puppeteers
from meltingpot.utils.puppeteers import clean_up

_NUM_COOPERATORS_KEY = 'num_cooperators'
_COOPERATE = mock.sentinel.cooperate
_DEFECT = mock.sentinel.defect


def _goals(puppeteer, num_cooperators, state=None):
  observations = [{_NUM_COOPERATORS_KEY: n} for n in num_cooperators]
  goals, state = puppeteers.goals_from_observations(
      puppeteer, observations, state
  )
  return goals, state


class ConditionalCleanerTest(parameterized.TestCase):

  @parameterized.parameters(0, 1, 2)
  def test_niceness_period(self, niceness_period):
    puppeteer = clean_up.ConditionalCleaner(
        clean_goal=_COOPERATE,
        eat_goal=_DEFECT,
        coplayer_cleaning_signal=_NUM_COOPERATORS_KEY,
        recency_window=1,
        threshold=100,
        reciprocation_period=1,
        niceness_period=niceness_period,
    )
    num_cooperators = [0, 0, 0]
    expected = [_COOPERATE] * niceness_period
    expected += [_DEFECT] * (len(num_cooperators) - niceness_period)
    actual, _ = _goals(puppeteer, num_cooperators)
    self.assertEqual(actual, expected)

  def test_reciprocation_trigger(self):
    puppeteer = clean_up.ConditionalCleaner(
        clean_goal=_COOPERATE,
        eat_goal=_DEFECT,
        coplayer_cleaning_signal=_NUM_COOPERATORS_KEY,
        recency_window=1,
        threshold=4,
        reciprocation_period=1,
        niceness_period=0,
    )
    num_cooperators = [0, 1, 2, 3, 4]
    expected = [_DEFECT, _DEFECT, _DEFECT, _DEFECT, _COOPERATE]
    actual, _ = _goals(puppeteer, num_cooperators)
    self.assertSequenceEqual(actual, expected)

  @parameterized.parameters(1, 2)
  def test_reciprocation_period(self, reciprocation_period):
    puppeteer = clean_up.ConditionalCleaner(
        clean_goal=_COOPERATE,
        eat_goal=_DEFECT,
        coplayer_cleaning_signal=_NUM_COOPERATORS_KEY,
        recency_window=1,
        threshold=1,
        reciprocation_period=reciprocation_period,
        niceness_period=0,
    )
    num_cooperators = [1, 0, 0, 0, 0]
    expected = [_COOPERATE] * reciprocation_period
    expected += [_DEFECT] * (len(num_cooperators) - reciprocation_period)
    actual, _ = _goals(puppeteer, num_cooperators)
    self.assertSequenceEqual(actual, expected)

  @parameterized.parameters(
      [(1, 0, 0, 1), (_DEFECT, _DEFECT, _DEFECT, _DEFECT)],
      [(1, 0, 1), (_DEFECT, _DEFECT, _COOPERATE)],
      [(1, 1), (_DEFECT, _COOPERATE)],
  )
  def test_recency_window(self, num_cooperators, expected):
    puppeteer = clean_up.ConditionalCleaner(
        clean_goal=_COOPERATE,
        eat_goal=_DEFECT,
        coplayer_cleaning_signal=_NUM_COOPERATORS_KEY,
        recency_window=3,
        threshold=2,
        reciprocation_period=1,
        niceness_period=0,
    )
    actual, _ = _goals(puppeteer, num_cooperators)
    self.assertSequenceEqual(actual, expected)

  def test_niceness_persists(self):
    puppeteer = clean_up.ConditionalCleaner(
        clean_goal=_COOPERATE,
        eat_goal=_DEFECT,
        coplayer_cleaning_signal=_NUM_COOPERATORS_KEY,
        recency_window=1,
        threshold=1,
        reciprocation_period=1,
        niceness_period=4,
    )
    num_cooperators = [1, 0, 0, 0, 0]
    expected = [_COOPERATE, _COOPERATE, _COOPERATE, _COOPERATE, _DEFECT]
    actual, _ = _goals(puppeteer, num_cooperators)
    self.assertSequenceEqual(actual, expected)

  def test_reciprocation_extends_niceness(self):
    puppeteer = clean_up.ConditionalCleaner(
        clean_goal=_COOPERATE,
        eat_goal=_DEFECT,
        coplayer_cleaning_signal=_NUM_COOPERATORS_KEY,
        recency_window=1,
        threshold=1,
        reciprocation_period=4,
        niceness_period=2,
    )
    num_cooperators = [1, 0, 0, 0, 0]
    expected = [_COOPERATE, _COOPERATE, _COOPERATE, _COOPERATE, _DEFECT]
    actual, _ = _goals(puppeteer, num_cooperators)
    self.assertSequenceEqual(actual, expected)

  def test_reciprocation_extends_reciprocation(self):
    puppeteer = clean_up.ConditionalCleaner(
        clean_goal=_COOPERATE,
        eat_goal=_DEFECT,
        coplayer_cleaning_signal=_NUM_COOPERATORS_KEY,
        recency_window=1,
        threshold=1,
        reciprocation_period=3,
        niceness_period=0,
    )
    num_cooperators = [1, 1, 0, 0, 0]
    expected = [_COOPERATE, _COOPERATE, _COOPERATE, _COOPERATE, _DEFECT]
    actual, _ = _goals(puppeteer, num_cooperators)
    self.assertSequenceEqual(actual, expected)

  def test_resets_on_first(self):
    puppeteer = clean_up.ConditionalCleaner(
        clean_goal=_COOPERATE,
        eat_goal=_DEFECT,
        coplayer_cleaning_signal=_NUM_COOPERATORS_KEY,
        recency_window=8,
        threshold=1,
        reciprocation_period=8,
        niceness_period=1,
    )
    _, state = _goals(puppeteer, [0, 0, 1, 0])
    num_cooperators = [0, 0, 0, 0]
    expected = [_COOPERATE, _DEFECT, _DEFECT, _DEFECT]
    actual, _ = _goals(puppeteer, num_cooperators, state)
    self.assertSequenceEqual(actual, expected)

  def test_impulse_response(self):
    puppeteer = clean_up.ConditionalCleaner(
        clean_goal=_COOPERATE,
        eat_goal=_DEFECT,
        coplayer_cleaning_signal=_NUM_COOPERATORS_KEY,
        recency_window=4,
        threshold=1,
        reciprocation_period=2,
        niceness_period=0,
    )
    num_defections = [1, 0, 0, 0, 0, 0]
    expected = [
        _COOPERATE, _COOPERATE, _COOPERATE, _COOPERATE, _COOPERATE, _DEFECT
    ]
    actual, _ = _goals(puppeteer, num_defections)
    self.assertSequenceEqual(actual, expected)

  def test_boxcar_response(self):
    puppeteer = clean_up.ConditionalCleaner(
        clean_goal=_COOPERATE,
        eat_goal=_DEFECT,
        coplayer_cleaning_signal=_NUM_COOPERATORS_KEY,
        recency_window=4,
        threshold=1,
        reciprocation_period=2,
        niceness_period=0,
    )
    num_defections = [1, 1, 1, 0, 0, 0, 0, 0]
    expected = [
        _COOPERATE, _COOPERATE, _COOPERATE, _COOPERATE, _COOPERATE, _COOPERATE,
        _COOPERATE, _DEFECT
    ]
    actual, _ = _goals(puppeteer, num_defections)
    self.assertSequenceEqual(actual, expected)

if __name__ == '__main__':
  absltest.main()
