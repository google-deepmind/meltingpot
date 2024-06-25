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
_NUM_OTHERS_WHO_CLEANED_THIS_STEP_KEY = 'NUM_OTHERS_WHO_CLEANED_THIS_STEP'
_RGB_KEY = 'RGB'
_RGB_TIMEOUT = 0
_RGB_NORMAL = 1
_SANCTION = mock.sentinel.sanction


def _goals(puppeteer, num_cooperators, state=None):
  observations = [{_NUM_COOPERATORS_KEY: n} for n in num_cooperators]
  goals, state = puppeteers.goals_from_observations(
      puppeteer, observations, state
  )
  return goals, state


def _goals_3_goals(puppeteer, cleaned, rgbs, state=None):
  assert len(cleaned) == len(rgbs), (
      'Cleaned and RGB observations must match in length.')
  observations = [{_NUM_OTHERS_WHO_CLEANED_THIS_STEP_KEY: clean,
                   _RGB_KEY: rgb} for clean, rgb in zip(cleaned, rgbs)]
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


class CorrigibleReciprocatorTest(parameterized.TestCase):

  @parameterized.parameters(1, 2, 3)
  def test_defects_if_unsanctioned(self, threshold):
    puppeteer = clean_up.CorrigibleReciprocator(
        cooperate_goal=_COOPERATE,
        defect_goal=_DEFECT,
        num_others_cooperating_cumulant='NUM_OTHERS_WHO_CLEANED_THIS_STEP',
        recency_window=1,
        threshold=threshold,
    )
    num_steps = 3
    expected = [_DEFECT] * num_steps
    actual, _ = _goals_3_goals(puppeteer, [1] * num_steps,
                               rgbs=[_RGB_NORMAL] * num_steps)
    self.assertEqual(actual, expected)

  def test_sanction_triggers_cooperation(self):
    puppeteer = clean_up.CorrigibleReciprocator(
        cooperate_goal=_COOPERATE,
        defect_goal=_DEFECT,
        num_others_cooperating_cumulant='NUM_OTHERS_WHO_CLEANED_THIS_STEP',
        recency_window=1,
        threshold=2,
        timeout_steps=1,
        corrigible_threshold=2,
    )
    expected = [_DEFECT, _COOPERATE, _COOPERATE]
    actual, _ = _goals_3_goals(puppeteer, [0, 0, 0],
                               rgbs=[_RGB_TIMEOUT, _RGB_TIMEOUT, _RGB_NORMAL])
    self.assertEqual(actual, expected)

  @parameterized.parameters(1, 2, 3, 25, 100)
  def test_sanctioned_cooperation_expires(self, motivation):
    puppeteer = clean_up.CorrigibleReciprocator(
        cooperate_goal=_COOPERATE,
        defect_goal=_DEFECT,
        num_others_cooperating_cumulant='NUM_OTHERS_WHO_CLEANED_THIS_STEP',
        recency_window=1,
        steps_to_cooperate_when_motivated=motivation,
        threshold=2,
        timeout_steps=1,
        corrigible_threshold=2,
    )

    num_steps = motivation + 2
    expected = [_DEFECT] + [_COOPERATE] * motivation + [_DEFECT]
    actual, _ = _goals_3_goals(
        puppeteer, [0] * num_steps,
        rgbs=[_RGB_TIMEOUT, _RGB_TIMEOUT] + [_RGB_NORMAL] * motivation)
    self.assertEqual(actual, expected)


class SanctionerAlternatorTest(parameterized.TestCase):

  def test_nice_starts_cooperating(self):
    puppeteer = clean_up.SanctionerAlternator(
        cooperate_goal=_COOPERATE,
        defect_goal=_DEFECT,
        sanction_goal=_SANCTION,
        num_others_cooperating_cumulant='NUM_OTHERS_WHO_CLEANED_THIS_STEP',
        threshold=1,
        recency_window=50,
        steps_to_sanction_when_motivated=100,
        alternating_steps=200,
        nice=True,
    )
    expected = [_COOPERATE]
    actual, _ = _goals_3_goals(puppeteer, [0], rgbs=[1])
    self.assertEqual(actual, expected)

  def test_no_nice_starts_defecting(self):
    puppeteer = clean_up.SanctionerAlternator(
        cooperate_goal=_COOPERATE,
        defect_goal=_DEFECT,
        sanction_goal=_SANCTION,
        num_others_cooperating_cumulant='NUM_OTHERS_WHO_CLEANED_THIS_STEP',
        threshold=1,
        recency_window=50,
        steps_to_sanction_when_motivated=100,
        alternating_steps=200,
        nice=False,
    )
    expected = [_DEFECT]
    actual, _ = _goals_3_goals(puppeteer, [0], rgbs=[1])
    self.assertEqual(actual, expected)

  @parameterized.parameters(1, 5, 10)
  def test_nice_for_recency_window_then_sanction(self, window):
    puppeteer = clean_up.SanctionerAlternator(
        cooperate_goal=_COOPERATE,
        defect_goal=_DEFECT,
        sanction_goal=_SANCTION,
        num_others_cooperating_cumulant='NUM_OTHERS_WHO_CLEANED_THIS_STEP',
        threshold=1,
        recency_window=window,
        steps_to_sanction_when_motivated=100,
        alternating_steps=200,
        nice=True,
    )
    num_steps = window + 1
    expected = [_COOPERATE] * (window-1) + [_SANCTION] * 2
    actual, _ = _goals_3_goals(puppeteer, [0] * num_steps,
                               rgbs=[_RGB_NORMAL] * num_steps)
    self.assertEqual(actual, expected)

  @parameterized.parameters(1, 5, 10)
  def test_alternate_if_others_cooperating(self, alternating):
    puppeteer = clean_up.SanctionerAlternator(
        cooperate_goal=_COOPERATE,
        defect_goal=_DEFECT,
        sanction_goal=_SANCTION,
        num_others_cooperating_cumulant='NUM_OTHERS_WHO_CLEANED_THIS_STEP',
        threshold=1,
        recency_window=50,
        steps_to_sanction_when_motivated=100,
        alternating_steps=alternating,
        nice=True,
    )
    num_steps = alternating * 4
    expected = ([_COOPERATE] * alternating + [_DEFECT] * alternating) * 2
    actual, _ = _goals_3_goals(puppeteer, [1] * num_steps,
                               rgbs=[_RGB_NORMAL] * num_steps)
    self.assertEqual(actual, expected)

if __name__ == '__main__':
  absltest.main()
