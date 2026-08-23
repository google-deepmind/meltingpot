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
"""Regression tests for SanctionerAlternator."""

from unittest import mock

from absl.testing import absltest
from meltingpot.testing import puppeteers
from meltingpot.utils.puppeteers import clean_up


_COOPERATE = mock.sentinel.cooperate
_DEFECT = mock.sentinel.defect
_SANCTION = mock.sentinel.sanction
_SIGNAL = 'NUM_OTHERS_WHO_CLEANED_THIS_STEP'


class SanctionerAlternatorNotNiceTest(absltest.TestCase):

  def test_not_nice_alternates_after_starting_with_defection(self):
    puppeteer = clean_up.SanctionerAlternator(
        cooperate_goal=_COOPERATE,
        defect_goal=_DEFECT,
        sanction_goal=_SANCTION,
        num_others_cooperating_cumulant=_SIGNAL,
        threshold=1,
        recency_window=1,
        steps_to_sanction_when_motivated=10,
        alternating_steps=2,
        nice=False,
    )
    observations = [{_SIGNAL: 1}] * 6

    goals, _ = puppeteers.goals_from_observations(puppeteer, observations)

    self.assertEqual(
        goals,
        [_DEFECT, _DEFECT, _COOPERATE, _COOPERATE, _DEFECT, _DEFECT],
    )


if __name__ == '__main__':
  absltest.main()
