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
"""Tests of alterator puppeteer."""

from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from meltingpot.testing import puppeteers
from meltingpot.utils.puppeteers import alternator

_GOAL_A = mock.sentinel.goal_a
_GOAL_B = mock.sentinel.goal_b
_GOAL_C = mock.sentinel.goal_c


class AlternatorTest(parameterized.TestCase):

  @parameterized.parameters(1, 2, 3)
  def test_goal_sequence(self, steps_per_goal):
    puppeteer = alternator.Alternator(
        goals=[_GOAL_A, _GOAL_C, _GOAL_A, _GOAL_B],
        steps_per_goal=steps_per_goal,
    )
    num_steps = steps_per_goal * 4 * 2
    observations = [{}] * num_steps
    expected = (
        [_GOAL_A] * steps_per_goal +
        [_GOAL_C] * steps_per_goal +
        [_GOAL_A] * steps_per_goal +
        [_GOAL_B] * steps_per_goal) * 2
    actual, _ = puppeteers.goals_from_observations(puppeteer, observations)
    self.assertSequenceEqual(actual, expected)

  def test_resets_on_restart(self):
    puppeteer = alternator.Alternator(
        goals=[_GOAL_A, _GOAL_B, _GOAL_C], steps_per_goal=1)
    observations = [{}] * 4
    episode_1, state = puppeteers.goals_from_observations(
        puppeteer, observations
    )
    episode_2, _ = puppeteers.goals_from_observations(
        puppeteer, observations, state=state
    )
    expected = [_GOAL_A, _GOAL_B, _GOAL_C, _GOAL_A]
    self.assertSequenceEqual([episode_1, episode_2], [expected, expected])


if __name__ == '__main__':
  absltest.main()
