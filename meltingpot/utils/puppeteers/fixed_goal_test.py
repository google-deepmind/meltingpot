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
"""Tests of fixed goal puppeteer."""

from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from meltingpot.testing import puppeteers
from meltingpot.utils.puppeteers import fixed_goal


class FixedGoalTest(parameterized.TestCase):

  def test_goal_sequence(self):
    puppeteer = fixed_goal.FixedGoal(mock.sentinel.goal)
    observations = [{}] * 3
    expected = [mock.sentinel.goal] * 3
    actual, _ = puppeteers.goals_from_observations(puppeteer, observations)
    self.assertSequenceEqual(actual, expected)


if __name__ == '__main__':
  absltest.main()
