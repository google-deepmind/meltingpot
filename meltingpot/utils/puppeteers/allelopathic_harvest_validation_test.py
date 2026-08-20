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
"""Validation tests for allelopathic_harvest puppeteers."""

from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from meltingpot.utils.puppeteers import allelopathic_harvest


_INITIAL = mock.sentinel.initial
_PREFERENCES = (
    mock.sentinel.red,
    mock.sentinel.green,
    mock.sentinel.blue,
)


class ConventionFollowerValidationTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('empty', ()),
      ('too_few', _PREFERENCES[:2]),
      ('too_many', _PREFERENCES + (mock.sentinel.extra,)),
  )
  def test_requires_one_goal_per_rgb_channel(self, preference_goals):
    with self.assertRaisesRegex(ValueError, 'exactly 3 goals'):
      allelopathic_harvest.ConventionFollower(
          initial_goal=_INITIAL,
          preference_goals=preference_goals,
          color_threshold=200,
      )

  @parameterized.parameters(0, -1)
  def test_requires_positive_recency_window(self, recency_window):
    with self.assertRaisesRegex(ValueError, 'recency_window must be positive'):
      allelopathic_harvest.ConventionFollower(
          initial_goal=_INITIAL,
          preference_goals=_PREFERENCES,
          color_threshold=200,
          recency_window=recency_window,
      )


if __name__ == '__main__':
  absltest.main()
