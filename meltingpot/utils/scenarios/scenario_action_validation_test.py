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
"""Regression tests for focal action count validation."""

from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from meltingpot.utils.scenarios import scenario as scenario_utils


class ScenarioActionValidationTest(parameterized.TestCase):

  @parameterized.parameters(([0],), ([1],), ([0, 1, 2],))
  def test_rejects_wrong_number_of_focal_actions(self, focal_action):
    scenario = object.__new__(scenario_utils.Scenario)
    scenario._is_focal = (True, False, True)
    scenario._focal_action_subject = mock.Mock()
    scenario._background_population = mock.Mock()

    with self.assertRaisesRegex(ValueError, 'Expected 2 focal actions'):
      scenario._await_full_action(focal_action)

    scenario._focal_action_subject.on_next.assert_not_called()
    scenario._background_population.await_action.assert_not_called()


if __name__ == '__main__':
  absltest.main()
