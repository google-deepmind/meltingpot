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
"""Validation tests for SanctionerAlternator."""

from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from meltingpot.utils.puppeteers import clean_up


class SanctionerAlternatorValidationTest(parameterized.TestCase):

  @parameterized.parameters(0, -1)
  def test_rejects_nonpositive_alternating_steps(self, alternating_steps):
    with self.assertRaisesRegex(ValueError, 'alternating_steps must be positive'):
      clean_up.SanctionerAlternator(
          cooperate_goal=mock.sentinel.cooperate,
          defect_goal=mock.sentinel.defect,
          sanction_goal=mock.sentinel.sanction,
          num_others_cooperating_cumulant='NUM_OTHERS_COOPERATING',
          threshold=1,
          alternating_steps=alternating_steps,
      )


if __name__ == '__main__':
  absltest.main()
