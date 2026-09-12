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
"""Tests for population configuration validation."""

from unittest import mock

from absl.testing import absltest
from meltingpot.utils.policies import policy
from meltingpot.utils.scenarios import population


class PopulationValidationTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.policy = mock.Mock(spec_set=policy.Policy)

  def test_rejects_missing_role_candidates(self):
    with self.assertRaisesRegex(ValueError, 'No population candidates'):
      population.Population(
          policies={'bot': self.policy},
          names_by_role={},
          roles=['resident'],
      )

  def test_rejects_empty_role_candidates(self):
    with self.assertRaisesRegex(ValueError, 'candidates.*empty'):
      population.Population(
          policies={'bot': self.policy},
          names_by_role={'resident': ()},
          roles=['resident'],
      )

  def test_rejects_unknown_policy_candidate(self):
    with self.assertRaisesRegex(ValueError, 'unknown policies'):
      population.Population(
          policies={'bot': self.policy},
          names_by_role={'resident': ('missing_bot',)},
          roles=['resident'],
      )


if __name__ == '__main__':
  absltest.main()
