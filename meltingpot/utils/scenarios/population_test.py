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
"""Tests for scenario populations."""

from unittest import mock

from absl.testing import absltest
from meltingpot.utils.policies import policy
from meltingpot.utils.scenarios import population


class PopulationTest(absltest.TestCase):

  def test_sampling_candidates_have_deterministic_order(self):
    policies = {
        'bot_a': mock.Mock(spec_set=policy.Policy),
        'bot_b': mock.Mock(spec_set=policy.Policy),
    }
    bot_population = population.Population(
        policies=policies,
        names_by_role={'role': {'bot_b', 'bot_a'}},
        roles=['role'],
    )
    self.addCleanup(bot_population.close)

    with mock.patch.object(
        population.random, 'choice', return_value='bot_a') as choice:
      sampled = bot_population._sample_names()

    self.assertEqual(sampled, ['bot_a'])
    choice.assert_called_once_with(('bot_a', 'bot_b'))


if __name__ == '__main__':
  absltest.main()
