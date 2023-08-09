# Copyright 2020 DeepMind Technologies Limited.
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
"""Tests of scenarios."""

from absl.testing import absltest
from absl.testing import parameterized
import dm_env
from meltingpot import scenario
from meltingpot.testing import substrates as test_utils
import numpy as np


@parameterized.named_parameters((name, name) for name in scenario.SCENARIOS)
class ScenarioTest(test_utils.SubstrateTestCase):

  def test_scenario(self, name):
    factory = scenario.get_factory(name)
    num_players = factory.num_focal_players()
    action_spec = [factory.action_spec()] * num_players
    reward_spec = [factory.timestep_spec().reward] * num_players
    discount_spec = factory.timestep_spec().discount
    observation_spec = dict(factory.timestep_spec().observation)
    observation_spec['COLLECTIVE_REWARD'] = dm_env.specs.Array(
        shape=(), dtype=np.float64, name='COLLECTIVE_REWARD')
    observation_spec = [observation_spec] * num_players
    with factory.build() as env:
      with self.subTest('step'):
        self.assert_step_matches_specs(env)
      with self.subTest('discount_spec'):
        self.assertSequenceEqual(env.action_spec(), action_spec)
      with self.subTest('reward_spec'):
        self.assertSequenceEqual(env.reward_spec(), reward_spec)
      with self.subTest('discount_spec'):
        self.assertEqual(env.discount_spec(), discount_spec)
      with self.subTest('observation_spec'):
        self.assertSequenceEqual(env.observation_spec(), observation_spec)
      with self.subTest('only_permitted'):
        self.assertContainsSubset(factory.timestep_spec().observation,
                                  scenario.PERMITTED_OBSERVATIONS)


if __name__ == '__main__':
  absltest.main()
