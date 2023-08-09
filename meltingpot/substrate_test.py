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
"""Tests for substrate."""

from absl.testing import absltest
from absl.testing import parameterized
import dm_env
from meltingpot import substrate
from meltingpot.testing import substrates as test_utils
import numpy as np


@parameterized.named_parameters((name, name) for name in substrate.SUBSTRATES)
class PerSubstrateTestCase(test_utils.SubstrateTestCase):

  def test_substrate(self, name):
    factory = substrate.get_factory(name)
    roles = factory.default_player_roles()
    action_spec = [factory.action_spec()] * len(roles)
    reward_spec = [factory.timestep_spec().reward] * len(roles)
    discount_spec = factory.timestep_spec().discount
    observation_spec = dict(factory.timestep_spec().observation)
    observation_spec['COLLECTIVE_REWARD'] = dm_env.specs.Array(
        shape=(), dtype=np.float64, name='COLLECTIVE_REWARD')
    observation_spec = [observation_spec] * len(roles)
    with factory.build(roles) as env:
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

if __name__ == '__main__':
  absltest.main()
