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
import numpy as np

from meltingpot.python import substrate

REWARD_SPEC = dm_env.specs.Array(shape=[], dtype=np.float64, name='REWARD')
ACTION_SPEC = dm_env.specs.DiscreteArray(
    num_values=1, dtype=np.int32, name='action')


class SubstrateTest(parameterized.TestCase):

  @parameterized.parameters(
      [[42], [123], [1337], [12481632]])
  def test_seed_causes_determinism(self, seed):
    config = substrate.get_config('allelopathic_harvest')
    with config.unlocked():
      config.env_seed = seed

    env = substrate.build(config)
    prev_obs = env.reset().observation[0]
    obs = []
    for _ in range(10):
      env = substrate.build(config)
      obs.append(env.reset().observation[0])

    np.testing.assert_equal(obs, [prev_obs] * 10)

  @parameterized.named_parameters(
      (name, name) for name in substrate.AVAILABLE_SUBSTRATES)
  def test_substrate_creation(self, substrate_name):
    config = substrate.get_config(substrate_name)
    with substrate.build(config) as env:
      reset_timestep = env.reset()
      action_spec = env.action_spec()
      observation_spec = env.observation_spec()
      reward_spec = env.reward_spec()

    with self.subTest('reset_reward'):
      self.assertNotEqual(reset_timestep.reward, None)
    with self.subTest('reset_discount'):
      self.assertNotEqual(reset_timestep.discount, None)
    with self.subTest('observation_spec'):
      self.assertLen(observation_spec, config.num_players)
    with self.subTest('action_spec'):
      spec = ACTION_SPEC.replace(num_values=len(config.action_set))
      self.assertEqual(action_spec, (spec,) * config.num_players)
    with self.subTest('reward_spec'):
      self.assertEqual(reward_spec, [REWARD_SPEC] * config.num_players)


if __name__ == '__main__':
  absltest.main()
