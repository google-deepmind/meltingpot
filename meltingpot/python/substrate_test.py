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


def _get_lua_randomization_map():
  """Replaces first row of walls with items randomized by Lua."""
  config = substrate.get_config('running_with_scissors_in_the_matrix')
  head, line, *tail = config.lab2d_settings.simulation.map.split('\n')
  # Replace line 1 (walls) with a row of 'a' (items randomized by Lua).
  new_map = '\n'.join([head, 'a' * len(line), *tail])
  return new_map


_LUA_RANDOMIZED_LINE = 1
_LUA_RANDOMIZATION_MAP = _get_lua_randomization_map()


class SubstrateTest(parameterized.TestCase):

  @parameterized.product(seed=[42, 123, 1337, 12481632])
  def test_seed_causes_determinism(self, seed):
    config = substrate.get_config('running_with_scissors_in_the_matrix')
    with config.unlocked():
      config.env_seed = seed

    env1 = substrate.build(config)
    env2 = substrate.build(config)
    for episode in range(5):
      obs1 = env1.reset().observation[0]['WORLD.RGB']
      obs2 = env2.reset().observation[0]['WORLD.RGB']
      np.testing.assert_equal(
          obs1, obs2, f'Episode {episode} mismatch: {obs1} != {obs2} ')

  @parameterized.product(seed=[None, 42, 123, 1337, 12481632])
  def test_episodes_are_randomized(self, seed):
    config = substrate.get_config('running_with_scissors_in_the_matrix')
    with config.unlocked():
      config.env_seed = seed
    env = substrate.build(config)

    obs = env.reset().observation[0]['WORLD.RGB']
    for episode in range(4):
      last_obs = obs
      obs = env.reset().observation[0]['WORLD.RGB']
      with self.assertRaises(
          AssertionError,
          msg=f'Episodes {episode} and {episode+1} match: {last_obs} == {obs}'):
        np.testing.assert_equal(last_obs, obs)

  def test_no_seed_causes_nondeterminism(self):
    config = substrate.get_config('running_with_scissors_in_the_matrix')
    with config.unlocked():
      config.env_seed = None

    env1 = substrate.build(config)
    env2 = substrate.build(config)
    for episode in range(5):
      obs1 = env1.reset().observation[0]['WORLD.RGB']
      obs2 = env2.reset().observation[0]['WORLD.RGB']
      with self.assertRaises(
          AssertionError, msg=f'Episode {episode} match: {obs1} == {obs2}'):
        np.testing.assert_equal(obs1, obs2)

  @parameterized.product(seed=[None, 42, 123, 1337, 12481632])
  def test_episodes_are_randomized_in_lua(self, seed):
    config = substrate.get_config('running_with_scissors_in_the_matrix')
    with config.unlocked():
      config.env_seed = seed
      config.lab2d_settings.simulation.map = _LUA_RANDOMIZATION_MAP
    env = substrate.build(config)

    obs = env.reset().observation[0]['WORLD.RGB'][_LUA_RANDOMIZED_LINE]
    for episode in range(4):
      last_obs = obs
      obs = env.reset().observation[0]['WORLD.RGB'][_LUA_RANDOMIZED_LINE]
      with self.assertRaises(
          AssertionError,
          msg=f'Episodes {episode} and {episode+1} match: {last_obs} == {obs}'):
        np.testing.assert_equal(last_obs, obs)

  def test_no_seed_causes_nondeterminism_for_lua(self):
    config = substrate.get_config('running_with_scissors_in_the_matrix')
    with config.unlocked():
      config.env_seed = None
      config.lab2d_settings.simulation.map = _LUA_RANDOMIZATION_MAP

    env1 = substrate.build(config)
    env2 = substrate.build(config)
    for episode in range(5):
      obs1 = env1.reset().observation[0]['WORLD.RGB'][_LUA_RANDOMIZED_LINE]
      obs2 = env2.reset().observation[0]['WORLD.RGB'][_LUA_RANDOMIZED_LINE]
      with self.assertRaises(
          AssertionError, msg=f'Episode {episode} match {obs1} == {obs2}'):
        np.testing.assert_equal(obs1, obs2)

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
