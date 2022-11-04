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
import numpy as np

from meltingpot.python import substrate
from meltingpot.python.testing import substrates as test_utils


def _get_lua_randomization_map():
  """Replaces first row of walls with items randomized by Lua."""
  config = substrate.get_config('running_with_scissors_in_the_matrix')
  head, line, *tail = config.lab2d_settings.simulation.map.split('\n')
  # Replace line 1 (walls) with a row of 'a' (items randomized by Lua).
  new_map = '\n'.join([head, 'a' * len(line), *tail])
  return new_map


_LUA_RANDOMIZED_LINE = 1
_LUA_RANDOMIZATION_MAP = _get_lua_randomization_map()


class GeneralTestCase(parameterized.TestCase):

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
class SubstrateTestCase(test_utils.SubstrateTestCase):

  def test_matches_spec(self, name):
    config = substrate.get_config(name)
    with substrate.build(config) as env:
      with self.subTest('discount'):
        self.assert_discount_matches_spec(env)
      with self.subTest('reward'):
        self.assert_reward_matches_spec(env)
      with self.subTest('observation'):
        self.assert_observation_matches_spec(env)

  def test_spec_in_config_matches_environment(self, name):
    config = substrate.get_config(name)
    action_spec = [config.action_spec] * config.num_players
    reward_spec = [config.timestep_spec.reward] * config.num_players
    observation_spec = [
        dict(config.timestep_spec.observation)] * config.num_players
    with substrate.build(config) as env:
      with self.subTest('discount_spec'):
        self.assertSequenceEqual(env.action_spec(), action_spec)
      with self.subTest('reward_spec'):
        self.assertSequenceEqual(env.reward_spec(), reward_spec)
      with self.subTest('discount_spec'):
        self.assertEqual(env.discount_spec(), config.timestep_spec.discount)
      with self.subTest('observation_spec'):
        self.assertSequenceEqual(env.observation_spec(), observation_spec)

if __name__ == '__main__':
  absltest.main()
