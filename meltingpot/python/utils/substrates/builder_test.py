# Copyright 2022 DeepMind Technologies Limited.
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
"""Tests for builder.py."""

import copy

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

from meltingpot.python.configs import substrates as substrate_configs
from meltingpot.python.utils.substrates import builder

_TEST_CONFIG = substrate_configs.get_config(
    'running_with_scissors_in_the_matrix')


def _get_lua_randomization_map():
  """Replaces first row of walls with items randomized by Lua."""
  head, line, *tail = _TEST_CONFIG.lab2d_settings.simulation.map.split('\n')
  # Replace line 1 (walls) with a row of 'a' (items randomized by Lua).
  new_map = '\n'.join([head, 'a' * len(line), *tail])
  return new_map


_LUA_RANDOMIZED_LINE = 1
_LUA_RANDOMIZATION_MAP = _get_lua_randomization_map()


class GeneralTestCase(parameterized.TestCase):

  @parameterized.product(seed=[42, 123, 1337, 12481632])
  def test_seed_causes_determinism(self, seed):
    config = copy.deepcopy(_TEST_CONFIG).unlock()
    config.env_seed = seed

    env1 = self.enter_context(builder.builder(**config))
    env2 = self.enter_context(builder.builder(**config))
    for episode in range(5):
      obs1 = env1.reset().observation['WORLD.RGB']
      obs2 = env2.reset().observation['WORLD.RGB']
      np.testing.assert_equal(
          obs1, obs2, f'Episode {episode} mismatch: {obs1} != {obs2} ')

  @parameterized.product(seed=[None, 42, 123, 1337, 12481632])
  def test_episodes_are_randomized(self, seed):
    config = copy.deepcopy(_TEST_CONFIG).unlock()
    config.env_seed = seed
    env = self.enter_context(builder.builder(**config))

    obs = env.reset().observation['WORLD.RGB']
    for episode in range(4):
      last_obs = obs
      obs = env.reset().observation['WORLD.RGB']
      with self.assertRaises(
          AssertionError,
          msg=f'Episodes {episode} and {episode+1} match: {last_obs} == {obs}'):
        np.testing.assert_equal(last_obs, obs)

  def test_no_seed_causes_nondeterminism(self):
    config = copy.deepcopy(_TEST_CONFIG).unlock()
    config.env_seed = None

    env1 = self.enter_context(builder.builder(**config))
    env2 = self.enter_context(builder.builder(**config))
    for episode in range(5):
      obs1 = env1.reset().observation['WORLD.RGB']
      obs2 = env2.reset().observation['WORLD.RGB']
      with self.assertRaises(
          AssertionError, msg=f'Episode {episode} match: {obs1} == {obs2}'):
        np.testing.assert_equal(obs1, obs2)

  @parameterized.product(seed=[None, 42, 123, 1337, 12481632])
  def test_episodes_are_randomized_in_lua(self, seed):
    config = copy.deepcopy(_TEST_CONFIG).unlock()
    config.env_seed = seed
    config.lab2d_settings.simulation.map = _LUA_RANDOMIZATION_MAP
    env = self.enter_context(builder.builder(**config))

    obs = env.reset().observation['WORLD.RGB'][_LUA_RANDOMIZED_LINE]
    for episode in range(4):
      last_obs = obs
      obs = env.reset().observation['WORLD.RGB'][_LUA_RANDOMIZED_LINE]
      with self.assertRaises(
          AssertionError,
          msg=f'Episodes {episode} and {episode+1} match: {last_obs} == {obs}'):
        np.testing.assert_equal(last_obs, obs)

  def test_no_seed_causes_nondeterminism_for_lua(self):
    config = copy.deepcopy(_TEST_CONFIG).unlock()
    config.env_seed = None
    config.lab2d_settings.simulation.map = _LUA_RANDOMIZATION_MAP

    env1 = self.enter_context(builder.builder(**config))
    env2 = self.enter_context(builder.builder(**config))
    for episode in range(5):
      obs1 = env1.reset().observation['WORLD.RGB'][_LUA_RANDOMIZED_LINE]
      obs2 = env2.reset().observation['WORLD.RGB'][_LUA_RANDOMIZED_LINE]
      with self.assertRaises(
          AssertionError, msg=f'Episode {episode} match {obs1} == {obs2}'):
        np.testing.assert_equal(obs1, obs2)


if __name__ == '__main__':
  absltest.main()
