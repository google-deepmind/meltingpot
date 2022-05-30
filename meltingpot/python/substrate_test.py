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

import dataclasses

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

from meltingpot.python import substrate
from meltingpot.python.utils.substrates import specs


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
      spec = specs.action(len(config.action_set))
      self.assertSequenceEqual(action_spec, [spec] * config.num_players)
    with self.subTest('reward_spec'):
      self.assertSequenceEqual(reward_spec, [specs.REWARD] * config.num_players)

  def test_observables(self):
    config = substrate.get_config('running_with_scissors_in_the_matrix')
    with substrate.build(config) as env:
      received = []
      observables = env.observables()
      for field in dataclasses.fields(observables):
        getattr(observables, field.name).subscribe(
            on_next=received.append,
            on_error=lambda e: received.append(type(e)),
            on_completed=lambda: received.append('DONE'),
        )

      expected = []
      timestep = env.reset()
      events = list(env.events())
      expected.extend([timestep] + events)
      for n in range(2):
        action = [n] * config.num_players
        timestep = env.step(action)
        events = list(env.events())
        expected.extend([action, timestep] + events)
      expected.extend(['DONE', 'DONE', 'DONE'])

    self.assertEqual(received, expected)

  @parameterized.named_parameters(
      (name, name) for name in substrate.AVAILABLE_SUBSTRATES)
  def test_observations_match_observation_spec(self, name):
    config = substrate.get_config(name)
    with substrate.build(config) as env:
      observation_spec = env.observation_spec()[0]
      timestep = env.reset()
      observation = timestep.observation[0]

    with self.subTest('missing'):
      self.assertSameElements(observation_spec, observation)
    for name, spec in observation_spec.items():
      with self.subTest(name):
        spec.validate(observation[name])

  @parameterized.named_parameters(
      (name, name) for name in substrate.AVAILABLE_SUBSTRATES)
  def test_spec_in_config_matches_environment(self, name):
    config = substrate.get_config(name)
    with substrate.build(config) as env:
      action_spec = env.action_spec()[0]
      discount_spec = env.discount_spec()
      reward_spec = env.reward_spec()[0]
      observation_spec = env.observation_spec()[0]

    with self.subTest('action_spec'):
      self.assertEqual(action_spec, config.action_spec)

    with self.subTest('reward_spec'):
      self.assertEqual(reward_spec, config.timestep_spec.reward)

    with self.subTest('discount_spec'):
      self.assertEqual(discount_spec, config.timestep_spec.discount)

    with self.subTest('missing'):
      self.assertSameElements(
          observation_spec, config.timestep_spec.observation)

    for name, spec in observation_spec.items():
      with self.subTest(f'observation_spec {name}'):
        self.assertEqual(spec, config.timestep_spec.observation[name])

if __name__ == '__main__':
  absltest.main()
