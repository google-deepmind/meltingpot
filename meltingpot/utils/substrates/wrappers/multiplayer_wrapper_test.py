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
"""Tests for multiplayer_wrapper."""

from unittest import mock

from absl.testing import absltest
import dm_env
import dmlab2d
from meltingpot.utils.substrates.wrappers import multiplayer_wrapper
import numpy as np

ACT_SPEC = dm_env.specs.BoundedArray(
    shape=(), minimum=0, maximum=4, dtype=np.int8)
ACT_VALUE = np.ones([], dtype=np.int8)
RGB_SPEC = dm_env.specs.Array(shape=(8, 8, 3), dtype=np.int8)
RGB_VALUE = np.ones((8, 8, 3), np.int8)
REWARD_SPEC = dm_env.specs.Array(shape=(), dtype=np.float32)
REWARD_VALUE = np.ones((), dtype=np.float32)


class Lab2DToListsWrapperTest(absltest.TestCase):

  def test_get_num_players(self):
    env = mock.Mock(spec_set=dmlab2d.Environment)
    env.action_spec.return_value = {
        '1.MOVE': ACT_SPEC,
        '2.MOVE': ACT_SPEC,
        '3.MOVE': ACT_SPEC
    }
    wrapped = multiplayer_wrapper.Wrapper(
        env, individual_observation_names=[], global_observation_names=[])
    self.assertEqual(wrapped._get_num_players(), 3)

  def test_get_rewards(self):
    env = mock.Mock(spec_set=dmlab2d.Environment)
    env.action_spec.return_value = {
        '1.MOVE': ACT_SPEC,
        '2.MOVE': ACT_SPEC,
        '3.MOVE': ACT_SPEC
    }
    wrapped = multiplayer_wrapper.Wrapper(
        env, individual_observation_names=[], global_observation_names=[])
    source = {
        '1.RGB': RGB_VALUE,
        '2.RGB': RGB_VALUE * 2,
        '3.RGB': RGB_VALUE * 3,
        '1.REWARD': 10,
        '2.REWARD': 20,
        '3.REWARD': 30,
        'WORLD.RGB': RGB_VALUE
    }
    rewards = wrapped._get_rewards(source)
    self.assertEqual(rewards, [10, 20, 30])

  def test_get_observations(self):
    env = mock.Mock(spec_set=dmlab2d.Environment)
    env.action_spec.return_value = {
        '1.MOVE': ACT_SPEC,
        '2.MOVE': ACT_SPEC,
        '3.MOVE': ACT_SPEC,
    }
    wrapped = multiplayer_wrapper.Wrapper(
        env,
        individual_observation_names=['RGB'],
        global_observation_names=['WORLD.RGB'])
    source = {
        '1.RGB': RGB_VALUE * 1,
        '2.RGB': RGB_VALUE * 2,
        '3.RGB': RGB_VALUE * 3,
        '1.OTHER': RGB_SPEC,
        '2.OTHER': RGB_SPEC,
        '3.OTHER': RGB_SPEC,
        'WORLD.RGB': RGB_VALUE,
    }
    actual = wrapped._get_observations(source)
    expected = [
        {'RGB': RGB_VALUE * 1, 'WORLD.RGB': RGB_VALUE},
        {'RGB': RGB_VALUE * 2, 'WORLD.RGB': RGB_VALUE},
        {'RGB': RGB_VALUE * 3, 'WORLD.RGB': RGB_VALUE},
    ]
    np.testing.assert_equal(actual, expected)

  def test_get_action(self):
    env = mock.Mock(spec_set=dmlab2d.Environment)
    env.action_spec.return_value = {
        '1.MOVE': ACT_SPEC,
        '2.MOVE': ACT_SPEC,
        '3.MOVE': ACT_SPEC,
    }
    wrapped = multiplayer_wrapper.Wrapper(
        env,
        individual_observation_names=[],
        global_observation_names=[])
    source = [
        {'MOVE': ACT_VALUE * 1},
        {'MOVE': ACT_VALUE * 2},
        {'MOVE': ACT_VALUE * 3},
    ]
    actual = wrapped._get_action(source)
    expected = {
        '1.MOVE': ACT_VALUE * 1,
        '2.MOVE': ACT_VALUE * 2,
        '3.MOVE': ACT_VALUE * 3,
    }
    np.testing.assert_equal(actual, expected)

  def test_spec(self):
    env = mock.Mock(spec_set=dmlab2d.Environment)
    env.action_spec.return_value = {
        '1.MOVE': ACT_SPEC,
        '2.MOVE': ACT_SPEC,
        '3.MOVE': ACT_SPEC,
    }
    env.observation_spec.return_value = {
        '1.RGB': RGB_SPEC,
        '2.RGB': RGB_SPEC,
        '3.RGB': RGB_SPEC,
        '1.OTHER': RGB_SPEC,
        '2.OTHER': RGB_SPEC,
        '3.OTHER': RGB_SPEC,
        '1.REWARD': REWARD_SPEC,
        '2.REWARD': REWARD_SPEC,
        '3.REWARD': REWARD_SPEC,
        'WORLD.RGB': RGB_SPEC
    }
    wrapped = multiplayer_wrapper.Wrapper(
        env,
        individual_observation_names=['RGB'],
        global_observation_names=['WORLD.RGB'])

    with self.subTest('action_spec'):
      self.assertEqual(wrapped.action_spec(), [
          {'MOVE': ACT_SPEC.replace(name='MOVE')},
          {'MOVE': ACT_SPEC.replace(name='MOVE')},
          {'MOVE': ACT_SPEC.replace(name='MOVE')},
      ])
    with self.subTest('observation_spec'):
      self.assertEqual(wrapped.observation_spec(), [
          {'RGB': RGB_SPEC, 'WORLD.RGB': RGB_SPEC},
          {'RGB': RGB_SPEC, 'WORLD.RGB': RGB_SPEC},
          {'RGB': RGB_SPEC, 'WORLD.RGB': RGB_SPEC},
      ])
    with self.subTest('reward_spec'):
      self.assertEqual(
          wrapped.reward_spec(), [REWARD_SPEC, REWARD_SPEC, REWARD_SPEC])

  def test_step(self):
    env = mock.Mock(spec_set=dmlab2d.Environment)
    env.action_spec.return_value = {
        '1.MOVE': ACT_VALUE * 1,
        '2.MOVE': ACT_VALUE * 2,
        '3.MOVE': ACT_VALUE * 3,
    }
    env.step.return_value = dm_env.transition(1, {
        '1.RGB': RGB_VALUE * 1,
        # Intentionally missing 2.RGB
        '3.RGB': RGB_VALUE * 3,
        '1.OTHER': RGB_VALUE,
        '2.OTHER': RGB_VALUE,
        '3.OTHER': RGB_VALUE,
        '1.REWARD': REWARD_VALUE * 10,
        '2.REWARD': REWARD_VALUE * 20,
        '3.REWARD': REWARD_VALUE * 30,
        'WORLD.RGB': RGB_VALUE,
    })
    wrapped = multiplayer_wrapper.Wrapper(
        env,
        individual_observation_names=['RGB'],
        global_observation_names=['WORLD.RGB'])

    actions = [
        {'MOVE': ACT_VALUE * 1},
        {'MOVE': ACT_VALUE * 2},
        {'MOVE': ACT_VALUE * 3},
    ]
    actual = wrapped.step(actions)
    expected = dm_env.transition(
        reward=[
            REWARD_VALUE * 10,
            REWARD_VALUE * 20,
            REWARD_VALUE * 30,
        ],
        observation=[
            {'RGB': RGB_VALUE * 1, 'WORLD.RGB': RGB_VALUE},
            {'WORLD.RGB': RGB_VALUE},
            {'RGB': RGB_VALUE * 3, 'WORLD.RGB': RGB_VALUE},
        ])

    with self.subTest('timestep'):
      np.testing.assert_equal(actual, expected)
    with self.subTest('action'):
      (action,), _ = env.step.call_args
      np.testing.assert_equal(action, {
          '1.MOVE': ACT_VALUE * 1,
          '2.MOVE': ACT_VALUE * 2,
          '3.MOVE': ACT_VALUE * 3,
      })

  def test_reset(self):
    env = mock.Mock(spec_set=dmlab2d.Environment)
    env.action_spec.return_value = {
        '1.MOVE': ACT_VALUE * 1,
        '2.MOVE': ACT_VALUE * 2,
        '3.MOVE': ACT_VALUE * 3,
    }
    env.reset.return_value = dm_env.restart({
        '1.RGB': RGB_VALUE * 1,
        '2.RGB': RGB_VALUE * 2,
        '3.RGB': RGB_VALUE * 3,
        '1.OTHER': RGB_VALUE,
        '2.OTHER': RGB_VALUE,
        '3.OTHER': RGB_VALUE,
        '1.REWARD': REWARD_VALUE * 0,
        '2.REWARD': REWARD_VALUE * 0,
        '3.REWARD': REWARD_VALUE * 0,
        'WORLD.RGB': RGB_VALUE,
    })
    wrapped = multiplayer_wrapper.Wrapper(
        env,
        individual_observation_names=['RGB'],
        global_observation_names=['WORLD.RGB'])
    actual = wrapped.reset()
    expected = dm_env.TimeStep(
        step_type=dm_env.StepType.FIRST,
        reward=[
            REWARD_VALUE * 0,
            REWARD_VALUE * 0,
            REWARD_VALUE * 0,
        ],
        discount=0.,
        observation=[
            {'RGB': RGB_VALUE * 1, 'WORLD.RGB': RGB_VALUE},
            {'RGB': RGB_VALUE * 2, 'WORLD.RGB': RGB_VALUE},
            {'RGB': RGB_VALUE * 3, 'WORLD.RGB': RGB_VALUE},
        ])
    np.testing.assert_equal(actual, expected)

  def test_observation(self):
    env = mock.Mock(spec_set=dmlab2d.Environment)
    env.action_spec.return_value = {
        '1.MOVE': ACT_SPEC,
        '2.MOVE': ACT_SPEC,
        '3.MOVE': ACT_SPEC,
    }
    env.observation.return_value = {
        '1.RGB': RGB_VALUE * 1,
        '2.RGB': RGB_VALUE * 2,
        '3.RGB': RGB_VALUE * 3,
        '1.OTHER': RGB_VALUE,
        '2.OTHER': RGB_VALUE,
        '3.OTHER': RGB_VALUE,
        '1.REWARD': REWARD_VALUE * 0,
        '2.REWARD': REWARD_VALUE * 0,
        '3.REWARD': REWARD_VALUE * 0,
        'WORLD.RGB': RGB_VALUE,
    }
    wrapped = multiplayer_wrapper.Wrapper(
        env,
        individual_observation_names=['RGB'],
        global_observation_names=['WORLD.RGB'])
    actual = wrapped.observation()
    expected = [
        {'RGB': RGB_VALUE * 1, 'WORLD.RGB': RGB_VALUE},
        {'RGB': RGB_VALUE * 2, 'WORLD.RGB': RGB_VALUE},
        {'RGB': RGB_VALUE * 3, 'WORLD.RGB': RGB_VALUE},
    ]
    np.testing.assert_equal(actual, expected)

if __name__ == '__main__':
  absltest.main()
