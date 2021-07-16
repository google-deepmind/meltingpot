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
"""Tests for all_observations_wrapper."""

from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import dm_env
import numpy as np

from meltingpot.python.utils.scenarios.wrappers import all_observations_wrapper
from meltingpot.python.utils.scenarios.wrappers import base

GLOBAL_KEY = all_observations_wrapper.GLOBAL_KEY
REWARDS_KEY = all_observations_wrapper.REWARDS_KEY
OBSERVATIONS_KEY = all_observations_wrapper.OBSERVATIONS_KEY
ACTIONS_KEY = all_observations_wrapper.ACTIONS_KEY

ACTION_1 = 'action_1'
ACTION_2 = 'action_2'
OBSERVATION_1 = 'observation_1'
OBSERVATION_2 = 'observation_2'
ACTION_SPEC = dm_env.specs.DiscreteArray(num_values=5, dtype=np.int32)
REWARD_SPEC = dm_env.specs.Array(shape=[], dtype=np.float32)


class AllObservationsWrapperTest(parameterized.TestCase):

  def test_observation_spec(self):
    env = mock.Mock(spec_set=base.Substrate)
    env.observation_spec.return_value = [{
        OBSERVATION_1: dm_env.specs.Array(shape=[1], dtype=np.float32),
        OBSERVATION_2: dm_env.specs.Array(shape=[2], dtype=np.float32),
    }] * 2
    env.action_spec.return_value = [ACTION_SPEC] * 2
    env.reward_spec.return_value = [REWARD_SPEC] * 2

    wrapped = all_observations_wrapper.Wrapper(
        env,
        observations_to_share=[OBSERVATION_1],
        share_actions=True,
        share_rewards=True)

    actual = wrapped.observation_spec()
    expected = [{
        OBSERVATION_1: dm_env.specs.Array(shape=[1], dtype=np.float32),
        OBSERVATION_2: dm_env.specs.Array(shape=[2], dtype=np.float32),
        GLOBAL_KEY: {
            OBSERVATIONS_KEY: {
                OBSERVATION_1:
                    dm_env.specs.Array(
                        shape=[2, 1], dtype=np.float32, name=OBSERVATION_1)
            },
            REWARDS_KEY: REWARD_SPEC.replace(shape=[2], name=REWARDS_KEY),
            ACTIONS_KEY: dm_env.specs.BoundedArray(
                shape=[2], dtype=ACTION_SPEC.dtype, minimum=ACTION_SPEC.minimum,
                maximum=ACTION_SPEC.maximum, name=ACTIONS_KEY),
        }
    }] * 2
    self.assertEqual(actual, expected)

  def test_reset(self):
    env = mock.Mock(spec_set=base.Substrate)
    env.action_spec.return_value = [ACTION_SPEC] * 2
    env.reward_spec.return_value = [REWARD_SPEC] * 2
    env.reset.return_value = dm_env.restart([
        {
            OBSERVATION_1: np.ones([1]),
            OBSERVATION_2: np.ones([2]),
        },
        {
            OBSERVATION_1: np.ones([1]) * 2,
            OBSERVATION_2: np.ones([2]) * 2,
        },
    ])._replace(reward=[np.array(0), np.array(0)])
    wrapped = all_observations_wrapper.Wrapper(
        env,
        observations_to_share=[OBSERVATION_1],
        share_actions=True,
        share_rewards=True)

    actual = wrapped.reset()
    expected = dm_env.restart([
        {
            OBSERVATION_1: np.ones([1]),
            OBSERVATION_2: np.ones([2]),
            GLOBAL_KEY: {
                OBSERVATIONS_KEY: {
                    OBSERVATION_1: np.array([[1.], [2.]])
                },
                REWARDS_KEY: np.zeros([2], dtype=np.float32),
                ACTIONS_KEY: np.zeros([2], dtype=np.int32),
            }
        },
        {
            OBSERVATION_1: np.ones([1]) * 2,
            OBSERVATION_2: np.ones([2]) * 2,
            GLOBAL_KEY: {
                OBSERVATIONS_KEY: {
                    OBSERVATION_1: np.array([[1.], [2.]])
                },
                REWARDS_KEY: np.zeros([2], dtype=np.float32),
                ACTIONS_KEY: np.zeros([2], dtype=np.int32),
            }
        },
    ])._replace(reward=[np.array(0), np.array(0)])
    np.testing.assert_equal(actual, expected)

  def test_step(self):
    env = mock.Mock(spec_set=base.Substrate)
    env.action_spec.return_value = [ACTION_SPEC] * 2
    env.reward_spec.return_value = [REWARD_SPEC] * 2
    env.step.return_value = dm_env.transition(
        reward=[np.array(1), np.array(2)],
        observation=[
            {
                OBSERVATION_1: np.ones([1]),
                OBSERVATION_2: np.ones([2]),
            },
            {
                OBSERVATION_1: np.ones([1]) * 2,
                OBSERVATION_2: np.ones([2]) * 2,
            },
        ])
    wrapped = all_observations_wrapper.Wrapper(
        env,
        observations_to_share=[OBSERVATION_1],
        share_actions=True,
        share_rewards=True)

    actual = wrapped.step([
        np.array(3, dtype=np.int32),
        np.array(4, dtype=np.int32),
    ])
    expected = dm_env.transition(
        reward=[np.array(1), np.array(2)],
        observation=[
            {
                OBSERVATION_1: np.ones([1]),
                OBSERVATION_2: np.ones([2]),
                GLOBAL_KEY: {
                    OBSERVATIONS_KEY: {
                        OBSERVATION_1: np.array([[1.], [2.]])
                    },
                    REWARDS_KEY: np.array([1, 2], dtype=np.float32),
                    ACTIONS_KEY: np.array([3, 4], dtype=np.int32),
                }
            },
            {
                OBSERVATION_1: np.ones([1]) * 2,
                OBSERVATION_2: np.ones([2]) * 2,
                GLOBAL_KEY: {
                    OBSERVATIONS_KEY: {
                        OBSERVATION_1: np.array([[1.], [2.]])
                    },
                    REWARDS_KEY: np.array([1, 2], dtype=np.float32),
                    ACTIONS_KEY: np.array([3, 4], dtype=np.int32),
                }
            },
        ],
    )
    np.testing.assert_equal(actual, expected)


if __name__ == '__main__':
  absltest.main()
