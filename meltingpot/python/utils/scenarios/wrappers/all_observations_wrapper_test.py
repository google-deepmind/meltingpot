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
import immutabledict
import numpy as np

from meltingpot.python.utils.scenarios.wrappers import all_observations_wrapper
from meltingpot.python.utils.substrates import substrate

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


def _restart(observation, reward):
  return dm_env.restart(observation=observation)._replace(reward=reward)


class _ExpectedArray(np.ndarray):
  """Used for test __eq__ comparisons involving nested numpy arrays."""

  def __eq__(self, other):
    if not isinstance(other, np.ndarray):
      return NotImplemented
    elif self.shape != other.shape:
      return False
    elif self.dtype != other.dtype:
      return False
    else:
      return super().__eq__(other).all()


def _expect_array(*args, **kwargs):
  value = np.array(*args, **kwargs)
  return _ExpectedArray(value.shape, value.dtype, value)


def _expect_zeros(*args, **kwargs):
  value = np.zeros(*args, **kwargs)
  return _ExpectedArray(value.shape, value.dtype, value)


def _expect_ones(*args, **kwargs):
  value = np.ones(*args, **kwargs)
  return _ExpectedArray(value.shape, value.dtype, value)


class AllObservationsWrapperTest(parameterized.TestCase):

  def test_observation_spec(self):
    env = mock.Mock(spec_set=substrate.Substrate)
    env.events.return_value = ()
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
    expected = ({
        OBSERVATION_1:
            dm_env.specs.Array(shape=[1], dtype=np.float32),
        OBSERVATION_2:
            dm_env.specs.Array(shape=[2], dtype=np.float32),
        GLOBAL_KEY:
            immutabledict.immutabledict({
                OBSERVATIONS_KEY:
                    immutabledict.immutabledict({
                        OBSERVATION_1:
                            dm_env.specs.Array(
                                shape=[2, 1],
                                dtype=np.float32,
                                name=OBSERVATION_1)
                    }),
                REWARDS_KEY:
                    REWARD_SPEC.replace(shape=[2], name=REWARDS_KEY),
                ACTIONS_KEY:
                    dm_env.specs.BoundedArray(
                        shape=[2],
                        dtype=ACTION_SPEC.dtype,
                        minimum=ACTION_SPEC.minimum,
                        maximum=ACTION_SPEC.maximum,
                        name=ACTIONS_KEY),
            })
    },) * 2
    self.assertEqual(actual, expected)

  def test_reset(self):
    env = mock.Mock(spec_set=substrate.Substrate)
    env.events.return_value = ()
    env.action_spec.return_value = (ACTION_SPEC,) * 2
    env.reward_spec.return_value = (REWARD_SPEC,) * 2
    env.reset.return_value = _restart(
        reward=(
            np.array(0., dtype=REWARD_SPEC.dtype),
            np.array(0., dtype=REWARD_SPEC.dtype),
        ),
        observation=(
            immutabledict.immutabledict({
                OBSERVATION_1: np.ones([1], dtype=np.float32),
                OBSERVATION_2: np.ones([2], dtype=np.float32),
            }),
            immutabledict.immutabledict({
                OBSERVATION_1: np.ones([1], dtype=np.float32) * 2,
                OBSERVATION_2: np.ones([2], dtype=np.float32) * 2,
            }),
        ),
    )
    wrapped = all_observations_wrapper.Wrapper(
        env,
        observations_to_share=[OBSERVATION_1],
        share_actions=True,
        share_rewards=True)

    actual = wrapped.reset()
    expected = _restart(
        reward=(
            _expect_array(0, dtype=REWARD_SPEC.dtype),
            _expect_array(0, dtype=REWARD_SPEC.dtype),
        ),
        observation=(
            immutabledict.immutabledict({
                OBSERVATION_1:
                    _expect_ones([1], dtype=np.float32),
                OBSERVATION_2:
                    _expect_ones([2], dtype=np.float32),
                GLOBAL_KEY:
                    immutabledict.immutabledict({
                        OBSERVATIONS_KEY:
                            immutabledict.immutabledict({
                                OBSERVATION_1:
                                    _expect_array([[1.], [2.]],
                                                  dtype=np.float32)
                            }),
                        REWARDS_KEY:
                            _expect_zeros([2], dtype=REWARD_SPEC.dtype),
                        ACTIONS_KEY:
                            _expect_zeros([2], dtype=ACTION_SPEC.dtype),
                    }),
            }),
            immutabledict.immutabledict({
                OBSERVATION_1:
                    _expect_ones([1], dtype=np.float32) * 2,
                OBSERVATION_2:
                    _expect_ones([2], dtype=np.float32) * 2,
                GLOBAL_KEY:
                    immutabledict.immutabledict({
                        OBSERVATIONS_KEY:
                            immutabledict.immutabledict({
                                OBSERVATION_1:
                                    _expect_array([[1.], [2.]],
                                                  dtype=np.float32)
                            }),
                        REWARDS_KEY:
                            _expect_zeros([2], dtype=REWARD_SPEC.dtype),
                        ACTIONS_KEY:
                            _expect_zeros([2], dtype=ACTION_SPEC.dtype),
                    }),
            }),
        ),
    )
    self.assertEqual(actual, expected)

  def test_step(self):
    env = mock.Mock(spec_set=substrate.Substrate)
    env.events.return_value = ()
    env.action_spec.return_value = [ACTION_SPEC] * 2
    env.reward_spec.return_value = [REWARD_SPEC] * 2
    env.step.return_value = dm_env.transition(
        reward=(
            np.array(1, dtype=REWARD_SPEC.dtype),
            np.array(2, dtype=REWARD_SPEC.dtype),
        ),
        observation=(
            immutabledict.immutabledict({
                OBSERVATION_1: _expect_ones([1], dtype=np.float32),
                OBSERVATION_2: _expect_ones([2], dtype=np.float32),
            }),
            immutabledict.immutabledict({
                OBSERVATION_1: _expect_ones([1], dtype=np.float32) * 2,
                OBSERVATION_2: _expect_ones([2], dtype=np.float32) * 2,
            }),
        ),
    )
    wrapped = all_observations_wrapper.Wrapper(
        env,
        observations_to_share=[OBSERVATION_1],
        share_actions=True,
        share_rewards=True)

    actual = wrapped.step([3, 4])
    expected = dm_env.transition(
        reward=(
            _expect_array(1, dtype=REWARD_SPEC.dtype),
            _expect_array(2, dtype=REWARD_SPEC.dtype),
        ),
        observation=(
            immutabledict.immutabledict({
                OBSERVATION_1:
                    _expect_ones([1], dtype=np.float32),
                OBSERVATION_2:
                    _expect_ones([2], dtype=np.float32),
                GLOBAL_KEY:
                    immutabledict.immutabledict({
                        OBSERVATIONS_KEY:
                            immutabledict.immutabledict({
                                OBSERVATION_1:
                                    _expect_array([[1.], [2.]],
                                                  dtype=np.float32)
                            }),
                        REWARDS_KEY:
                            _expect_array([1, 2], dtype=REWARD_SPEC.dtype),
                        ACTIONS_KEY:
                            _expect_array([3, 4], dtype=ACTION_SPEC.dtype),
                    }),
            }),
            immutabledict.immutabledict({
                OBSERVATION_1:
                    _expect_ones([1], dtype=np.float32) * 2,
                OBSERVATION_2:
                    _expect_ones([2], dtype=np.float32) * 2,
                GLOBAL_KEY:
                    immutabledict.immutabledict({
                        OBSERVATIONS_KEY:
                            immutabledict.immutabledict({
                                OBSERVATION_1:
                                    _expect_array([[1.], [2.]],
                                                  dtype=np.float32)
                            }),
                        REWARDS_KEY:
                            _expect_array([1, 2], dtype=REWARD_SPEC.dtype),
                        ACTIONS_KEY:
                            _expect_array([3, 4], dtype=ACTION_SPEC.dtype),
                    }),
            }),
        ),
    )
    self.assertEqual(actual, expected)

  def test_can_be_applied_twice(self):
    env = mock.Mock(spec_set=substrate.Substrate)
    env.events.return_value = ()
    env.observation_spec.return_value = [{
        OBSERVATION_1: dm_env.specs.Array(shape=[1], dtype=np.float32),
        OBSERVATION_2: dm_env.specs.Array(shape=[2], dtype=np.float32),
    }] * 2
    env.action_spec.return_value = (ACTION_SPEC,) * 2
    env.reward_spec.return_value = (REWARD_SPEC,) * 2
    env.reset.return_value = _restart(
        reward=(
            _expect_array(0, dtype=REWARD_SPEC.dtype),
            _expect_array(0, dtype=REWARD_SPEC.dtype),
        ),
        observation=(
            immutabledict.immutabledict({
                OBSERVATION_1: _expect_ones([1], dtype=np.float32),
                OBSERVATION_2: _expect_ones([2], dtype=np.float32),
            }),
            immutabledict.immutabledict({
                OBSERVATION_1: _expect_ones([1], dtype=np.float32) * 2,
                OBSERVATION_2: _expect_ones([2], dtype=np.float32) * 2,
            }),
        ),
    )

    wrapped = all_observations_wrapper.Wrapper(
        env,
        observations_to_share=[OBSERVATION_1],
        share_actions=False,
        share_rewards=True)
    double_wrapped = all_observations_wrapper.Wrapper(
        wrapped,
        observations_to_share=[OBSERVATION_2],
        share_actions=True,
        share_rewards=False)

    expected_equivalent = all_observations_wrapper.Wrapper(
        env,
        observations_to_share=[OBSERVATION_1, OBSERVATION_2],
        share_actions=True,
        share_rewards=True)

    double_wrapped_shared = double_wrapped.reset().observation[0][GLOBAL_KEY]
    expected_shared = expected_equivalent.reset().observation[0][GLOBAL_KEY]

    with self.subTest('specs'):
      self.assertEqual(double_wrapped.observation_spec()[0][GLOBAL_KEY],
                       expected_equivalent.observation_spec()[0][GLOBAL_KEY])
    with self.subTest('shared_rewards'):
      np.testing.assert_equal(double_wrapped_shared[REWARDS_KEY],
                              expected_shared[REWARDS_KEY])
    with self.subTest('shared_actions'):
      np.testing.assert_equal(double_wrapped_shared[ACTIONS_KEY],
                              expected_shared[ACTIONS_KEY])
    with self.subTest('shared_observations'):
      np.testing.assert_equal(
          dict(double_wrapped_shared[OBSERVATIONS_KEY]),
          dict(expected_shared[OBSERVATIONS_KEY]))

if __name__ == '__main__':
  absltest.main()
