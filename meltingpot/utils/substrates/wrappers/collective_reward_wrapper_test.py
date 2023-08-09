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
from meltingpot.utils.substrates.wrappers import collective_reward_wrapper
import numpy as np

RGB_SPEC = dm_env.specs.Array(shape=(2, 1), dtype=np.int8)
COLLECTIVE_REWARD_SPEC = dm_env.specs.Array(shape=(), dtype=np.float64)

NUM_PLAYERS = 3
REWARDS = [1.0, 2.0, 3.0]
RGB = np.zeros((2, 1))


class CollectiveRewardWrapperTest(absltest.TestCase):

  def test_get_timestep(self):
    env = mock.Mock(spec_set=dmlab2d.Environment)
    wrapped = collective_reward_wrapper.CollectiveRewardWrapper(env)

    source = dm_env.TimeStep(
        step_type=dm_env.StepType.MID,
        reward=REWARDS,
        discount=1.0,
        observation=[{'RGB': RGB} for _ in range(NUM_PLAYERS)])
    actual = wrapped._get_timestep(source)
    added_key = collective_reward_wrapper._COLLECTIVE_REWARD_OBS
    collective_reward = np.sum(REWARDS)
    expected_observation = [
        {'RGB': RGB, added_key: collective_reward},
    ] * NUM_PLAYERS
    expected_timestep = dm_env.TimeStep(
        step_type=dm_env.StepType.MID,
        reward=REWARDS,
        discount=1.0,
        observation=expected_observation)
    np.testing.assert_equal(actual, expected_timestep)

  def test_spec(self):
    env = mock.Mock(spec_set=dmlab2d.Environment)
    env.observation_spec.return_value = [{
        'RGB': RGB_SPEC,
    }] * NUM_PLAYERS
    wrapped = collective_reward_wrapper.CollectiveRewardWrapper(env)

    added_key = collective_reward_wrapper._COLLECTIVE_REWARD_OBS
    self.assertEqual(wrapped.observation_spec(), [
        {'RGB': RGB_SPEC, added_key: COLLECTIVE_REWARD_SPEC}] * NUM_PLAYERS)

if __name__ == '__main__':
  absltest.main()
