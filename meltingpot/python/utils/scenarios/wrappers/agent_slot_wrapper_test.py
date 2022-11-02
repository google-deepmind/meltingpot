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
"""Tests for agent_slot_wrapper."""

from unittest import mock

from absl.testing import absltest
import dm_env
import numpy as np

from meltingpot.python.utils.scenarios.wrappers import agent_slot_wrapper
from meltingpot.python.utils.substrates import substrate

AGENT_SLOT = agent_slot_wrapper.AGENT_SLOT
RGB_SPEC = dm_env.specs.Array(shape=(8, 8, 3), dtype=np.int8)
RGB_VALUE = np.ones((8, 8, 3), np.int8)


class AgentSlotWrapperTest(absltest.TestCase):

  def test_augment_timestep(self):
    timestep = dm_env.restart([{'RGB': RGB_VALUE}] * 3)
    actual = agent_slot_wrapper._augment_timestep(timestep)
    expected = dm_env.restart([
        {
            'RGB': RGB_VALUE,
            AGENT_SLOT: np.array([1, 0, 0], dtype=np.float32)
        },
        {
            'RGB': RGB_VALUE,
            AGENT_SLOT: np.array([0, 1, 0], dtype=np.float32)
        },
        {
            'RGB': RGB_VALUE,
            AGENT_SLOT: np.array([0, 0, 1], dtype=np.float32)
        },
    ])

    with self.subTest('not_inplace'):
      self.assertIsNot(actual.observation, timestep.observation)
    with self.subTest('not_inplace'):
      np.testing.assert_equal(actual, expected)

  def test_added_slot(self):
    env = mock.Mock(spec_set=substrate.Substrate)
    env.events.return_value = ()
    env.observation_spec.return_value = [{'RGB': RGB_SPEC}] * 3
    env.reset.return_value = dm_env.restart([{'RGB': RGB_VALUE}] * 3)
    env.step.return_value = dm_env.transition(1, [{'RGB': RGB_VALUE}] * 3)

    wrapped = agent_slot_wrapper.Wrapper(env)

    expected_spec = [{
        'RGB':
            RGB_SPEC,
        AGENT_SLOT:
            dm_env.specs.Array(shape=[3], dtype=np.float32, name=AGENT_SLOT),
    }] * 3
    expected_observations = [
        {
            'RGB': RGB_VALUE,
            AGENT_SLOT: np.array([1, 0, 0], dtype=np.float32)
        },
        {
            'RGB': RGB_VALUE,
            AGENT_SLOT: np.array([0, 1, 0], dtype=np.float32)
        },
        {
            'RGB': RGB_VALUE,
            AGENT_SLOT: np.array([0, 0, 1], dtype=np.float32)
        },
    ]

    with self.subTest('observation_spec'):
      self.assertEqual(wrapped.observation_spec(), expected_spec)
    with self.subTest('reset'):
      np.testing.assert_equal(wrapped.reset(),
                              dm_env.restart(expected_observations))
    with self.subTest('observation'):
      np.testing.assert_equal(
          wrapped.step(mock.sentinel.action),
          dm_env.transition(1, expected_observations))


if __name__ == '__main__':
  absltest.main()
