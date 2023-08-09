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
"""Tests for discrete_action_wrapper."""

from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import dm_env
import dmlab2d
from meltingpot.utils.substrates.wrappers import discrete_action_wrapper
import numpy as np

MOVE_SPEC = dm_env.specs.BoundedArray(
    shape=(), minimum=0, maximum=3, dtype=np.int8, name='MOVE')
TURN_SPEC = dm_env.specs.BoundedArray(
    shape=(), minimum=0, maximum=3, dtype=np.int8, name='TURN')
VALID_VALUE_0 = np.zeros([], dtype=np.int8)
VALID_VALUE_1 = np.array(3, dtype=np.int8)
INVALID_VALUE = np.array(4, dtype=np.int8)


class Lab2DToListsWrapperTest(parameterized.TestCase):

  def test_valid_set(self):
    env = mock.Mock(spec_set=dmlab2d.Environment)
    env.action_spec.return_value = [
        {'MOVE': MOVE_SPEC, 'TURN': TURN_SPEC},
        {'MOVE': MOVE_SPEC, 'TURN': TURN_SPEC},
    ]
    discrete_action_wrapper.Wrapper(env, action_table=[
        {'MOVE': VALID_VALUE_0, 'TURN': VALID_VALUE_0},
        {'MOVE': VALID_VALUE_0, 'TURN': VALID_VALUE_1},
        {'MOVE': VALID_VALUE_1, 'TURN': VALID_VALUE_0},
        {'MOVE': VALID_VALUE_1, 'TURN': VALID_VALUE_1},
    ])

  @parameterized.named_parameters(
      ('empty', []),
      ('out_of_bounds', [{'MOVE': INVALID_VALUE, 'TURN': VALID_VALUE_0}]),
      ('missing_key', [{'TURN': VALID_VALUE_0}]),
      ('extra_key', [{'INVALID': VALID_VALUE_0}]),
  )
  def test_invalid_set(self, action_table):
    env = mock.Mock(spec_set=dmlab2d.Environment)
    env.action_spec.return_value = [
        {'MOVE': MOVE_SPEC, 'TURN': TURN_SPEC},
        {'MOVE': MOVE_SPEC, 'TURN': TURN_SPEC},
    ]
    with self.assertRaises(ValueError):
      discrete_action_wrapper.Wrapper(env, action_table=action_table)

  def test_action_spec(self):
    env = mock.Mock(spec_set=dmlab2d.Environment)
    env.action_spec.return_value = [
        {'MOVE': MOVE_SPEC, 'TURN': TURN_SPEC},
        {'MOVE': MOVE_SPEC, 'TURN': TURN_SPEC},
    ]
    wrapped = discrete_action_wrapper.Wrapper(env, action_table=[
        {'MOVE': VALID_VALUE_0, 'TURN': VALID_VALUE_0},
        {'MOVE': VALID_VALUE_0, 'TURN': VALID_VALUE_1},
        {'MOVE': VALID_VALUE_1, 'TURN': VALID_VALUE_0},
    ])
    actual = wrapped.action_spec()
    expected = (
        dm_env.specs.DiscreteArray(num_values=3, dtype=np.int64, name='action'),
    ) * 2
    self.assertEqual(actual, expected)

  def test_step(self):
    env = mock.Mock(spec_set=dmlab2d.Environment)
    env.action_spec.return_value = [
        {'MOVE': MOVE_SPEC, 'TURN': TURN_SPEC},
        {'MOVE': MOVE_SPEC, 'TURN': TURN_SPEC},
    ]
    env.step.return_value = mock.sentinel.timestep
    wrapped = discrete_action_wrapper.Wrapper(env, action_table=[
        {'MOVE': VALID_VALUE_0, 'TURN': VALID_VALUE_0},
        {'MOVE': VALID_VALUE_0, 'TURN': VALID_VALUE_1},
        {'MOVE': VALID_VALUE_1, 'TURN': VALID_VALUE_0},
    ])
    actual = wrapped.step([0, 2])

    with self.subTest('timestep'):
      np.testing.assert_equal(actual, mock.sentinel.timestep)
    with self.subTest('action'):
      (action,), _ = env.step.call_args
      self.assertEqual(action, [
          {'MOVE': VALID_VALUE_0, 'TURN': VALID_VALUE_0},
          {'MOVE': VALID_VALUE_1, 'TURN': VALID_VALUE_0},
      ])


if __name__ == '__main__':
  absltest.main()
