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
"""Reject invalid discrete action batches before the environment advances."""

from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import dm_env
import dmlab2d
from meltingpot import substrate
from meltingpot.utils.substrates.wrappers import discrete_action_wrapper
import numpy as np


class DiscreteActionValidationTest(parameterized.TestCase):

  def make_wrapper(self):
    env = mock.Mock(spec_set=dmlab2d.Environment)
    action_spec = {'move': dm_env.specs.BoundedArray((), np.int32, 0, 3)}
    env.action_spec.return_value = [action_spec, action_spec]
    env.step.return_value = dm_env.transition((1.0, 2.0), ({}, {}))
    wrapped = discrete_action_wrapper.Wrapper(
        env, action_table=[{'move': value} for value in range(4)]
    )
    self.addCleanup(wrapped.close)
    return wrapped, env

  @parameterized.parameters(-1, -4, -5, 4, 100, np.int64(-1))
  def test_out_of_range_indices_never_select_a_table_row(self, invalid):
    wrapped, env = self.make_wrapper()
    for index in (0, 1):
      with self.subTest(player=index):
        actions = [0, 0]
        actions[index] = invalid
        with self.assertRaisesRegex(ValueError, 'action'):
          wrapped.step(actions)
        env.step.assert_not_called()
    self.assertIs(wrapped.step([0, 3]), env.step.return_value)
    self.assertEqual(
        [int(action['move']) for action in env.step.call_args.args[0]], [0, 3]
    )

  @parameterized.parameters(0, 1, 3)
  def test_wrong_player_counts_leave_the_environment_untouched(self, count):
    wrapped, env = self.make_wrapper()
    with self.assertRaisesRegex(ValueError, 'actions'):
      wrapped.step([0] * count)
    env.step.assert_not_called()

  @parameterized.parameters(int, np.int32, np.int64, np.uint64)
  def test_integer_types_and_zero_dimensional_arrays_keep_action_mapping(
      self, dtype
  ):
    wrapped, env = self.make_wrapper()
    for actions in (
        [dtype(0), dtype(3)],
        np.asarray([1, 2], dtype=dtype),
        [np.asarray(2, dtype=dtype), np.asarray(0, dtype=dtype)],
    ):
      expected = [int(action) for action in actions]
      self.assertIs(wrapped.step(actions), env.step.return_value)
      actual = env.step.call_args.args[0]
      self.assertEqual([int(action['move']) for action in actual], expected)
      for action in actual:
        self.assertEqual(action['move'].dtype, np.int32)
        self.assertFalse(action['move'].flags.writeable)
      self.assertEqual([int(action) for action in actions], expected)

  @parameterized.parameters((1.0,), ('1',), (np.array([1]),))
  def test_noninteger_scalars_are_not_silently_coerced(self, invalid):
    wrapped, env = self.make_wrapper()
    with self.assertRaises(TypeError):
      wrapped.step([0, invalid])
    env.step.assert_not_called()

  def test_real_substrate_rejects_negative_actions_without_changing_observations(
      self,
  ):
    config = substrate.get_config('commons_harvest__open')
    env = substrate.build_from_config(config, roles=config.default_player_roles)
    self.addCleanup(env.close)
    env.reset()
    before = [
        {key: value.copy() for key, value in obs.items()}
        for obs in env.observation()
    ]
    raw_actions = []
    env.observables().dmlab2d.action.subscribe(raw_actions.append)
    action = [0] * len(env.action_spec())
    action[-1] = -1
    with self.assertRaises(ValueError):
      env.step(action)
    self.assertEmpty(raw_actions)
    np.testing.assert_equal(env.observation(), before)
    action[-1] = 0
    self.assertTrue(env.step(action).mid())
    self.assertLen(raw_actions, 1)


if __name__ == '__main__':
  absltest.main()
