# Copyright 2024 DeepMind Technologies Limited.
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


from absl.testing import absltest
from absl.testing import parameterized
import dm_env
from meltingpot.utils.evaluation import return_subject
import numpy as np
from reactivex import subject as rx_subject


def _send_timesteps_to_subject(subject, timesteps):
  results = []
  subject.subscribe(on_next=results.append)

  for n, timestep in enumerate(timesteps):
    subject.on_next(timestep)
    if results:
      return n, results.pop()
  return None, None


class ReturnSubjectTest(absltest.TestCase):

  def test(self):
    timesteps = [
        dm_env.restart(observation=[{}]),
        dm_env.transition(observation=[{}], reward=[2, 4]),
        dm_env.termination(observation=[{}], reward=[1, 3]),
    ]
    subject = return_subject.ReturnSubject()
    step_written, episode_returns = _send_timesteps_to_subject(
        subject, timesteps
    )

    with self.subTest('written_on_final_step'):
      self.assertEqual(step_written, 2)

    with self.subTest('returns'):
      np.testing.assert_equal(episode_returns, [3, 7])


class ReturnAccumulationTest(parameterized.TestCase):

  def accumulate(self, rewards):
    recorder = return_subject.ReturnSubject()
    self.addCleanup(recorder.dispose)
    source = rx_subject.Subject()
    self.addCleanup(source.dispose)
    self.addCleanup(source.subscribe(recorder).dispose)
    results = []
    self.addCleanup(recorder.subscribe(results.append).dispose)
    source.on_next(dm_env.restart(observation=()))
    before = [np.array(reward, copy=True) for reward in rewards]
    for index, reward in enumerate(rewards):
      timestep = (
          dm_env.termination(reward, ())
          if index == len(rewards) - 1
          else dm_env.transition(reward, ())
      )
      source.on_next(timestep)
      if index != len(rewards) - 1:
        self.assertEmpty(results)
    self.assertLen(results, 1)
    self.assertIsInstance(results[0], np.ndarray)
    for reward, original in zip(rewards, before):
      np.testing.assert_array_equal(reward, original)
    return results[0]

  @parameterized.parameters(
      {'rewards': [0, 0.5, 1.25], 'expected': 1.75},
      {
          'rewards': [[0, 0], [0.5, -0.25], [1.25, 0.5]],
          'expected': [1.75, 0.25],
      },
      {
          'rewards': [np.array([0, 0], np.int32), np.array([1.5, -2.5])],
          'expected': [1.5, -2.5],
      },
  )
  def test_fractional_rewards_after_integer_zeros(self, rewards, expected):
    actual = self.accumulate(rewards)
    np.testing.assert_array_equal(actual, expected)
    self.assertEqual(actual.dtype, np.float64)

  @parameterized.parameters(
      (np.int8, 120, 240.0),
      (np.uint8, 200, 400.0),
      (np.int64, 2**62, float(2**63)),
      (np.float16, 65504, 131008.0),
      (np.float32, 3e38, float(np.float32(3e38)) * 2),
  )
  def test_totals_do_not_overflow_the_input_dtype(self, dtype, value, expected):
    reward = np.array([value], dtype=dtype)
    with np.errstate(over='raise', invalid='raise'):
      actual = self.accumulate([reward, reward])
    np.testing.assert_array_equal(actual, [expected])
    self.assertTrue(np.isfinite(actual).all())

  def test_small_float32_rewards_are_not_lost_after_a_large_reward(self):
    rewards = [np.array([2**24], np.float32)]
    rewards += [np.array([1], np.float32)] * 32
    np.testing.assert_array_equal(self.accumulate(rewards), [2**24 + 32])

  def test_wider_floating_types_are_not_narrowed(self):
    reward = np.array([1], dtype=np.longdouble)
    epsilon = np.array([np.finfo(np.longdouble).eps], dtype=np.longdouble)
    for initial in (np.array([0], np.float64), np.array([0], np.longdouble)):
      actual = self.accumulate([initial, reward, epsilon])
      self.assertEqual(actual.dtype, np.result_type(np.float64, np.longdouble))
      np.testing.assert_array_equal(actual, reward + epsilon)

  @parameterized.parameters(np.float64, np.int64, np.float32)
  def test_vector_shape_and_ordinary_values_are_preserved(self, dtype):
    first = np.array([1, 2, 3], dtype=dtype)
    last = np.array([-2, 4, 8], dtype=dtype)
    actual = self.accumulate([first, last])
    self.assertEqual(actual.shape, (3,))
    np.testing.assert_array_equal(actual, [-1, 6, 11])

  def test_empty_populations_still_emit_an_empty_return(self):
    actual = self.accumulate([(), ()])
    self.assertEqual(actual.shape, (0,))
    self.assertEqual(actual.dtype, np.float64)

  def test_new_episode_does_not_change_previously_emitted_returns(self):
    recorder = return_subject.ReturnSubject()
    self.addCleanup(recorder.dispose)
    results = []
    self.addCleanup(recorder.subscribe(results.append).dispose)
    for reward in ([1, 2], [0.25, 0.5]):
      recorder.on_next(dm_env.restart(()))
      recorder.on_next(dm_env.termination(reward, ()))
    self.assertLen(results, 2)
    np.testing.assert_array_equal(results[0], [1, 2])
    np.testing.assert_array_equal(results[1], [0.25, 0.5])
    self.assertFalse(np.shares_memory(results[0], results[1]))

  def test_restart_discards_partial_return_and_retains_first_rewards(self):
    recorder = return_subject.ReturnSubject()
    self.addCleanup(recorder.dispose)
    results = []
    self.addCleanup(recorder.subscribe(results.append).dispose)
    recorder.on_next(dm_env.restart(()))
    recorder.on_next(dm_env.transition([100], ()))
    recorder.on_next(dm_env.TimeStep(dm_env.StepType.FIRST, [1], 1, ()))
    recorder.on_next(dm_env.termination([0.5], ()))
    self.assertLen(results, 1)
    np.testing.assert_array_equal(results[0], [1.5])

  def test_episode_order_and_missing_reward_controls(self):
    recorder = return_subject.ReturnSubject()
    self.addCleanup(recorder.dispose)
    with self.assertRaisesRegex(ValueError, 'First timestep'):
      recorder.on_next(dm_env.transition([1], ()))
    results = []
    self.addCleanup(recorder.subscribe(results.append).dispose)
    recorder.on_next(dm_env.restart(()))
    recorder.on_next(dm_env.termination(None, ()))
    self.assertEqual(results, [None])
    with self.assertRaisesRegex(ValueError, 'First timestep'):
      recorder.on_next(dm_env.termination([1], ()))


if __name__ == '__main__':
  absltest.main()
