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
"""Tests for default_observation_wrapper."""

from unittest import mock

from absl.testing import absltest
import dm_env
import numpy as np

from meltingpot.python.utils.scenarios.wrappers import default_observation_wrapper
from meltingpot.python.utils.substrates import substrate

TEST_KEY = 'test_key'
RGB_SPEC = dm_env.specs.Array(shape=(8, 8, 3), dtype=np.int8)
RGB_VALUE = np.ones((8, 8, 3), np.int8)
DEFAULT_SPEC = dm_env.specs.Array(shape=[0], dtype=np.double, name=TEST_KEY)
DEFAULT_VALUE = np.zeros(shape=[0], dtype=np.double)
TEST_SPEC = dm_env.specs.Array(shape=[2], dtype=np.double, name=TEST_KEY)
TEST_VALUE = np.zeros(shape=[2], dtype=np.double)


class WrapperTest(absltest.TestCase):

  def test_setdefault_no_change(self):
    original = {'a': 0}
    actual = default_observation_wrapper._setdefault(original, 'a', 100)
    self.assertIs(actual, original)

  def test_setdefault_change(self):
    original = {'a': 0}
    actual = default_observation_wrapper._setdefault(original, 'b', 100)
    with self.subTest('creates_copy'):
      self.assertIsNot(actual, original)
    with self.subTest('adds_value'):
      self.assertEqual(actual, {'a': 0, 'b': 100})

  def test_change(self):
    env = mock.Mock(spec_set=substrate.Substrate)
    env.events.return_value = ()
    env.observation_spec.return_value = [{'RGB': RGB_SPEC}]
    env.reset.return_value = dm_env.restart([{'RGB': RGB_VALUE}])
    env.step.return_value = dm_env.transition(1, [{'RGB': RGB_VALUE}])

    wrapped = default_observation_wrapper.Wrapper(
        env=env, key=TEST_KEY, default_spec=TEST_SPEC, default_value=TEST_VALUE)

    with self.subTest('observation_spec'):
      self.assertEqual(wrapped.observation_spec(), [{
          'RGB': RGB_SPEC,
          TEST_KEY: TEST_SPEC
      }])
    with self.subTest('reset'):
      np.testing.assert_equal(
          wrapped.reset(),
          dm_env.restart([{
              'RGB': RGB_VALUE,
              TEST_KEY: TEST_VALUE
          }]))
    with self.subTest('observation'):
      np.testing.assert_equal(
          wrapped.step(mock.sentinel.action),
          dm_env.transition(1, [{
              'RGB': RGB_VALUE,
              TEST_KEY: TEST_VALUE
          }]))

  def test_nochange(self):
    env = mock.Mock(spec_set=substrate.Substrate)
    env.events.return_value = ()
    env.observation_spec.return_value = [{
        'RGB': RGB_SPEC,
        TEST_KEY: DEFAULT_SPEC
    }]
    env.reset.return_value = dm_env.restart([{
        'RGB': RGB_VALUE,
        TEST_KEY: DEFAULT_VALUE
    }])
    env.step.return_value = dm_env.transition(1, [{
        'RGB': RGB_VALUE,
        TEST_KEY: DEFAULT_VALUE
    }])

    wrapped = default_observation_wrapper.Wrapper(
        env=env, key=TEST_KEY, default_spec=TEST_SPEC, default_value=TEST_VALUE)

    with self.subTest('observation_spec'):
      self.assertEqual(wrapped.observation_spec(), [{
          'RGB': RGB_SPEC,
          TEST_KEY: DEFAULT_SPEC
      }])
    with self.subTest('reset'):
      np.testing.assert_equal(
          wrapped.reset(),
          dm_env.restart([{
              'RGB': RGB_VALUE,
              TEST_KEY: DEFAULT_VALUE
          }]))
    with self.subTest('observation'):
      np.testing.assert_equal(
          wrapped.step(mock.sentinel.action),
          dm_env.transition(1, [{
              'RGB': RGB_VALUE,
              TEST_KEY: DEFAULT_VALUE
          }]))


if __name__ == '__main__':
  absltest.main()
