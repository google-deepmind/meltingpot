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
"""Tests for base wrapper."""

import inspect
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import dmlab2d
from meltingpot.utils.substrates.wrappers import base

_WRAPPED_METHODS = tuple([
    name for name, _ in inspect.getmembers(dmlab2d.Environment)
    if not name.startswith('_')
])


class WrapperTest(parameterized.TestCase):

  def test_instance(self):
    env = mock.Mock(spec_set=dmlab2d.Environment)
    wrapped = base.Lab2dWrapper(env=env)
    self.assertIsInstance(wrapped, dmlab2d.Environment)

  @parameterized.named_parameters(
      (name, name) for name in _WRAPPED_METHODS
  )
  def test_wrapped(self, method):
    env = mock.Mock(spec_set=dmlab2d.Environment)
    env_method = getattr(env, method)
    env_method.return_value = mock.sentinel

    wrapped = base.Lab2dWrapper(env=env)
    args = [object()]
    kwargs = {'a': object()}
    actual = getattr(wrapped, method)(*args, **kwargs)

    with self.subTest('args'):
      env_method.assert_called_once_with(*args, **kwargs)
    with self.subTest('return_value'):
      self.assertEqual(actual, env_method.return_value)


if __name__ == '__main__':
  absltest.main()
