# Copyright 2026 DeepMind Technologies Limited.
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
"""Tests for ResetWrapper."""

from unittest import mock

from absl.testing import absltest
import dmlab2d
from meltingpot.utils.substrates.wrappers import reset_wrapper


class ResetWrapperTest(absltest.TestCase):

  def test_forwards_reset_arguments_before_and_after_rebuild(self):
    first_env = mock.Mock(spec_set=dmlab2d.Environment)
    second_env = mock.Mock(spec_set=dmlab2d.Environment)
    build_environment = mock.Mock(side_effect=(first_env, second_env))
    wrapped = reset_wrapper.ResetWrapper(build_environment)

    first_arg = mock.sentinel.first_arg
    first_kwarg = mock.sentinel.first_kwarg
    wrapped.reset(first_arg, option=first_kwarg)

    first_env.reset.assert_called_once_with(first_arg, option=first_kwarg)
    first_env.close.assert_not_called()

    second_arg = mock.sentinel.second_arg
    second_kwarg = mock.sentinel.second_kwarg
    wrapped.reset(second_arg, option=second_kwarg)

    first_env.close.assert_called_once_with()
    second_env.reset.assert_called_once_with(second_arg, option=second_kwarg)
    self.assertEqual(build_environment.call_count, 2)


if __name__ == '__main__':
  absltest.main()
