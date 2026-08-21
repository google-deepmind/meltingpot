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
"""Regression tests for ObservablesWrapper reset behavior."""

from unittest import mock

from absl.testing import absltest
import dmlab2d
from meltingpot.utils.substrates.wrappers import observables_wrapper


class ObservablesWrapperResetTest(absltest.TestCase):

  def test_forwards_reset_arguments_and_emits_timestep(self):
    env = mock.Mock(spec_set=dmlab2d.Environment)
    timestep = mock.sentinel.timestep
    env.reset.return_value = timestep
    env.events.return_value = ()
    wrapped = observables_wrapper.ObservablesWrapper(env)
    observed = []
    subscription = wrapped.observables().timestep.subscribe(observed.append)
    self.addCleanup(subscription.dispose)

    reset_arg = mock.sentinel.reset_arg
    reset_option = mock.sentinel.reset_option
    actual = wrapped.reset(reset_arg, option=reset_option)

    env.reset.assert_called_once_with(reset_arg, option=reset_option)
    self.assertIs(actual, timestep)
    self.assertEqual(observed, [timestep])


if __name__ == '__main__':
  absltest.main()
