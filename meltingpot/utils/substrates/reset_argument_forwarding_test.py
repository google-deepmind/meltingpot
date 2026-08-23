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
"""Tests for reset argument forwarding in the substrate stack."""

from unittest import mock

from absl.testing import absltest
import dm_env
from meltingpot.utils.substrates import substrate
from meltingpot.utils.substrates.wrappers import multiplayer_wrapper


class ResetArgumentForwardingTest(absltest.TestCase):

  def test_multiplayer_wrapper_forwards_reset_arguments(self):
    env = mock.Mock()
    env.action_spec.return_value = {
        '1.move': dm_env.specs.DiscreteArray(num_values=2, name='move'),
    }
    env.reset.return_value = dm_env.restart(observation={})
    wrapper = multiplayer_wrapper.Wrapper(
        env,
        individual_observation_names=(),
        global_observation_names=(),
    )

    wrapper.reset(mock.sentinel.argument, option=mock.sentinel.option)

    env.reset.assert_called_once_with(
        mock.sentinel.argument, option=mock.sentinel.option
    )

  def test_substrate_forwards_reset_arguments(self):
    env = mock.Mock()
    env.observables.return_value = mock.sentinel.observables
    env.reset.return_value = dm_env.restart(observation=())
    env.events.return_value = ()
    wrapped = substrate.Substrate(env)

    wrapped.reset(mock.sentinel.argument, option=mock.sentinel.option)

    env.reset.assert_called_once_with(
        mock.sentinel.argument, option=mock.sentinel.option
    )


if __name__ == '__main__':
  absltest.main()
