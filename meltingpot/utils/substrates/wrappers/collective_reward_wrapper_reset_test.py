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
"""Regression test for collective reward reset argument forwarding."""

from unittest import mock

from absl.testing import absltest
import dm_env
import dmlab2d
from meltingpot.utils.substrates.wrappers import collective_reward_wrapper


class CollectiveRewardResetTest(absltest.TestCase):

  def test_reset_forwards_args_and_kwargs(self):
    env = mock.Mock(spec_set=dmlab2d.Environment)
    env.reset.return_value = dm_env.restart(
        observation=[{'RGB': mock.sentinel.rgb}],
    )
    wrapped = collective_reward_wrapper.CollectiveRewardWrapper(env)

    wrapped.reset(mock.sentinel.arg, option=mock.sentinel.option)

    env.reset.assert_called_once_with(
        mock.sentinel.arg, option=mock.sentinel.option)


if __name__ == '__main__':
  absltest.main()
