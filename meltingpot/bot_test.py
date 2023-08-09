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
"""Tests of bots."""

from absl.testing import absltest
from absl.testing import parameterized
from meltingpot import bot
from meltingpot.testing import bots as test_utils


@parameterized.named_parameters((name, name) for name in bot.BOTS)
class BotTest(test_utils.BotTestCase):

  def test_step_without_error(self, name):
    factory = bot.get_factory(name)
    with factory.build() as policy:
      self.assert_compatible(
          policy,
          timestep_spec=factory.timestep_spec(),
          action_spec=factory.action_spec())


if __name__ == '__main__':
  absltest.main()
