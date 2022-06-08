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

from meltingpot.python import bot as bot_factory
from meltingpot.python import scenario as scenario_factory
from meltingpot.python.testing import bots as test_utils


def _get_specs(substrate):
  scenario = next(iter(scenario_factory.SCENARIOS_BY_SUBSTRATE[substrate]))
  config = scenario_factory.get_config(scenario)
  return config.background_timestep_spec, config.background_action_spec


@parameterized.named_parameters(
    (name, name) for name in bot_factory.AVAILABLE_BOTS)
class BotTest(test_utils.BotTestCase):

  def test_step_without_error(self, name):
    bot_config = bot_factory.get_config(name)
    timestep_spec, action_spec = _get_specs(bot_config.substrate)
    with bot_factory.build(bot_config) as policy:
      self.assert_compatible(policy, timestep_spec, action_spec)


if __name__ == '__main__':
  absltest.main()
