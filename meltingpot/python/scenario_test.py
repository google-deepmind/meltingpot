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

from meltingpot.python import scenario as scenario_factory


class ScenarioTest(parameterized.TestCase):

  @parameterized.named_parameters(
      (scenario, scenario) for scenario in scenario_factory.AVAILABLE_SCENARIOS)
  def test_step_without_error(self, scenario):
    scenario_config = scenario_factory.get_config(scenario)
    num_players = scenario_config.num_players
    with scenario_factory.build(scenario_config) as scenario:
      scenario.reset()
      scenario.step([0] * num_players)


if __name__ == '__main__':
  absltest.main()
