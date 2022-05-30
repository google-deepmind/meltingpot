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
"""Tests of scenarios."""

from absl.testing import absltest
from absl.testing import parameterized

from meltingpot.python import scenario as scenario_factory


class ScenarioTest(parameterized.TestCase):

  @parameterized.named_parameters(
      (scenario, scenario) for scenario in scenario_factory.AVAILABLE_SCENARIOS)
  def test_observations_match_observation_spec(self, name):
    config = scenario_factory.get_config(name)
    with scenario_factory.build(config) as env:
      observation_spec = env.observation_spec()[0]
      timestep = env.reset()
      observation = timestep.observation[0]

    with self.subTest('missing'):
      self.assertSameElements(observation_spec, observation)
    for name, spec in observation_spec.items():
      with self.subTest(name):
        spec.validate(observation[name])

  @parameterized.named_parameters(
      (scenario, scenario) for scenario in scenario_factory.AVAILABLE_SCENARIOS)
  def test_spec_in_config_matches_environment(self, name):
    config = scenario_factory.get_config(name)
    with scenario_factory.build(config) as env:
      action_spec = env.action_spec()[0]
      discount_spec = env.discount_spec()
      reward_spec = env.reward_spec()[0]
      observation_spec = env.observation_spec()[0]

    with self.subTest('action_spec'):
      self.assertEqual(action_spec, config.action_spec)

    with self.subTest('reward_spec'):
      self.assertEqual(reward_spec, config.timestep_spec.reward)

    with self.subTest('discount_spec'):
      self.assertEqual(discount_spec, config.timestep_spec.discount)

    with self.subTest('missing'):
      self.assertSameElements(
          observation_spec, config.timestep_spec.observation)

    for name, spec in observation_spec.items():
      with self.subTest(f'observation_spec {name}'):
        self.assertEqual(spec, config.timestep_spec.observation[name])

  @parameterized.named_parameters(
      (scenario, scenario) for scenario in scenario_factory.AVAILABLE_SCENARIOS)
  def test_permitted_observations(self, scenario):
    scenario_config = scenario_factory.get_config(scenario)
    self.assertContainsSubset(
        scenario_config.timestep_spec.observation,
        scenario_factory.PERMITTED_OBSERVATIONS)


if __name__ == '__main__':
  absltest.main()
