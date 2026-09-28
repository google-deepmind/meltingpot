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
"""Scenario player partitions must not alias the caller's mutable mask."""

from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import dm_env
from meltingpot.utils.policies import fixed_action_policy
from meltingpot.utils.scenarios import population
from meltingpot.utils.scenarios import scenario as scenario_lib
from meltingpot.utils.substrates import substrate as substrate_lib
import numpy as np


class ScenarioMaskSnapshotTest(parameterized.TestCase):

  def make_scenario(self, mask):
    raw = mock.Mock(spec_set=substrate_lib.Substrate)
    raw.action_spec.return_value = tuple(
        dm_env.specs.DiscreteArray(index + 2) for index in range(4)
    )
    raw.observation_spec.return_value = tuple(
        {'position': dm_env.specs.Array((), np.int64, name=f'player{index}')}
        for index in range(4)
    )
    raw.reward_spec.return_value = tuple(
        dm_env.specs.Array((), np.float64, name=f'reward{index}')
        for index in range(4)
    )
    observations = tuple(
        {'position': index, 'hidden': index + 100} for index in range(4)
    )
    raw.observation.return_value = observations
    raw.events.return_value = ()
    raw.reset.return_value = dm_env.TimeStep(
        dm_env.StepType.FIRST, (10, 20, 30, 40), 1, observations
    )
    raw.step.return_value = dm_env.termination((11, 21, 31, 41), observations)
    background = population.Population(
        policies={'bot': fixed_action_policy.FixedActionPolicy(0)},
        names_by_role={'role': ('bot',)},
        roles=('role', 'role'),
    )
    scenario = scenario_lib.Scenario(
        substrate_lib.Substrate(raw), background, mask, ('position',)
    )
    self.addCleanup(scenario.close)
    return scenario, raw, background

  @parameterized.parameters('swap', 'clear', 'append', 'replace')
  def test_mutating_original_list_keeps_specs_and_observations(self, operation):
    mask = [True, False, True, False]
    scenario, raw, _ = self.make_scenario(mask)
    expected = (
        scenario.action_spec(),
        scenario.observation_spec(),
        scenario.reward_spec(),
        scenario.observation(),
    )
    if operation == 'swap':
      mask[:] = [False, True, False, True]
    elif operation == 'clear':
      mask.clear()
    elif operation == 'append':
      mask.append(True)
    else:
      mask[:] = [False] * 4
    actual = (
        scenario.action_spec(),
        scenario.observation_spec(),
        scenario.reward_spec(),
        scenario.observation(),
    )
    self.assertEqual(actual, expected)
    self.assertEqual(
        scenario.action_spec(), (raw.action_spec()[0], raw.action_spec()[2])
    )
    self.assertEqual(scenario.observation(), ({'position': 0}, {'position': 2}))

    self.assertEqual(scenario.reset().reward, (10, 30))
    self.assertEqual(scenario.step([1, 3]).reward, (11, 31))
    raw.step.assert_called_once_with((1, 0, 3, 0))

  @parameterized.parameters(False, True)
  def test_action_and_reward_routing_survives_list_mutation(self, after_reset):
    mask = [True, False, True, False]
    scenario, raw, background = self.make_scenario(mask)
    focal_seen, background_seen = [], []
    self.addCleanup(
        scenario.observables().timestep.subscribe(focal_seen.append).dispose
    )
    self.addCleanup(
        background.observables()
        .timestep.subscribe(background_seen.append)
        .dispose
    )
    first = scenario.reset() if after_reset else None
    mask[:] = [False, True, False, True]
    if first is None:
      first = scenario.reset()
    self.assertEqual(first.reward, (10, 30))
    last = scenario.step([1, 3])
    raw.step.assert_called_once_with((1, 0, 3, 0))
    self.assertTrue(last.last())
    self.assertEqual(last.reward, (11, 31))
    self.assertEqual([step.reward for step in focal_seen], [(10, 30), (11, 31)])
    self.assertEqual(
        [step.reward for step in background_seen], [(20, 40), (21, 41)]
    )
    self.assertEqual(background.await_action(), (0, 0))

  def test_numpy_mask_is_snapshotted(self):
    mask = np.array([True, False, True, False])
    scenario, raw, _ = self.make_scenario(mask)
    mask[:] = ~mask
    self.assertEqual(
        scenario.action_spec(), (raw.action_spec()[0], raw.action_spec()[2])
    )
    self.assertEqual(scenario.reset().reward, (10, 30))

  @parameterized.parameters(list, tuple)
  def test_unmodified_mask_and_action_count_validation_are_unchanged(
      self, factory
  ):
    mask = factory((True, False, True, False))
    scenario, raw, _ = self.make_scenario(mask)
    self.assertEqual(list(mask), [True, False, True, False])
    scenario.reset()
    with self.assertRaisesRegex(ValueError, 'Expected 2 focal actions'):
      scenario.step([1])
    raw.step.assert_not_called()
    last = scenario.step([1, 0])
    self.assertEqual(last.reward, (11, 31))
    self.assertEqual(list(mask), [True, False, True, False])


if __name__ == '__main__':
  absltest.main()
