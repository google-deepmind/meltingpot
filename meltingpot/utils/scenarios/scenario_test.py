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

from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import dm_env
import immutabledict
from meltingpot.utils.policies import fixed_action_policy
from meltingpot.utils.policies import policy
from meltingpot.utils.scenarios import population
from meltingpot.utils.scenarios import scenario as scenario_utils
from meltingpot.utils.substrates import substrate as substrate_lib
import numpy as np


def _track(source, fields):
  destination = []
  for field in fields:
    getattr(source, field).subscribe(
        on_next=destination.append,
        on_error=lambda e: destination.append(type(e)),
        on_completed=lambda: destination.append('DONE'),
    )
  return destination


@parameterized.parameters(
    ((), (), (), ()),
    (('a',), (True,), ('a',), ()),
    (('a',), (False,), (), ('a',)),
    (('a', 'b', 'c'), (True, True, False), ('a', 'b'), ('c',)),
    (('a', 'b', 'c'), (False, True, False), ('b',), ('a', 'c')),
)
class PartitionMergeTest(parameterized.TestCase):

  def test_partition(self, merged, is_focal, *expected):
    actual = scenario_utils._partition(merged, is_focal)
    self.assertEqual(actual, expected)

  def test_merge(self, expected, is_focal, *partions):
    actual = scenario_utils._merge(*partions, is_focal)  # pyrefly: ignore[bad-argument-count]
    self.assertEqual(actual, expected)


class ScenarioWrapperTest(absltest.TestCase):

  def test_scenario(self):
    substrate = mock.Mock(spec_set=substrate_lib.Substrate)
    substrate.reset.return_value = dm_env.TimeStep(
        step_type=dm_env.StepType.FIRST,
        discount=0,
        reward=(10, 20, 30, 40),
        observation=(
            immutabledict.immutabledict(ok=10, not_ok=100),
            immutabledict.immutabledict(ok=20, not_ok=200),
            immutabledict.immutabledict(ok=30, not_ok=300),
            immutabledict.immutabledict(ok=40, not_ok=400),
        ),
    )
    substrate.step.return_value = dm_env.transition(
        reward=(11, 21, 31, 41),
        observation=(
            immutabledict.immutabledict(ok=11, not_ok=101),
            immutabledict.immutabledict(ok=21, not_ok=201),
            immutabledict.immutabledict(ok=31, not_ok=301),
            immutabledict.immutabledict(ok=41, not_ok=401),
        ),
    )
    substrate.events.return_value = (
        mock.sentinel.event_0, mock.sentinel.event_1)
    substrate.action_spec.return_value = tuple(
        f'action_spec_{n}' for n in range(4)
    )
    substrate.observation_spec.return_value = tuple(
        immutabledict.immutabledict(
            ok=f'ok_spec_{n}', not_ok=f'not_ok_spec_{n}')
        for n in range(4)
    )
    substrate.reward_spec.return_value = tuple(
        f'reward_spec_{n}' for n in range(4)
    )
    substrate.observation.return_value = (
        immutabledict.immutabledict(ok=10, not_ok=100),
        immutabledict.immutabledict(ok=20, not_ok=200),
        immutabledict.immutabledict(ok=30, not_ok=300),
        immutabledict.immutabledict(ok=40, not_ok=400),
    )

    bots = {}
    for n in range(2):
      bot = mock.Mock(spec_set=policy.Policy)
      bot.initial_state.return_value = f'bot_state_{n}'
      bot.step.return_value = (n + 10, f'bot_state_{n}')
      bots[f'bot_{n}'] = bot
    background_population = population.Population(
        policies=bots,
        names_by_role={'role_0': {'bot_0'}, 'role_1': {'bot_1'}},
        roles=['role_0', 'role_1'],
    )

    with scenario_utils.Scenario(
        substrate=substrate_lib.Substrate(substrate),
        background_population=background_population,
        is_focal=[True, False, True, False],
        permitted_observations={'ok'}) as scenario:
      observables = scenario.observables()
      received = {
          'base': _track(observables, ['events', 'action', 'timestep']),
          'background': _track(observables.background, ['action', 'timestep']),
          'substrate': _track(
              observables.substrate, ['events', 'action', 'timestep']),
      }
      action_spec = scenario.action_spec()
      observation_spec = scenario.observation_spec()
      reward_spec = scenario.reward_spec()
      observation = scenario.observation()
      initial_timestep = scenario.reset()
      step_timestep = scenario.step([0, 1])

    with self.subTest(name='action_spec'):
      self.assertEqual(action_spec, ('action_spec_0', 'action_spec_2'))
    with self.subTest(name='observation_spec'):
      self.assertEqual(observation_spec,
                       (immutabledict.immutabledict(ok='ok_spec_0'),
                        immutabledict.immutabledict(ok='ok_spec_2')))
    with self.subTest(name='reward_spec'):
      self.assertEqual(reward_spec, ('reward_spec_0', 'reward_spec_2'))

    with self.subTest(name='observation'):
      expected = (
          immutabledict.immutabledict(ok=10),
          immutabledict.immutabledict(ok=30),
      )
      self.assertEqual(observation, expected)

    with self.subTest(name='events'):
      self.assertEmpty(scenario.events())

    with self.subTest(name='initial_timestep'):
      expected = dm_env.TimeStep(
          step_type=dm_env.StepType.FIRST,
          discount=0,
          reward=(10, 30),
          observation=(
              immutabledict.immutabledict(ok=10),
              immutabledict.immutabledict(ok=30),
          ),
      )
      self.assertEqual(initial_timestep, expected)

    with self.subTest(name='step_timestep'):
      expected = dm_env.transition(
          reward=(11, 31),
          observation=(
              immutabledict.immutabledict(ok=11),
              immutabledict.immutabledict(ok=31),
          ),
      )
      self.assertEqual(step_timestep, expected)

    with self.subTest(name='substrate_step'):
      substrate.step.assert_called_once_with((0, 10, 1, 11))

    with self.subTest(name='bot_0_step'):
      actual = bots['bot_0'].step.call_args_list[0]
      expected = mock.call(
          timestep=dm_env.TimeStep(
              step_type=dm_env.StepType.FIRST,
              discount=0,
              reward=20,
              observation=immutabledict.immutabledict(ok=20, not_ok=200),
          ),
          prev_state='bot_state_0')
      self.assertEqual(actual, expected)

    with self.subTest(name='bot_1_step'):
      actual = bots['bot_1'].step.call_args_list[0]
      expected = mock.call(
          timestep=dm_env.TimeStep(
              step_type=dm_env.StepType.FIRST,
              discount=0,
              reward=40,
              observation=immutabledict.immutabledict(ok=40, not_ok=400),
          ),
          prev_state='bot_state_1')
      self.assertEqual(actual, expected)

    with self.subTest(name='base_observables'):
      expected = [
          dm_env.TimeStep(
              step_type=dm_env.StepType.FIRST,
              discount=0,
              reward=(10, 30),
              observation=(
                  immutabledict.immutabledict(ok=10),
                  immutabledict.immutabledict(ok=30),
              ),
          ),
          [0, 1],
          dm_env.transition(
              reward=(11, 31),
              observation=(
                  immutabledict.immutabledict(ok=11),
                  immutabledict.immutabledict(ok=31),
              ),
          ),
          'DONE',
          'DONE',
          'DONE',
      ]
      self.assertEqual(received['base'], expected)

    with self.subTest(name='substrate_observables'):
      expected = [
          dm_env.TimeStep(
              step_type=dm_env.StepType.FIRST,
              discount=0,
              reward=(10, 20, 30, 40),
              observation=(
                  immutabledict.immutabledict(ok=10, not_ok=100),
                  immutabledict.immutabledict(ok=20, not_ok=200),
                  immutabledict.immutabledict(ok=30, not_ok=300),
                  immutabledict.immutabledict(ok=40, not_ok=400),
              ),
          ),
          mock.sentinel.event_0,
          mock.sentinel.event_1,
          (0, 10, 1, 11),
          dm_env.transition(
              reward=(11, 21, 31, 41),
              observation=(
                  immutabledict.immutabledict(ok=11, not_ok=101),
                  immutabledict.immutabledict(ok=21, not_ok=201),
                  immutabledict.immutabledict(ok=31, not_ok=301),
                  immutabledict.immutabledict(ok=41, not_ok=401),
              ),
          ),
          mock.sentinel.event_0,
          mock.sentinel.event_1,
          'DONE',
          'DONE',
          'DONE',
      ]
      self.assertEqual(received['substrate'], expected)

    with self.subTest(name='background_observables'):
      expected = [
          dm_env.TimeStep(
              step_type=dm_env.StepType.FIRST,
              discount=0,
              reward=(20, 40),
              observation=(
                  immutabledict.immutabledict(ok=20, not_ok=200),
                  immutabledict.immutabledict(ok=40, not_ok=400),
              ),
          ),
          (10, 11),
          dm_env.transition(
              reward=(21, 41),
              observation=(
                  immutabledict.immutabledict(ok=21, not_ok=201),
                  immutabledict.immutabledict(ok=41, not_ok=401),
              ),
          ),
          'DONE',
          'DONE',
      ]
      self.assertEqual(received['background'], expected)


class ScenarioRewardFreeRestartTest(absltest.TestCase):

  def test_split_timestep_preserves_absent_rewards(self):
    scenario = object.__new__(scenario_utils.Scenario)
    scenario._is_focal = (True, False, True)
    scenario._permitted_observations = frozenset({'ok'})
    timestep = dm_env.restart((
        {'ok': 1, 'hidden': 10},
        {'ok': 2, 'hidden': 20},
        {'ok': 3, 'hidden': 30},
    ))

    focal, background = scenario._split_timestep(timestep)

    self.assertIsNone(focal.reward)
    self.assertIsNone(background.reward)
    self.assertEqual(focal.observation, ({'ok': 1}, {'ok': 3}))
    self.assertEqual(
        background.observation, ({'ok': 2, 'hidden': 20},))


class ScenarioActionValidationTest(parameterized.TestCase):

  @parameterized.parameters(([0],), ([1],), ([0, 1, 2],))
  def test_rejects_wrong_number_of_focal_actions(self, focal_action):
    scenario = object.__new__(scenario_utils.Scenario)
    scenario._is_focal = (True, False, True)
    scenario._focal_action_subject = mock.Mock()
    scenario._background_population = mock.Mock()

    with self.assertRaisesRegex(ValueError, 'Expected 2 focal actions'):
      scenario._await_full_action(focal_action)

    scenario._focal_action_subject.on_next.assert_not_called()
    scenario._background_population.await_action.assert_not_called()


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
    scenario = scenario_utils.Scenario(
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
