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

import random
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import dm_env
import immutabledict

from meltingpot.python import bot as bot_factory
from meltingpot.python import scenario as scenario_factory
from meltingpot.python import substrate as substrate_factory


class ScenarioTest(parameterized.TestCase):

  @parameterized.named_parameters(
      (scenario,) * 2 for scenario in scenario_factory.AVAILABLE_SCENARIOS)
  def test_step_without_error(self, scenario):
    scenario_config = scenario_factory.get_config(scenario)
    num_players = scenario_config.num_players
    with scenario_factory.build(scenario_config) as scenario:
      scenario.reset()
      scenario.step([0] * num_players)

  @parameterized.named_parameters(
      (scenario,) * 2 for scenario in scenario_factory.AVAILABLE_SCENARIOS)
  def test_permitted_observations(self, scenario):
    scenario_config = scenario_factory.get_config(scenario)
    with scenario_factory.build(scenario_config) as scenario:
      scenario_spec = set(scenario.observation_spec()[0])
    with substrate_factory.build(scenario_config.substrate) as substrate:
      substrate_spec = set(substrate.observation_spec()[0])

    self.assertEqual(scenario_spec,
                     substrate_spec & scenario_factory.PERMITTED_OBSERVATIONS)


@parameterized.parameters(
    ((), (), (), ()),
    (('a',), (True,), ('a',), ()),
    (('a',), (False,), (), ('a',)),
    (('a', 'b', 'c'), (True, True, False), ('a', 'b'), ('c',)),
    (('a', 'b', 'c'), (False, True, False), ('b',), ('a', 'c')),
)
class PartitionMergeTest(parameterized.TestCase):

  def test_partition(self, merged, is_focal, *expected):
    actual = scenario_factory._partition(merged, is_focal)
    self.assertEqual(actual, expected)

  def test_merge(self, expected, is_focal, *partions):
    actual = scenario_factory._merge(*partions, is_focal)
    self.assertEqual(actual, expected)


class ScenarioWrapperTest(absltest.TestCase):

  def test_scenario(self):
    substrate = mock.Mock(spec_set=substrate_factory.Substrate)
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
    substrate.events.return_value = ()
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

    bots = {}
    for n in range(2):
      bot = mock.Mock(spec_set=bot_factory.Policy)
      bot.initial_state.return_value = f'bot_state_{n}'
      bot.step.return_value = (n + 10, f'bot_state_{n}')
      bots[f'bot_{n}'] = bot

    with scenario_factory.Scenario(
        substrate, bots,
        is_focal=[True, False, True, False],
        permitted_observations={'ok'}) as scenario:
      action_spec = scenario.action_spec()
      observation_spec = scenario.observation_spec()
      reward_spec = scenario.reward_spec()
      with mock.patch.object(
          random, 'choices', return_value=['bot_0', 'bot_1']):
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


if __name__ == '__main__':
  absltest.main()
