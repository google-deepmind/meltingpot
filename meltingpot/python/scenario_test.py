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

import functools
import random
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import dm_env

from meltingpot.python import bot as bot_factory
from meltingpot.python import scenario as scenario_factory
from meltingpot.python import substrate as substrate_factory


class ScenarioTest(parameterized.TestCase):

  @parameterized.named_parameters(
      (scenario, scenario) for scenario in scenario_factory.AVAILABLE_SCENARIOS)
  def test_step_without_error(self, scenario):
    scenario_config = scenario_factory.get_config(scenario)
    num_players = scenario_config.num_players
    with scenario_factory.build(scenario_config) as scenario:
      scenario.reset()
      scenario.step([0] * num_players)


class ScenarioWrapperTest(parameterized.TestCase):

  def test_wrapper(self):
    received_actions = [None] * 4
    received_bot_rewards = [None] * 2
    received_bot_observations = [None] * 2
    received_bot_states = [None] * 2

    def substrate_reset():
      return dm_env.TimeStep(
          step_type=dm_env.StepType.FIRST,
          discount=0,
          reward=[10, 20, 30, 40],
          observation=[
              dict(ok=10, not_ok=100),
              dict(ok=20, not_ok=200),
              dict(ok=30, not_ok=300),
              dict(ok=40, not_ok=400),
          ],
      )

    def substrate_step(actions):
      received_actions[:] = actions
      return dm_env.transition(
          reward=[11, 21, 31, 41],
          observation=[
              dict(ok=11, not_ok=101),
              dict(ok=21, not_ok=201),
              dict(ok=31, not_ok=301),
              dict(ok=41, not_ok=401),
          ],
      )

    def bot_step(timestep, prev_state, n):
      received_bot_rewards[n] = timestep.reward
      received_bot_observations[n] = timestep.observation
      received_bot_states[n] = prev_state
      return n + 2, prev_state

    substrate = mock.Mock(spec_set=substrate_factory.Substrate)
    substrate.action_spec.return_value = [object() for _ in range(4)]
    substrate.observation_spec.return_value = [
        dict(ok='ok_spec', not_ok='not_ok_spec') for _ in range(4)
    ]
    substrate.reward_spec.return_value = [object() for _ in range(4)]
    substrate.reset.side_effect = substrate_reset
    substrate.step.side_effect = substrate_step

    bots = {}
    for n in range(2):
      bot = mock.Mock(spec_set=bot_factory.Policy)
      bot.initial_state.return_value = f'state_{n}'
      bot.step.side_effect = functools.partial(bot_step, n=n)
      bots[f'bot_{n}'] = bot

    with scenario_factory.Scenario(
        substrate, bots, num_bots=2, permitted_observations={'ok'}) as scenario:
      action_spec = scenario.action_spec()
      observation_spec = scenario.observation_spec()
      reward_spec = scenario.reward_spec()
      with mock.patch.object(
          random, 'choices', return_value=['bot_0', 'bot_1']):
        timestep = scenario.reset()
      scenario.step([0, 1])

    with self.subTest(name='action_spec'):
      self.assertEqual(action_spec, substrate.action_spec()[:2])
    with self.subTest(name='observation_spec'):
      self.assertEqual(observation_spec, [dict(ok='ok_spec') for _ in range(2)])
    with self.subTest(name='reward_spec'):
      self.assertEqual(reward_spec, substrate.reward_spec()[:2])

    with self.subTest(name='received_bot_states'):
      self.assertEqual(received_bot_states, ['state_0', 'state_1'])
    with self.subTest(name='received_bot_rewards'):
      self.assertEqual(received_bot_rewards, [30, 40])
    with self.subTest(name='received_bot_observations'):
      self.assertEqual(received_bot_observations, [
          dict(ok=30, not_ok=300),
          dict(ok=40, not_ok=400),
      ])

    with self.subTest(name='received_actions'):
      self.assertEqual(received_actions, [0, 1, 2, 3])

    with self.subTest(name='agent_rewards'):
      self.assertEqual(timestep.reward, [10, 20])
    with self.subTest(name='agent_observations'):
      self.assertEqual(timestep.observation, [dict(ok=10), dict(ok=20)])


if __name__ == '__main__':
  absltest.main()
