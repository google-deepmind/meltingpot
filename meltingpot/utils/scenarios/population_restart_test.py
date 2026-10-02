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
"""Tests for forwarding reward-free restart timesteps to population policies."""

from absl.testing import absltest
from absl.testing import parameterized
import dm_env
from meltingpot.utils.evaluation import evaluation
from meltingpot.utils.evaluation import return_subject
from meltingpot.utils.policies import policy
from meltingpot.utils.scenarios import population
import numpy as np


class _RecordingPolicy(policy.Policy[int]):

  def __init__(self):
    self.calls = []

  def initial_state(self):
    return 0

  def step(self, timestep, prev_state):
    self.calls.append((timestep, prev_state))
    return timestep.observation['slot'] + 10 * prev_state, prev_state + 1

  def close(self):
    pass


class _TwoStepEnvironment(dm_env.Environment):

  def __init__(self):
    self.actions = []
    self.observations = [{'slot': 0}, {'slot': 1}]
    self.steps = 0

  def reset(self):
    self.steps = 0
    return dm_env.restart(self.observations)

  def step(self, action):
    self.actions.append(action)
    self.steps += 1
    if self.steps == 1:
      return dm_env.transition([1.0, 2.0], self.observations)
    return dm_env.termination([3.0, 4.0], self.observations)

  def observation_spec(self):
    return [{'slot': dm_env.specs.Array((), np.int64)}] * 2

  def action_spec(self):
    return [dm_env.specs.DiscreteArray(32)] * 2


class PopulationRestartTest(parameterized.TestCase):

  def make_population(self, num_players, shared):
    policies = [_RecordingPolicy() for _ in range(1 if shared else num_players)]
    bot_population = population.Population(
        policies={str(n): bot for n, bot in enumerate(policies)},
        names_by_role={
            str(n): [str(0 if shared else n)] for n in range(num_players)
        },
        roles=[str(n) for n in range(num_players)],
    )
    self.addCleanup(bot_population.close)
    return bot_population, policies

  @parameterized.product(num_players=(1, 2, 4), shared=(False, True))
  def test_restart_forwards_none_and_preserves_player_state(
      self, num_players, shared
  ):
    bot_population, policies = self.make_population(num_players, shared)
    emitted = []
    subscription = bot_population.observables().timestep.subscribe(
        emitted.append
    )
    self.addCleanup(subscription.dispose)
    observations = [{'slot': n} for n in range(num_players)]
    restart = dm_env.restart(observations)
    for _ in range(2):
      bot_population.reset()
      for bot in policies:
        bot.calls.clear()
      bot_population.send_timestep(restart)
      self.assertEqual(bot_population.await_action(), tuple(range(num_players)))
      for bot in policies:
        for timestep, state in bot.calls:
          self.assertIsNone(timestep.reward)
          self.assertIsNone(timestep.discount)
          self.assertTrue(timestep.first())
          self.assertEqual(state, 0)
          self.assertIs(
              timestep.observation, observations[timestep.observation['slot']]
          )
      self.assertIs(emitted[-1], restart)
      self.assertIsNone(restart.reward)
      rewards = [n + 0.5 for n in range(num_players)]
      timestep = dm_env.transition(rewards, observations, discount=0.75)
      bot_population.send_timestep(timestep)
      self.assertEqual(
          bot_population.await_action(),
          tuple(n + 10 for n in range(num_players)),
      )
      for bot in policies:
        calls = [call for call in bot.calls if call[0].mid()]
        for received, state in calls:
          self.assertEqual(
              received.reward, rewards[received.observation['slot']]
          )
          self.assertEqual(received.discount, 0.75)
          self.assertEqual(state, 1)

  @parameterized.parameters(list, tuple, np.asarray)
  def test_explicit_first_rewards_keep_their_values(self, container):
    bot_population, policies = self.make_population(2, False)
    rewards = container([0.0, 0.5])
    first = dm_env.restart([{'slot': 0}, {'slot': 1}])._replace(reward=rewards)
    bot_population.reset()
    bot_population.send_timestep(first)
    self.assertEqual(bot_population.await_action(), (0, 1))
    for n, bot in enumerate(policies):
      received, state = bot.calls[0]
      self.assertEqual(received.reward, rewards[n])
      self.assertEqual(state, 0)
    np.testing.assert_array_equal(rewards, [0.0, 0.5])

  def test_evaluation_loop_accepts_standard_restarts(self):
    bot_population, policies = self.make_population(2, False)
    environment = _TwoStepEnvironment()
    returns = []
    recorder = return_subject.ReturnSubject()
    self.addCleanup(recorder.dispose)
    subscription = bot_population.observables().timestep.subscribe(recorder)
    self.addCleanup(subscription.dispose)
    subscription = recorder.subscribe(returns.append)
    self.addCleanup(subscription.dispose)
    for _ in range(2):
      evaluation.run_episode(bot_population, environment)
    self.assertEqual(environment.actions, [(0, 1), (10, 11)] * 2)
    np.testing.assert_array_equal(returns, [[4.0, 6.0], [4.0, 6.0]])
    for bot in policies:
      self.assertEqual([state for _, state in bot.calls], [0, 1, 2] * 2)

  def test_action_protocol_errors_remain_unchanged(self):
    bot_population, _ = self.make_population(1, False)
    bot_population.reset()
    with self.assertRaisesRegex(RuntimeError, 'No timestep sent'):
      bot_population.await_action()
    timestep = dm_env.restart([{'slot': 0}])._replace(reward=[0.0])
    bot_population.send_timestep(timestep)
    with self.assertRaisesRegex(RuntimeError, 'Previous action not retrieved'):
      bot_population.send_timestep(timestep)
    self.assertEqual(bot_population.await_action(), (0,))


if __name__ == '__main__':
  absltest.main()
