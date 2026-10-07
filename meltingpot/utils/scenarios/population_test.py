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
"""Tests for scenario populations."""

import threading
from unittest import mock

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


class PopulationTest(absltest.TestCase):

  def test_sampling_candidates_have_deterministic_order(self):
    policies = {
        'bot_a': mock.Mock(spec_set=policy.Policy),
        'bot_b': mock.Mock(spec_set=policy.Policy),
    }
    bot_population = population.Population(
        policies=policies,
        names_by_role={'role': {'bot_b', 'bot_a'}},
        roles=['role'],
    )
    self.addCleanup(bot_population.close)

    with mock.patch.object(
        population.random, 'choice', return_value='bot_a') as choice:
      sampled = bot_population._sample_names()

    self.assertEqual(sampled, ['bot_a'])
    choice.assert_called_once_with(('bot_a', 'bot_b'))


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
      evaluation.run_episode(bot_population, environment)  # pyrefly: ignore[bad-argument-type]
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
class _CountingPolicy(policy.Policy[int]):
  """A deliberately unhashable policy with observable per-slot state."""

  __hash__ = None

  def __eq__(self, other):
    return isinstance(other, _CountingPolicy)

  def initial_state(self):
    return 0

  def step(self, timestep, prev_state):
    return timestep.observation['offset'] + prev_state, prev_state + 1

  def close(self):
    pass


def _timestep(step_type=dm_env.StepType.FIRST):
  return dm_env.TimeStep(
      step_type=step_type,
      reward=(0.0, 0.0),
      discount=1.0,
      observation=({'offset': 10}, {'offset': 20}),
  )


class AliasedPolicyTest(parameterized.TestCase):

  def make_population(self, first, second, same_name=False):
    if same_name:
      policies = {'shared': first}
      candidates = {'left': ['shared'], 'right': ['shared']}
    else:
      policies = {'first': first, 'second': second}
      candidates = {'left': ['first'], 'right': ['second']}
    result = population.Population(
        policies=policies, names_by_role=candidates, roles=['left', 'right']
    )
    self.addCleanup(result.close)
    return result

  def test_aliases_share_one_lock_by_identity(self):
    shared = _CountingPolicy()
    agents = self.make_population(shared, shared)
    self.assertIs(agents._locks['first'], agents._locks['second'])

  def test_equal_unhashable_policies_keep_distinct_locks(self):
    first, second = _CountingPolicy(), _CountingPolicy()
    self.assertEqual(first, second)
    agents = self.make_population(first, second)
    self.assertIsNot(agents._locks['first'], agents._locks['second'])

  @parameterized.parameters(False, True)
  def test_worker_steps_never_overlap_for_one_policy(self, same_name):
    shared = _CountingPolicy()
    agents = self.make_population(shared, shared, same_name)
    agents.reset()
    first_entered = threading.Event()
    release = threading.Event()
    overlap = threading.Event()
    guard = threading.Lock()
    entered = 0
    active = 0
    original_step = shared.step

    def step(timestep, prev_state):
      nonlocal entered, active
      with guard:
        entered += 1
        active += 1
        if active > 1:
          overlap.set()
        first_entered.set()
      try:
        if not release.wait(5):
          raise RuntimeError('test did not release the policy step')
        return original_step(timestep, prev_state)
      finally:
        with guard:
          active -= 1

    with mock.patch.object(shared, 'step', side_effect=step):
      agents.send_timestep(_timestep())
      try:
        self.assertTrue(first_entered.wait(5))
        collided = overlap.wait(0.25)
      finally:
        release.set()
      actions = agents.await_action()
    self.assertFalse(
        collided, 'one policy was entered concurrently through aliases'
    )
    self.assertEqual(actions, (10, 20))
    self.assertEqual(entered, 2)
    self.assertEqual(active, 0)

  def test_distinct_policies_still_step_concurrently(self):
    first, second = _CountingPolicy(), _CountingPolicy()
    agents = self.make_population(first, second)
    agents.reset()
    barrier = threading.Barrier(2, timeout=5)
    original = _CountingPolicy.step

    def step(timestep, prev_state):
      barrier.wait()
      return original(first, timestep, prev_state)

    with (
        mock.patch.object(first, 'step', side_effect=step),
        mock.patch.object(second, 'step', side_effect=step),
    ):
      agents.send_timestep(_timestep())
      self.assertEqual(agents.await_action(), (10, 20))

  def test_aliases_keep_independent_slot_states_and_observable_names(self):
    shared = _CountingPolicy()
    agents = self.make_population(shared, shared)
    names, actions = [], []
    agents.observables().names.subscribe(names.append)
    agents.observables().action.subscribe(actions.append)
    for _ in range(2):
      agents.reset()
      for count, step_type in enumerate(
          (dm_env.StepType.FIRST, dm_env.StepType.MID, dm_env.StepType.LAST)
      ):
        agents.send_timestep(_timestep(step_type))
        self.assertEqual(agents.await_action(), (10 + count, 20 + count))
    self.assertEqual(names, [['first', 'second'], ['first', 'second']])
    self.assertEqual(actions, [(10, 20), (11, 21), (12, 22)] * 2)


if __name__ == '__main__':
  absltest.main()
