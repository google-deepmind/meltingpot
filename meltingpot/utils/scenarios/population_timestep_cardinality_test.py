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
"""Population timesteps are validated before submitting partial worker batches."""

from absl.testing import absltest
from absl.testing import parameterized
import dm_env
from meltingpot.utils.policies import policy
from meltingpot.utils.scenarios import population
import numpy as np


class _CountingPolicy(policy.Policy[int]):

  def __init__(self):
    self.seen = []

  def initial_state(self):
    return 0

  def step(self, timestep, prev_state):
    self.seen.append((timestep, prev_state))
    return int(timestep.observation['offset']) + prev_state, prev_state + 1

  def close(self):
    pass


def _timestep(observations=2, rewards=2, step_type=dm_env.StepType.MID):
  return dm_env.TimeStep(
      step_type,
      tuple(float(n) for n in range(rewards)),
      1.0,
      tuple({'offset': 10 * (n + 1)} for n in range(observations)),
  )


class PopulationTimestepCardinalityTest(parameterized.TestCase):

  def make_population(self):
    policies = (_CountingPolicy(), _CountingPolicy())
    agents = population.Population(
        policies={'first': policies[0], 'second': policies[1]},
        names_by_role={'left': ['first'], 'right': ['second']},
        roles=['left', 'right'],
    )
    self.addCleanup(agents.close)
    # Unchanged upstream close does not join workers; keep failure cleanup safe.
    self.addCleanup(agents._executor.shutdown, wait=True)
    agents.reset()
    return agents, policies

  @parameterized.parameters((0, 2), (1, 2), (3, 2), (2, 0), (2, 1), (2, 3))
  def test_malformed_batch_is_rejected_before_observers_or_worker_state_changes(
      self, observations, rewards
  ):
    agents, policies = self.make_population()
    seen = []
    agents.observables().timestep.subscribe(seen.append)
    agents.send_timestep(_timestep(step_type=dm_env.StepType.FIRST))
    self.assertEqual(agents.await_action(), (10, 20))
    with self.assertRaisesRegex(ValueError, 'Expected 2'):
      agents.send_timestep(_timestep(observations, rewards))
    self.assertLen(seen, 1)
    for p in policies:
      self.assertLen(p.seen, 1)
    agents.send_timestep(_timestep())
    self.assertEqual(agents.await_action(), (11, 21))
    agents.send_timestep(_timestep(step_type=dm_env.StepType.LAST))
    self.assertEqual(agents.await_action(), (12, 22))
    self.assertLen(seen, 3)

  @parameterized.parameters(tuple, list, np.asarray)
  def test_valid_sequence_types_preserve_player_data_and_action_order(
      self, container
  ):
    agents, policies = self.make_population()
    timestep = _timestep(step_type=dm_env.StepType.FIRST)
    timestep = timestep._replace(
        observation=container(timestep.observation),
        reward=container(timestep.reward),
    )
    agents.send_timestep(timestep)
    self.assertEqual(agents.await_action(), (10, 20))
    for n, p in enumerate(policies):
      received, state = p.seen[0]
      self.assertEqual(received.observation['offset'], 10 * (n + 1))
      self.assertEqual(received.reward, float(n))
      self.assertEqual(received.step_type, dm_env.StepType.FIRST)
      self.assertEqual(state, 0)

  def test_pending_action_error_precedes_input_validation_without_losing_actions(
      self,
  ):
    agents, _ = self.make_population()
    agents.send_timestep(_timestep(step_type=dm_env.StepType.FIRST))
    with self.assertRaisesRegex(RuntimeError, 'Previous action'):
      agents.send_timestep(_timestep(1, 1))
    self.assertEqual(agents.await_action(), (10, 20))
    agents.send_timestep(_timestep())
    self.assertEqual(agents.await_action(), (11, 21))


if __name__ == '__main__':
  absltest.main()
