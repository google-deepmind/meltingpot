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
"""Checks serialized access to policy objects registered under multiple names."""

import threading
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import dm_env
from meltingpot.utils.policies import policy as policy_lib
from meltingpot.utils.scenarios import population


class _CountingPolicy(policy_lib.Policy[int]):
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
