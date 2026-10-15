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
"""Real-worker regression tests for population shutdown ordering."""

import concurrent.futures
import threading
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import dm_env
from meltingpot.utils.policies import policy
from meltingpot.utils.scenarios import population


class BlockingPolicy(policy.Policy):
  """Keep a step active until the test releases it, with bounded waits."""

  def __init__(self, error=None):
    self.started = threading.Event()
    self.release = threading.Event()
    self.finished = threading.Event()
    self.closed = threading.Event()
    self.events = []
    self.error = error

  def initial_state(self):
    return 0

  def step(self, timestep, prev_state):
    del timestep
    self.events.append('step-started')
    self.started.set()
    try:
      if not self.release.wait(10):
        raise TimeoutError('Test did not release the policy step')
      if self.closed.is_set():
        raise RuntimeError('Policy resources closed while step was running')
      if self.error is not None:
        raise self.error
      return 7, prev_state + 1
    finally:
      self.events.append('step-finished')
      self.finished.set()

  def close(self):
    self.events.append('policy-closed')
    self.closed.set()


class PopulationShutdownTest(parameterized.TestCase):

  def make_population(self, policies):
    names = [str(index) for index in range(len(policies))]
    result = population.Population(
        policies=dict(zip(names, policies)),
        names_by_role={name: [name] for name in names},
        roles=names,
    )

    # Release first even if a test assertion fails, then join every worker.
    def cleanup():
      for bot in policies:
        bot.release.set()
      result._executor.shutdown(wait=True)

    self.addCleanup(cleanup)
    return result

  def send_first(self, group, bots):
    group.reset()
    timestep = dm_env.TimeStep(
        step_type=dm_env.StepType.FIRST,
        reward=[0.0] * len(bots),
        discount=1.0,
        observation=[{} for _ in bots],
    )
    group.send_timestep(timestep)
    for bot in bots:
      self.assertTrue(bot.started.wait(5))

  @parameterized.parameters(False, True)
  def test_close_waits_for_a_running_step_before_closing_its_policy(
      self, failing
  ):
    error = ValueError('synthetic step failure') if failing else None
    bot = BlockingPolicy(error)
    group = self.make_population([bot])
    self.send_first(group, [bot])
    step_future = group._action_futures[0]
    completed = []
    group.observables().timestep.subscribe(
        on_completed=lambda: completed.append('done')
    )
    shutdown_entered = threading.Event()
    shutdown = group._executor.shutdown

    def observed_shutdown(*args, **kwargs):
      shutdown_entered.set()
      return shutdown(*args, **kwargs)

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as closer:
      with mock.patch.object(
          group._executor, 'shutdown', side_effect=observed_shutdown
      ):
        close_future = closer.submit(group.close)
        try:
          self.assertTrue(shutdown_entered.wait(5))
          self.assertFalse(bot.closed.wait(0.2))
          self.assertFalse(close_future.done())
          self.assertEmpty(completed)
        finally:
          bot.release.set()
          close_future.result(timeout=5)
    self.assertEqual(
        bot.events, ['step-started', 'step-finished', 'policy-closed']
    )
    self.assertEqual(completed, ['done'])
    if failing:
      with self.assertRaises(ValueError) as caught:
        step_future.result()
      self.assertIs(caught.exception, error)
    else:
      self.assertEqual(step_future.result(), 7)

  def test_close_waits_for_every_policy(self):
    bots = [BlockingPolicy(), BlockingPolicy()]
    group = self.make_population(bots)
    self.send_first(group, bots)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as closer:
      close_future = closer.submit(group.close)
      try:
        bots[0].release.set()
        self.assertTrue(bots[0].finished.wait(5))
        self.assertFalse(bots[1].closed.wait(0.2))
        self.assertFalse(close_future.done())
      finally:
        for bot in bots:
          bot.release.set()
        close_future.result(timeout=5)
    for bot in bots:
      self.assertEqual(
          bot.events, ['step-started', 'step-finished', 'policy-closed']
      )

  def test_completed_actions_and_observable_completion_are_preserved(self):
    bot = BlockingPolicy()
    bot.release.set()
    group = self.make_population([bot])
    actions, names, completions = [], [], []
    group.observables().action.subscribe(
        actions.append, on_completed=lambda: completions.append('action')
    )
    group.observables().names.subscribe(
        names.append, on_completed=lambda: completions.append('names')
    )
    group.observables().timestep.subscribe(
        on_completed=lambda: completions.append('timestep')
    )
    self.send_first(group, [bot])
    self.assertEqual(group.await_action(), (7,))
    group.close()
    self.assertEqual(actions, [(7,)])
    self.assertEqual(names, [['0']])
    self.assertCountEqual(completions, ['action', 'names', 'timestep'])
    self.assertEqual(
        bot.events, ['step-started', 'step-finished', 'policy-closed']
    )
    with self.assertRaises(RuntimeError):
      group._executor.submit(lambda: None)

  def test_close_before_any_step_still_releases_policies(self):
    bot = BlockingPolicy()
    group = self.make_population([bot])
    group.close()
    self.assertEqual(bot.events, ['policy-closed'])


if __name__ == '__main__':
  absltest.main()
