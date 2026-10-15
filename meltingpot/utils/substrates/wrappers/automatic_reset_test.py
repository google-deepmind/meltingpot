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
"""Automatic reset paths rebuild environments just like explicit resets."""

from absl.testing import absltest
import dm_env
import dmlab2d
from meltingpot import substrate
from meltingpot.utils.substrates import builder
from meltingpot.utils.substrates.wrappers import reset_wrapper
import numpy as np


class _EpisodeEnvironment(dmlab2d.Environment):
  """Small environment implementing dm_env's automatic-reset contract."""

  def __init__(self, number):
    self.number = number
    self.actions = []
    self.reset_arguments = []
    self.close_count = 0
    self.index = 0
    self.needs_reset = True

  def reset(self, *args, **kwargs):
    self.reset_arguments.append((args, kwargs))
    self.index = 0
    self.needs_reset = False
    return dm_env.restart(self.observation())

  def step(self, action):
    if self.needs_reset:
      return self.reset()
    if action == 'fail':
      raise ValueError('step rejected')
    self.actions.append(action)
    self.index += 1
    if self.index == 2:
      self.needs_reset = True
      return dm_env.termination(7.0, self.observation())
    return dm_env.transition(3.0, self.observation(), discount=0.0)

  def observation(self):
    return {'instance': self.number, 'step': self.index}

  def close(self):
    self.close_count += 1


class AutomaticResetTest(absltest.TestCase):

  def make_wrapper(self):
    environments = []

    def build():
      environment = _EpisodeEnvironment(len(environments))
      environments.append(environment)
      return environment

    wrapped = reset_wrapper.ResetWrapper(build)
    self.addCleanup(wrapped.close)
    return wrapped, environments

  def test_each_terminal_transition_rebuilds_before_the_next_episode(self):
    wrapped, environments = self.make_wrapper()
    timestep = wrapped.reset()
    for episode in range(3):
      self.assertTrue(timestep.first())
      self.assertLen(environments, episode + 1)
      self.assertEqual(timestep.observation['instance'], episode)
      middle = wrapped.step(1)
      self.assertTrue(middle.mid())
      self.assertEqual(middle.discount, 0.0)
      self.assertLen(environments, episode + 1)
      terminal = wrapped.step(2)
      self.assertTrue(terminal.last())
      self.assertEqual(terminal.reward, 7.0)
      previous = environments[-1]
      timestep = wrapped.step('ignored after LAST')
      self.assertEqual(previous.actions, [1, 2])
      self.assertEqual(previous.close_count, 1)
    self.assertTrue(timestep.first())
    self.assertEqual(timestep.observation['instance'], 3)

  def test_first_step_reuses_initial_environment_then_explicit_reset_rebuilds(
      self,
  ):
    wrapped, environments = self.make_wrapper()
    first = wrapped.step('ignored before FIRST')
    self.assertTrue(first.first())
    self.assertLen(environments, 1)
    self.assertEmpty(environments[0].actions)
    self.assertLen(environments[0].reset_arguments, 1)
    wrapped.reset('argument', seed=19)
    self.assertLen(environments, 2)
    self.assertEqual(environments[0].close_count, 1)
    self.assertEqual(
        environments[1].reset_arguments, [(('argument',), {'seed': 19})]
    )
    self.assertTrue(wrapped.step(1).mid())

  def test_reset_after_last_does_not_rebuild_twice_and_step_errors_can_retry(
      self,
  ):
    wrapped, environments = self.make_wrapper()
    wrapped.reset()
    with self.assertRaisesRegex(ValueError, 'step rejected'):
      wrapped.step('fail')
    self.assertTrue(wrapped.step(1).mid())
    self.assertTrue(wrapped.step(2).last())
    wrapped.reset()
    self.assertLen(environments, 2)
    self.assertTrue(wrapped.step(1).mid())
    self.assertLen(environments, 2)
    self.assertEqual(environments[0].close_count, 1)
    self.assertEqual(environments[1].close_count, 0)

  def test_real_lab2d_rebuilds_only_at_automatic_episode_boundaries(self):
    config = substrate.get_config('commons_harvest__open')
    settings = config.lab2d_settings_builder(
        roles=config.default_player_roles, config=config
    )
    settings['maxEpisodeLengthFrames'] = 2
    wrapped = builder.builder(settings, env_seed=137)
    self.addCleanup(wrapped.close)
    action = {
        name: np.zeros(spec.shape, dtype=spec.dtype)
        for name, spec in wrapped.action_spec().items()
    }
    timestep = wrapped.step(action)
    for _ in range(3):
      self.assertTrue(timestep.first())
      environment = wrapped._env
      steps = 0
      while not timestep.last():
        timestep = wrapped.step(action)
        self.assertIs(wrapped._env, environment)
        for name, spec in wrapped.observation_spec().items():
          spec.validate(timestep.observation[name])
        steps += 1
        self.assertLessEqual(steps, 3)
      self.assertEqual(steps, 2)
      timestep = wrapped.step({'ignored after LAST': 99})
      self.assertIsNot(wrapped._env, environment)
      self.assertTrue(timestep.first())


if __name__ == '__main__':
  absltest.main()
