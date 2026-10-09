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
"""Self-contained tests for the runnable substrate API example."""

import contextlib
import io
import types
import unittest
from unittest import mock

from examples.tutorial import substrate_basics


class _Timestep:

  def __init__(self, is_last=False):
    self.observation = ({'RGB': None}, {'RGB': None}, {'RGB': None})
    self.reward = (0.0, 0.0, 0.0)
    self._is_last = is_last

  def last(self):
    return self._is_last


class _Environment:

  def __init__(self):
    self.reset_calls = 0
    self.step_actions = []
    self.terminated_steps = False

  def __enter__(self):
    return self

  def __exit__(self, *_args):
    return None

  def reset(self):
    self.reset_calls += 1
    return _Timestep()

  def step(self, actions):
    self.step_actions.append(actions)
    return _Timestep(is_last=self.terminated_steps)

  def action_spec(self):
    return ('discrete',) * 3

  def observation_spec(self):
    return ('image',) * 3

  def reward_spec(self):
    return ('scalar',) * 3


class SubstrateBasicsTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    self.environment = _Environment()
    self.config = types.SimpleNamespace(
        valid_roles=('predator', 'prey'),
        default_player_roles=('predator', 'prey', 'prey'),
    )
    self.substrate_api = mock.Mock()
    self.substrate_api.get_config.return_value = self.config
    self.substrate_api.build.return_value = self.environment
    meltingpot = types.ModuleType('meltingpot')
    setattr(meltingpot, 'substrate', self.substrate_api)
    self.module_patch = mock.patch.dict(
        'sys.modules', {'meltingpot': meltingpot}
    )
    self.module_patch.start()
    self.addCleanup(self.module_patch.stop)

  def test_default_roles_and_steps(self):
    with contextlib.redirect_stdout(io.StringIO()) as stdout:
      substrate_basics.explore_substrate('predator_prey__open', steps=2)
    self.substrate_api.get_config.assert_called_once_with('predator_prey__open')
    self.substrate_api.build.assert_called_once_with(
        'predator_prey__open', roles=('predator', 'prey', 'prey')
    )
    self.assertEqual(self.environment.step_actions, [[0, 0, 0]] * 2)
    self.assertEqual(self.environment.reset_calls, 1)
    self.assertIn('Valid roles:', stdout.getvalue())
    self.assertIn('Action specification:', stdout.getvalue())

  def test_explicit_valid_assignment(self):
    with contextlib.redirect_stdout(io.StringIO()):
      substrate_basics.explore_substrate(
          'predator_prey__open', roles=['prey', 'predator'], steps=0
      )
    self.substrate_api.build.assert_called_once_with(
        'predator_prey__open', roles=('prey', 'predator')
    )
    self.assertEqual(self.environment.step_actions, [])

  def test_inspect_only_does_not_build(self):
    with contextlib.redirect_stdout(io.StringIO()) as stdout:
      substrate_basics.explore_substrate(
          'predator_prey__open', inspect_only=True
      )
    self.substrate_api.build.assert_not_called()
    self.assertIn('Default roles', stdout.getvalue())

  def test_invalid_role_has_discoverable_options(self):
    with self.assertRaisesRegex(ValueError, 'valid roles'):
      substrate_basics.explore_substrate(
          'predator_prey__open', roles=['unknown']
      )
    self.substrate_api.build.assert_not_called()

  def test_empty_roles_are_rejected(self):
    with self.assertRaisesRegex(ValueError, 'At least one'):
      substrate_basics.explore_substrate('predator_prey__open', roles=[])
    self.substrate_api.build.assert_not_called()

  def test_negative_step_count_is_rejected(self):
    with self.assertRaisesRegex(ValueError, 'steps must not be negative'):
      substrate_basics.explore_substrate('predator_prey__open', steps=-1)
    self.substrate_api.get_config.assert_not_called()

  def test_terminated_episode_is_reset_before_next_step(self):
    self.environment.terminated_steps = True
    with contextlib.redirect_stdout(io.StringIO()):
      substrate_basics.explore_substrate('predator_prey__open', steps=3)
    self.assertEqual(self.environment.reset_calls, 3)
    self.assertEqual(len(self.environment.step_actions), 3)

  def test_cli_accepts_role_overrides(self):
    with contextlib.redirect_stdout(io.StringIO()):
      result = substrate_basics.main([
          '--substrate',
          'predator_prey__open',
          '--roles',
          'prey',
          'predator',
          '--steps',
          '1',
      ])
    self.assertEqual(result, 0)
    self.substrate_api.build.assert_called_once_with(
        'predator_prey__open', roles=('prey', 'predator')
    )

  def test_cli_fails_early_for_bad_roles(self):
    with contextlib.redirect_stderr(io.StringIO()):
      with self.assertRaises(SystemExit) as error:
        substrate_basics.main(['--roles', 'invalid'])
    self.assertEqual(error.exception.code, 2)
    self.substrate_api.build.assert_not_called()

  def test_cli_inspection_skips_native_environment(self):
    with contextlib.redirect_stdout(io.StringIO()):
      self.assertEqual(substrate_basics.main(['--inspect-only']), 0)
    self.substrate_api.build.assert_not_called()


if __name__ == '__main__':
  unittest.main()
