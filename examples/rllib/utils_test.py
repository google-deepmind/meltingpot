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
"""Tests for utils.py."""

# Ray and Gymnasium are optional example dependencies and absent from core CI.
# pylint: disable=import-error
# pytype: disable=import-error

import importlib.util
import unittest

from absl.testing import absltest

def _module_available(module):
  try:
    return importlib.util.find_spec(module) is not None
  except ModuleNotFoundError:
    return False


_RLLIB_AVAILABLE = all(
    _module_available(module) for module in ('gymnasium', 'ray.rllib'))


@unittest.skipUnless(_RLLIB_AVAILABLE, 'requires the optional RLlib extras')
class MeltingPotEnvTests(absltest.TestCase):
  """Tests for MeltingPotEnv for RLLib."""

  def setUp(self):
    super().setUp()
    from gymnasium.spaces import discrete  # pylint: disable=g-import-not-at-top
    from meltingpot import substrate  # pylint: disable=g-import-not-at-top
    from meltingpot.configs.substrates import (  # pylint: disable=g-import-not-at-top
        commons_harvest__open)
    from . import utils  # pylint: disable=g-import-not-at-top

    self._discrete = discrete
    self._substrate = substrate
    self._substrate_config = commons_harvest__open
    self._utils = utils
    # Create a new MeltingPotEnv for each test case
    env_config = self._substrate.get_config('commons_harvest__open')
    roles = env_config.default_player_roles
    self._num_players = len(roles)
    self._env = self._utils.env_creator({
        'substrate': 'commons_harvest__open',
        'roles': roles,
    })

  def test_action_space_size(self):
    """Test the action space is the correct size."""
    actions_count = len(self._substrate_config.ACTION_SET)
    env_action_space = self._env.action_space['player_1']
    self.assertEqual(env_action_space, self._discrete.Discrete(actions_count))

  def test_reset_number_agents(self):
    """Test that reset() returns observations for all agents."""
    obs, _ = self._env.reset()
    self.assertLen(obs, self._num_players)

  def test_step(self):
    """Test step() returns rewards for all agents."""
    self._env.reset()

    # Create dummy actions
    actions = {}
    for player_idx in range(0, self._num_players):
      actions['player_' + str(player_idx)] = 1

    # Step
    _, rewards, _, _, _ = self._env.step(actions)

    # Check we have one reward per agent
    self.assertLen(rewards, self._num_players)

  def test_render_modes_metadata(self):
    """Test that render modes are given in the metadata."""
    self.assertIn('rgb_array', self._env.metadata['render.modes'])

  def test_render_rgb_array(self):
    """Test that render('rgb_array') returns the full world."""
    self._env.reset()
    render = self._env.render()
    self.assertEqual(render.shape, (144, 192, 3))


if __name__ == '__main__':
  absltest.main()
