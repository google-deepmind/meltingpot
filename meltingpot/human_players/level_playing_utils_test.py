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
"""Tests for human-player level utilities."""

import contextlib
import io
from unittest import mock

from absl.testing import absltest
import dm_env
from meltingpot.human_players import level_playing_utils
from ml_collections import config_dict
import numpy as np


class RunEpisodeTerminalRewardTest(absltest.TestCase):

  def test_terminal_reward_is_included_in_final_score(self):
    frame = np.zeros((1, 1, 3), dtype=np.uint8)
    first = dm_env.restart(
        observation={
            '1.REWARD': 0.0,
            'WORLD.RGB': frame,
        }
    )
    last = dm_env.termination(
        reward=0.0,
        observation={
            '1.REWARD': 5.0,
            'WORLD.RGB': frame,
        },
    )
    env = mock.MagicMock()
    env.__enter__.return_value = env
    env.action_spec.return_value = {'1.MOVE': mock.sentinel.action_spec}
    env.observation_spec.return_value = {
        'WORLD.RGB': dm_env.specs.Array(
            shape=(1, 1, 3), dtype=np.uint8, name='WORLD.RGB'
        ),
    }
    env.reset.return_value = first
    env.step.return_value = last
    env_builder = mock.Mock(return_value=env)
    full_config = config_dict.ConfigDict({
        'lab2d_settings': {
            'numPlayers': 1,
        },
    })
    output = io.StringIO()

    with contextlib.redirect_stdout(output):
      level_playing_utils.run_episode(
          render_observation='WORLD.RGB',
          config_overrides={},
          action_map={'MOVE': lambda: 0},
          full_config=full_config,
          interactive=level_playing_utils.RenderType.NONE,
          env_builder=env_builder,
      )

    self.assertIn('Player 1: score is 5', output.getvalue())


class RunEpisodePlayerPrefixesTest(absltest.TestCase):

  def test_rejects_prefix_count_mismatch_before_building_environment(self):
    full_config = config_dict.ConfigDict({
        'lab2d_settings': {
            'numPlayers': 2,
        },
    })
    env_builder = mock.Mock()

    with self.assertRaisesRegex(ValueError, 'same length'):
      level_playing_utils.run_episode(
          render_observation='WORLD.RGB',
          config_overrides={},
          action_map={},
          full_config=full_config,
          interactive=level_playing_utils.RenderType.NONE,
          env_builder=env_builder,
          player_prefixes=('red',),
      )

    env_builder.assert_not_called()


class ActionReaderTest(absltest.TestCase):

  def test_only_reads_actions_supported_by_selected_player(self):
    env = mock.Mock()
    env.action_spec.return_value = {
        '1.move': mock.sentinel.move_spec,
        '2.fire': mock.sentinel.fire_spec,
    }
    move = mock.Mock(return_value=3)
    fire = mock.Mock(return_value=1)
    reader = level_playing_utils.ActionReader(
        env, {'move': move, 'fire': fire}
    )

    player_one_actions = reader.step('1')

    self.assertEqual(player_one_actions, {'1.move': 3, '2.fire': 0})
    move.assert_called_once_with()
    fire.assert_not_called()

  def test_uses_other_players_own_action_names(self):
    env = mock.Mock()
    env.action_spec.return_value = {
        '1.move': mock.sentinel.move_spec,
        '2.fire': mock.sentinel.fire_spec,
    }
    move = mock.Mock(return_value=3)
    fire = mock.Mock(return_value=1)
    reader = level_playing_utils.ActionReader(
        env, {'move': move, 'fire': fire}
    )

    player_two_actions = reader.step('2')

    self.assertEqual(player_two_actions, {'1.move': 0, '2.fire': 1})
    move.assert_not_called()
    fire.assert_called_once_with()


if __name__ == '__main__':
  absltest.main()
