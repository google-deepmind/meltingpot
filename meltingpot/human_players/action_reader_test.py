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
"""Tests for human-player action reading."""

from unittest import mock

from absl.testing import absltest
from meltingpot.human_players import level_playing_utils


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
