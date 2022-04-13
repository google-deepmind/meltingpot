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
"""Tests of the human_players levels."""

import collections
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from dm_env import specs
import numpy as np
import pygame

import dmlab2d
from meltingpot.python.configs.substrates import allelopathic_harvest as mp_allelopathic_harvest
from meltingpot.python.configs.substrates import arena_running_with_scissors_in_the_matrix as mp_arena_running_with_scissors_itm
from meltingpot.python.configs.substrates import bach_or_stravinsky_in_the_matrix as mp_bach_or_stravinsky_itm
from meltingpot.python.configs.substrates import capture_the_flag as mp_capture_the_flag
from meltingpot.python.configs.substrates import chemistry_metabolic_cycles as mp_chemistry_metabolic_cycles
from meltingpot.python.configs.substrates import chicken_in_the_matrix as mp_chicken_itm
from meltingpot.python.configs.substrates import clean_up as mp_clean_up
from meltingpot.python.configs.substrates import collaborative_cooking_passable as mp_collaborative_cooking_passable
from meltingpot.python.configs.substrates import commons_harvest_closed as mp_commons_harvest_closed
from meltingpot.python.configs.substrates import king_of_the_hill as mp_king_of_the_hill
from meltingpot.python.configs.substrates import prisoners_dilemma_in_the_matrix as mp_prisoners_dilemma_itm
from meltingpot.python.configs.substrates import pure_coordination_in_the_matrix as mp_pure_coordination_itm
from meltingpot.python.configs.substrates import rationalizable_coordination_in_the_matrix as mp_rationalizable_coordination_itm
from meltingpot.python.configs.substrates import running_with_scissors_in_the_matrix as mp_running_with_scissors_itm
from meltingpot.python.configs.substrates import stag_hunt_in_the_matrix as mp_stag_hunt_itm
from meltingpot.python.configs.substrates import territory_rooms as mp_territory_rooms
from meltingpot.python.human_players import level_playing_utils
from meltingpot.python.human_players import play_allelopathic_harvest
from meltingpot.python.human_players import play_any_paintball_game
from meltingpot.python.human_players import play_anything_in_the_matrix
from meltingpot.python.human_players import play_clean_up
from meltingpot.python.human_players import play_collaborative_cooking
from meltingpot.python.human_players import play_commons_harvest
from meltingpot.python.human_players import play_grid_land
from meltingpot.python.human_players import play_territory


class HumanActionReaderTest(parameterized.TestCase):

  @parameterized.parameters(
      (
          {  # Capture the following key events,
              'move': level_playing_utils.get_direction_pressed,
          },  # given this action name, key pressed, for this player index; and
          pygame.K_w, '1',
          # Expecting this action list out.
          {'1.move': 1, '2.move': 0, '3.move': 0},
      ), (
          {  # Capture the following key events,
              'move': level_playing_utils.get_direction_pressed,
          },  # given this action name, key pressed, for this player index; and
          pygame.K_s, '3',
          # Expecting this action list out.
          {'1.move': 0, '2.move': 0, '3.move': 3},
      ), (
          {  # Capture the following key events,
              'move': level_playing_utils.get_direction_pressed,
          },  # given this action name, key pressed, for this player index; and
          pygame.K_s, '1',
          # Expecting this action list out.
          {'1.move': 3, '2.move': 0, '3.move': 0},
      ), (
          {  # Capture the following key events,
              'move': level_playing_utils.get_direction_pressed,
          },  # given action name, irrelevant key pressed, for player 0; and
          pygame.K_x, '1',
          # Expecting this action list out.
          {'1.move': 0, '2.move': 0, '3.move': 0},
      ), (
          {  # Capture the following key events (don't need to make sense),
              'move': level_playing_utils.get_space_key_pressed,
          },  # given action name, irrelevant key pressed, for player 0; and
          pygame.K_SPACE, '1',
          # Expecting this action list out.
          {'1.move': 1, '2.move': 0, '3.move': 0},
      ),
  )
  @mock.patch.object(pygame, 'key')
  def test_human_action(self, action_map, key_pressed, player_index,
                        expected_action, mock_key):
    retval = collections.defaultdict(bool)
    retval[key_pressed] = True
    mock_key.get_pressed.return_value = retval

    move_array = specs.BoundedArray(
        shape=tuple(), dtype=np.intc, minimum=0, maximum=4, name='move')
    action_spec = {
        '1.move': move_array,
        '2.move': move_array,
        '3.move': move_array,
    }
    with mock.patch.object(dmlab2d, 'Lab2d') as env:
      env.action_spec.return_value = action_spec
      har = level_playing_utils.ActionReader(env, action_map)
      np.testing.assert_array_equal(har.step(player_index), expected_action)


class PlayLevelTest(parameterized.TestCase):

  @parameterized.parameters(
      (mp_allelopathic_harvest, play_allelopathic_harvest),
      (mp_arena_running_with_scissors_itm, play_anything_in_the_matrix),
      (mp_bach_or_stravinsky_itm, play_anything_in_the_matrix),
      (mp_capture_the_flag, play_any_paintball_game),
      (mp_chemistry_metabolic_cycles, play_grid_land),
      (mp_chicken_itm, play_anything_in_the_matrix),
      (mp_clean_up, play_clean_up),
      (mp_collaborative_cooking_passable, play_collaborative_cooking),
      (mp_commons_harvest_closed, play_commons_harvest),
      (mp_king_of_the_hill, play_any_paintball_game),
      (mp_prisoners_dilemma_itm, play_anything_in_the_matrix),
      (mp_pure_coordination_itm, play_anything_in_the_matrix),
      (mp_rationalizable_coordination_itm, play_anything_in_the_matrix),
      (mp_running_with_scissors_itm, play_anything_in_the_matrix),
      (mp_stag_hunt_itm, play_anything_in_the_matrix),
      (mp_territory_rooms, play_territory),
      )
  @mock.patch.object(pygame, 'key')
  @mock.patch.object(pygame, 'display')
  @mock.patch.object(pygame, 'event')
  @mock.patch.object(pygame, 'time')
  def test_run_level(
      self, config_module, play_module, unused_k, unused_d, unused_e, unused_t):
    full_config = config_module.get_config()
    full_config['lab2d_settings']['maxEpisodeLengthFrames'] = 10
    level_playing_utils.run_episode(
        'RGB', {}, play_module._ACTION_MAP, full_config)


if __name__ == '__main__':
  absltest.main()
