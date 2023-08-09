# Copyright 2022 DeepMind Technologies Limited.
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
"""A simple human player for testing `boat_race`.

Use `WASD` keys to move the character around.
Use `Q and E` to turn the character.
Use ` ` to row (effectively, but needs coordinated stroke).
Use `x` to flail (row ineffectively, but with safe steady progress).
Use `TAB` to switch between players.
"""

import argparse
import json

from meltingpot.configs.substrates import boat_race__eight_races
from meltingpot.human_players import level_playing_utils
from meltingpot.utils.substrates import game_object_utils
from ml_collections import config_dict

MAX_SCREEN_WIDTH = 600
MAX_SCREEN_HEIGHT = 800
FRAMES_PER_SECOND = 8

environment_configs = {
    'boat_race__eight_races': boat_race__eight_races,
}

_ACTION_MAP = {
    'move': level_playing_utils.get_direction_pressed,
    'turn': level_playing_utils.get_turn_pressed,
    'row': level_playing_utils.get_space_key_pressed,
    'flail': level_playing_utils.get_key_x_pressed,
}


def verbose_fn(env_timestep, player_index, current_player_index):
  lua_index = player_index + 1
  if (env_timestep.observation['WORLD.RACE_START'].any() and
      player_index == current_player_index):
    print('WORLD.RACE_START', env_timestep.observation['WORLD.RACE_START'])
  for obs in [f'{lua_index}.PADDLES', f'{lua_index}.FLAILS']:
    if env_timestep.observation[obs]:
      print(obs, env_timestep.observation[obs])


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
      '--level_name', type=str, default='boat_race__eight_races',
      choices=environment_configs.keys(),
      help='Level name to load')
  parser.add_argument(
      '--observation', type=str, default='RGB', help='Observation to render')
  parser.add_argument(
      '--settings', type=json.loads, default={}, help='Settings as JSON string')
  # Activate verbose mode with --verbose=True.
  parser.add_argument(
      '--verbose', type=bool, default=False, help='Print debug information')
  # Activate events printing mode with --print_events=True.
  parser.add_argument(
      '--print_events', type=bool, default=False, help='Print events')
  parser.add_argument(
      '--override_flail_effectiveness', type=float, default=0.1,
      help='Override flail effectiveness to make debugging easier.')

  args = parser.parse_args()
  env_module = environment_configs[args.level_name]
  env_config = env_module.get_config()
  with config_dict.ConfigDict(env_config).unlocked() as env_config:
    roles = env_config.default_player_roles
    env_config.lab2d_settings = env_module.build(roles, env_config)
    # For easier debug, override the flailEffectiveness
    game_object_utils.get_first_named_component(
        env_config.lab2d_settings['simulation']['prefabs']['seat_L'],
        'BoatManager'
    )['kwargs']['flailEffectiveness'] = args.override_flail_effectiveness

  level_playing_utils.run_episode(
      args.observation, args.settings, _ACTION_MAP, env_config,
      level_playing_utils.RenderType.PYGAME, MAX_SCREEN_WIDTH,
      MAX_SCREEN_HEIGHT, FRAMES_PER_SECOND,
      verbose_fn if args.verbose else None,
      print_events=args.print_events)


if __name__ == '__main__':
  main()
