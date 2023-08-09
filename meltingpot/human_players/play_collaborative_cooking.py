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
"""A simple human player for testing `collaborative_cooking`.

Use `WASD` keys to move the character around.
Use `Q and E` to turn the character.
Use `SPACE` to use the interact action.
Use `TAB` to switch between players.
"""

import argparse
import json

from meltingpot.configs.substrates import collaborative_cooking__asymmetric
from meltingpot.configs.substrates import collaborative_cooking__circuit
from meltingpot.configs.substrates import collaborative_cooking__cramped
from meltingpot.configs.substrates import collaborative_cooking__crowded
from meltingpot.configs.substrates import collaborative_cooking__figure_eight
from meltingpot.configs.substrates import collaborative_cooking__forced
from meltingpot.configs.substrates import collaborative_cooking__ring
from meltingpot.human_players import level_playing_utils
from ml_collections import config_dict

MAX_SCREEN_WIDTH = 800
MAX_SCREEN_HEIGHT = 600
FRAMES_PER_SECOND = 8


_ACTION_MAP = {
    'move': level_playing_utils.get_direction_pressed,
    'turn': level_playing_utils.get_turn_pressed,
    'interact': level_playing_utils.get_space_key_pressed,
}

environment_configs = {
    'collaborative_cooking__asymmetric': collaborative_cooking__asymmetric,
    'collaborative_cooking__circuit': collaborative_cooking__circuit,
    'collaborative_cooking__cramped': collaborative_cooking__cramped,
    'collaborative_cooking__crowded': collaborative_cooking__crowded,
    'collaborative_cooking__figure_eight': collaborative_cooking__figure_eight,
    'collaborative_cooking__forced': collaborative_cooking__forced,
    'collaborative_cooking__ring': collaborative_cooking__ring,
}


def verbose_fn(env_timestep, player_index, current_player_index):
  if player_index != current_player_index:
    return
  for obs in ['ADDED_INGREDIENT_TO_COOKING_POT',
              'COLLECTED_SOUP_FROM_COOKING_POT']:
    lua_index = player_index + 1
    if env_timestep.observation[f'{lua_index}.{obs}']:
      print(obs, env_timestep.observation[f'{lua_index}.{obs}'])


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
      '--level_name',
      type=str,
      default='collaborative_cooking__cramped',
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

  args = parser.parse_args()
  env_module = environment_configs[args.level_name]
  env_config = env_module.get_config()
  with config_dict.ConfigDict(env_config).unlocked() as env_config:
    roles = env_config.default_player_roles
    env_config.lab2d_settings = env_module.build(roles, env_config)
  level_playing_utils.run_episode(
      args.observation, args.settings, _ACTION_MAP, env_config,
      level_playing_utils.RenderType.PYGAME, MAX_SCREEN_WIDTH,
      MAX_SCREEN_HEIGHT, FRAMES_PER_SECOND,
      verbose_fn if args.verbose else None,
      print_events=args.print_events)


if __name__ == '__main__':
  main()
