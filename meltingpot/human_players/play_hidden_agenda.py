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
"""A simple human player for playing the `Hidden Agenda` level interactively.

Use `WASD` keys to move the character around. `Q` and `E` to turn.
Use 'Space' key for the impostor to fire a beam.
Use numerical keys to vote.
Use 'Tab' to switch between players.
"""

import argparse
import json

from meltingpot.configs.substrates import hidden_agenda
from meltingpot.human_players import level_playing_utils
from ml_collections import config_dict

MAX_SCREEN_WIDTH = 800
MAX_SCREEN_HEIGHT = 600
FRAMES_PER_SECOND = 8


_ACTION_MAP = {
    'move': level_playing_utils.get_direction_pressed,
    'turn': level_playing_utils.get_turn_pressed,
    'tag': level_playing_utils.get_space_key_pressed,
    'vote': level_playing_utils.get_key_number_pressed,
}

environment_configs = {
    'hidden_agenda': hidden_agenda,
}


def verbose_fn(env_timestep, player_index, current_player_index):
  """Prints out relevant observations and rewards at every timestep."""
  del current_player_index
  lua_index = player_index + 1
  for obs in ['VOTING']:
    obs_name = f'{lua_index}.{obs}'
    if env_timestep.observation[obs_name].any():
      print(obs_name, env_timestep.observation[obs_name])


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
      '--level_name', type=str, default='hidden_agenda',
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
