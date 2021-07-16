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

"""A simple human player for testing paintball games.

Use `WASD` keys to move the character around.
Use `Q and E` to turn the character.
Use `SPACE` to fire the zapper.
Use `TAB` to switch between players.
"""

import argparse
import json

from meltingpot.python.configs.substrates import capture_the_flag as mp_capture_the_flag
from meltingpot.python.configs.substrates import king_of_the_hill as mp_king_of_the_hill
from meltingpot.python.human_players import level_playing_utils


environment_configs = {
    'mp_capture_the_flag': mp_capture_the_flag,
    'mp_king_of_the_hill': mp_king_of_the_hill,
}

_ACTION_MAP = {
    'move': level_playing_utils.get_direction_pressed,
    'turn': level_playing_utils.get_turn_pressed,
    'fireZap': level_playing_utils.get_space_key_pressed,
}


def verbose_fn(unused_env, unused_player_index):
  pass


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
      '--level_name', type=str, default='mp_capture_the_flag',
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
  env_config = environment_configs[args.level_name]
  level_playing_utils.run_episode(
      args.observation, args.settings, _ACTION_MAP,
      env_config.get_config(), level_playing_utils.RenderType.PYGAME,
      verbose_fn=verbose_fn if args.verbose else None,
      print_events=args.print_events)


if __name__ == '__main__':
  main()
