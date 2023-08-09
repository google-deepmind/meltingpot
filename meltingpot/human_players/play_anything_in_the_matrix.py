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
"""A simple human player for testing `*_in_the_matrix`.

Use `WASD` keys to move the character around.
Use `Q and E` to turn the character.
Use `SPACE` to fire the interaction beam.
Use `TAB` to switch between players.
"""

import argparse
import json

from meltingpot.configs.substrates import bach_or_stravinsky_in_the_matrix__arena as bach_or_stravinsky_itm
from meltingpot.configs.substrates import bach_or_stravinsky_in_the_matrix__repeated as bach_or_stravinsky_itm__repeated
from meltingpot.configs.substrates import chicken_in_the_matrix__arena as chicken_itm
from meltingpot.configs.substrates import chicken_in_the_matrix__repeated as chicken_itm__repeated
from meltingpot.configs.substrates import prisoners_dilemma_in_the_matrix__arena as prisoners_dilemma_itm
from meltingpot.configs.substrates import prisoners_dilemma_in_the_matrix__repeated as prisoners_dilemma_itm__repeated
from meltingpot.configs.substrates import pure_coordination_in_the_matrix__arena as pure_coord_itm
from meltingpot.configs.substrates import pure_coordination_in_the_matrix__repeated as pure_coord_itm__repeated
from meltingpot.configs.substrates import rationalizable_coordination_in_the_matrix__arena as rational_coord_itm
from meltingpot.configs.substrates import rationalizable_coordination_in_the_matrix__repeated as rational_coord_itm__repeated
from meltingpot.configs.substrates import running_with_scissors_in_the_matrix__arena as rws_itm__arena
from meltingpot.configs.substrates import running_with_scissors_in_the_matrix__one_shot as rws_itm
from meltingpot.configs.substrates import running_with_scissors_in_the_matrix__repeated as rws_itm__repeated
from meltingpot.configs.substrates import stag_hunt_in_the_matrix__arena as stag_hunt_itm
from meltingpot.configs.substrates import stag_hunt_in_the_matrix__repeated as stag_hunt_itm__repeated
from meltingpot.human_players import level_playing_utils
from ml_collections import config_dict

environment_configs = {
    'bach_or_stravinsky_in_the_matrix__arena': bach_or_stravinsky_itm,
    'bach_or_stravinsky_in_the_matrix__repeated':
        bach_or_stravinsky_itm__repeated,
    'chicken_in_the_matrix__arena': chicken_itm,
    'chicken_in_the_matrix__repeated': chicken_itm__repeated,
    'prisoners_dilemma_in_the_matrix__arena': prisoners_dilemma_itm,
    'prisoners_dilemma_in_the_matrix__repeated':
        prisoners_dilemma_itm__repeated,
    'pure_coordination_in_the_matrix__arena': pure_coord_itm,
    'pure_coordination_in_the_matrix__repeated': pure_coord_itm__repeated,
    'rationalizable_coordination_in_the_matrix__arena': rational_coord_itm,
    'rationalizable_coordination_in_the_matrix__repeated':
        rational_coord_itm__repeated,
    'running_with_scissors_in_the_matrix__arena': rws_itm__arena,
    'running_with_scissors_in_the_matrix__one_shot': rws_itm,
    'running_with_scissors_in_the_matrix__repeated': rws_itm__repeated,
    'stag_hunt_in_the_matrix__arena': stag_hunt_itm,
    'stag_hunt_in_the_matrix__repeated': stag_hunt_itm__repeated,
}

_ACTION_MAP = {
    'move': level_playing_utils.get_direction_pressed,
    'turn': level_playing_utils.get_turn_pressed,
    'interact': level_playing_utils.get_space_key_pressed,
}


def verbose_fn(env_timestep, player_index, current_player_index):
  """Print using this function once enabling the option --verbose=True."""
  lua_index = player_index + 1
  collected_resource_1 = env_timestep.observation[
      f'{lua_index}.COLLECTED_RESOURCE_1']
  collected_resource_2 = env_timestep.observation[
      f'{lua_index}.COLLECTED_RESOURCE_2']
  destroyed_resource_1 = env_timestep.observation[
      f'{lua_index}.DESTROYED_RESOURCE_1']
  destroyed_resource_2 = env_timestep.observation[
      f'{lua_index}.DESTROYED_RESOURCE_2']
  interacted_this_step = env_timestep.observation[
      f'{lua_index}.INTERACTED_THIS_STEP']
  argmax_interact_inventory_1 = env_timestep.observation[
      f'{lua_index}.ARGMAX_INTERACTION_INVENTORY_WAS_1']
  argmax_interact_inventory_2 = env_timestep.observation[
      f'{lua_index}.ARGMAX_INTERACTION_INVENTORY_WAS_2']
  # Only print observations from current player.
  if player_index == current_player_index:
    print(
        f'player: {player_index} --- \n' +
        f'  collected_resource_1: {collected_resource_1} \n' +
        f'  collected_resource_2: {collected_resource_2} \n' +
        f'  destroyed_resource_1: {destroyed_resource_1} \n' +
        f'  destroyed_resource_1: {destroyed_resource_2} \n' +
        f'  interacted_this_step: {interacted_this_step} \n' +
        f'  argmax_interaction_inventory_1: {argmax_interact_inventory_1} \n' +
        f'  argmax_interaction_inventory_2: {argmax_interact_inventory_2} \n'
    )


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
      '--level_name', type=str,
      default='prisoners_dilemma_in_the_matrix__repeated',
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
      args.observation, args.settings, _ACTION_MAP,
      env_config, level_playing_utils.RenderType.PYGAME,
      verbose_fn=verbose_fn if args.verbose else None,
      print_events=args.print_events)


if __name__ == '__main__':
  main()
