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
"""A human player for testing fruit_market.

Note: The real agents can make and accept offers up to size 3 (up to 3 apples
for up to 3 bananas). However this human player script only allows offers up to
size 1. The reason is just that we started to run out of keys on the keyboard to
represent higher offers.

Use `WASD` keys to move the player around.
Use `Q and E` to turn the player.
Use `TAB` to switch which player you are controlling.
Use 'Z' to eat an apple from your inventory.
Use 'X' to eat a banana from your inventory.
"""
import argparse
import json

from meltingpot.configs.substrates import fruit_market__concentric_rivers
from meltingpot.human_players import level_playing_utils
from ml_collections import config_dict
import pygame


def get_offer_apple_pressed() -> int:
  """Sets apple offer to either -1, 0, or 1."""
  key_pressed = pygame.key.get_pressed()
  if key_pressed[pygame.K_1]:
    return -1
  if key_pressed[pygame.K_2]:
    return 1
  return 0


def get_offer_banana_pressed() -> int:
  """Sets banana offer to either -1, 0, or 1."""
  key_pressed = pygame.key.get_pressed()
  if key_pressed[pygame.K_3]:
    return -1
  if key_pressed[pygame.K_4]:
    return 1
  return 0


def get_push_pull() -> int:
  """Sets shove to either -1, 0, or 1."""
  if level_playing_utils.get_right_shift_pressed():
    return 1
  if level_playing_utils.get_left_control_pressed():
    return -1
  return 0

environment_configs = {
    'fruit_market__concentric_rivers': fruit_market__concentric_rivers,
}

_ACTION_MAP = {
    # Basic movement actions
    'move': level_playing_utils.get_direction_pressed,
    'turn': level_playing_utils.get_turn_pressed,
    # Trade actions
    'eat_apple': level_playing_utils.get_key_z_pressed,
    'eat_banana': level_playing_utils.get_key_x_pressed,
    'offer_apple': get_offer_apple_pressed,  # 1 and 2
    'offer_banana': get_offer_banana_pressed,  # 3 and 4
    'offer_cancel': level_playing_utils.get_key_number_five_pressed,
    # Grappling actions
    'hold': level_playing_utils.get_space_key_pressed,
    'shove': get_push_pull,
}


def verbose_fn(env_timestep, player_index, current_player_index):
  """Print using this function once enabling the option --verbose=True."""
  lua_index = player_index + 1
  inventory = env_timestep.observation[f'{lua_index}.INVENTORY']
  hunger = env_timestep.observation[f'{lua_index}.HUNGER']
  my_offer = env_timestep.observation[f'{lua_index}.MY_OFFER']
  offers = env_timestep.observation[f'{lua_index}.OFFERS']
  # Only print offer observations from player 0.
  if player_index == current_player_index:
    print(
        f'player: {player_index} --- inventory: {inventory}, hunger: {hunger}')
    print(f'**player 0 view of offers:\n{offers}')
    print(f'**player 0 view of own offer: {my_offer}')


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
      '--level_name',
      type=str,
      default='fruit_market__concentric_rivers',
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
