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
"""Configuration for Factory of the Commons: Either Or."""

from meltingpot.configs.substrates import factory_commons as base_config
from meltingpot.utils.substrates import map_helpers
from meltingpot.utils.substrates import specs
from ml_collections import config_dict

build = base_config.build

ASCII_MAP = """
;_____________________,
!          c          |
!         cCc         |
!  ls  ls  C  lt  lt  |
!  Oj  Oj     O#  O#  |
!   z   z      z   z  |
!   x   x      x   x  |
!         cCc         |
!         cCc         |
!  ls  ls     lt  lt  |
!  Oj  Oj     O#  O#  |
!   z   z      z   z  |
!   x   x  C   x   x  |
!         cCc         |
!          c          |
_______________________
"""

blue_cube_live = {
    "type": "all", "list": ["tiled_floor", "blue_cube_wait", "blue_cube_live"]}
blue_cube_wait = {
    "type": "all", "list": ["tiled_floor", "blue_cube_wait"]}

# `prefab` determines which prefab game object to use for each `char` in the
# ascii map.
CHAR_PREFAB_MAP = {
    " ": {"type": "all", "list": ["tiled_floor", "apple", "spawn_point"]},
    # Graspable objects.
    "c": map_helpers.a_or_b_with_odds(blue_cube_wait,
                                      blue_cube_live, odds=(1, 1)),
    "C": blue_cube_live,  # This blue cube will always be present.
    # New dynamic components.
    "l": {"type": "all", "list": ["tiled_floor", "hopper_body",
                                  "hopper_indicator_blue_cube"]},
    "O": {"type": "all", "list": ["tiled_floor", "hopper_mouth"]},
    "D": {"type": "all", "list": ["tiled_floor", "dispenser_body",
                                  "dispenser_indicator_apple"]},
    "t": {"type": "all", "list": ["tiled_floor", "dispenser_body",
                                  "dispenser_indicator_two_apples"]},
    "s": {"type": "all", "list": ["tiled_floor", "dispenser_body",
                                  "dispenser_indicator_cube_apple"]},
    "#": {"type": "all", "list": ["tiled_floor", "dispenser_belt",
                                  "apple_dispensing_animation"]},
    "j": {"type": "all", "list": ["tiled_floor", "dispenser_belt",
                                  "cube_apple_dispensing_animation"]},
    "z": {"type": "all", "list": ["tiled_floor", "floor_marking_top"]},
    "x": {"type": "all", "list": ["tiled_floor", "floor_marking_bottom"]},
    # Static components.
    ";": {"type": "all", "list": ["tiled_floor", "nw_wall_corner"]},
    ",": {"type": "all", "list": ["tiled_floor", "ne_wall_corner"]},
    "_": "wall_horizontal",
    "T": "wall_t_coupling",
    "|": {"type": "all", "list": ["tiled_floor", "wall_east"]},
    "!": {"type": "all", "list": ["tiled_floor", "wall_west"]},
    "i": {"type": "all", "list": ["tiled_floor", "wall_middle"]},
    "~": {"type": "all", "list": ["tiled_floor", "threshold"]},
}


def get_config():
  """Default configuration."""
  config = base_config.get_config()
  # Specify a recommended number of players to particate in each episode.
  config.recommended_num_players = 3
  # Override the map layout settings.
  config.layout = config_dict.ConfigDict()
  config.layout.ascii_map = ASCII_MAP
  config.layout.char_prefab_map = CHAR_PREFAB_MAP

  # The roles assigned to each player.
  config.valid_roles = frozenset({"default"})
  config.default_player_roles = ("default",) * 3

  # The specs of the environment (from a single-agent perspective).
  config.timestep_spec = specs.timestep({
      "RGB": specs.OBSERVATION["RGB"],
      "READY_TO_SHOOT": specs.OBSERVATION["READY_TO_SHOOT"],
      "STAMINA": specs.float64(),
      # Debug only.
      "WORLD.RGB": specs.rgb(128, 184),
  })
  return config
