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
"""Configuration for predator_prey__alley_hunt.

Example video: https://youtu.be/ctVjhn7VYgo

See predator_prey.py for a detailed description applicable to all predator_prey
substrates.

In this variant prey must forage for apples in a maze with many dangerous
dead-end corridors where they could easily be trapped by predators.
"""

from meltingpot.configs.substrates import predator_prey as base_config
from meltingpot.utils.substrates import specs
from ml_collections import config_dict

build = base_config.build

ASCII_MAP = """
;________________________,
!aa''''''''''''''''''''aa|
!a'''''''''a''=+''''''''a|
!''=~~~+''=+''|!''=~~~+''|
!''[__,!''|!''|!''[___]''|
!''''a|!''|!aa|!'''''''''|
!''=~~J!''|L~~J!'a'=~~~+'|
!''|///!''[____]'a'|///!a|
!''|///!'''''''''''[__,L~J
!''[___]'XX''''X''''<*[__,
!''''''''''a''''XX''<****|
!'aa'''X''''''a'''XX<****|
!''''''''''a''''XX''<****|
!''=~~~+'''''''X''''<*=~~J
!''|///!'XX''''''''=~~J;_,
!''|///!''=~~~~+'a'|///!a|
!''[__,!''|;__,!'a'[___]'|
!''''a|!''|!aa|!'''''''''|
!''=~~J!''|!''|!''=~~~+''|
!''[___]''[]''|!''[___]''|
!a'''''''''a''[]''''''''a|
!aa''''''''''''''''''''aa|
L~~~~~~~~~~~~~~~~~~~~~~~~J
"""

# `prefab` determines which prefab game object to use for each `char` in the
# ascii map.
CHAR_PREFAB_MAP = {
    "*": {"type": "all", "list": ["safe_grass", "spawn_point_prey"]},
    "X": {"type": "all", "list": ["tiled_floor", "spawn_point_predator"]},
    "a": {"type": "all", "list": ["tiled_floor", "apple"]},
    ";": "nw_wall_corner",
    ",": "ne_wall_corner",
    "J": "se_wall_corner",
    "L": "sw_wall_corner",
    "_": "wall_north",
    "|": "wall_east",
    "~": "wall_south",
    "!": "wall_west",
    "=": "nw_inner_wall_corner",
    "+": "ne_inner_wall_corner",
    "]": "se_inner_wall_corner",
    "[": "sw_inner_wall_corner",
    "'": "tiled_floor",
    "<": "safe_grass_w_edge",
    ">": "safe_grass",
    "/": "fill",
}


def get_config():
  """Default configuration."""
  config = base_config.get_config()

  # Override the map layout settings.
  config.layout = config_dict.ConfigDict()
  config.layout.ascii_map = ASCII_MAP
  config.layout.char_prefab_map = CHAR_PREFAB_MAP

  # The specs of the environment (from a single-agent perspective).
  config.timestep_spec = specs.timestep({
      "RGB": specs.OBSERVATION["RGB"],
      "STAMINA": specs.float64(),
      # Debug only (do not use the following observations in policies).
      "WORLD.RGB": specs.rgb(184, 208),
  })

  # The roles assigned to each player.
  config.default_player_roles = ("predator",) * 5 + ("prey",) * 8

  return config
