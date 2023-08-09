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
"""Configuration for predator_prey__orchard.

Example video: https://youtu.be/gtd-ziZYJRI

See predator_prey.py for a detailed description applicable to all predator_prey
substrates.

In this variant there are two areas of the map containing food: an apple-rich
region to the north of the safe tall grass and an acorn-rich region to the east.
There are two possible prey strategies focusing on either apples or acorns.
However, in this case it is clear that focusing on acorns is the better
strategy since they are relatively close to the safe tall grass. They can easily
be collected and brought back to safety for consumption.
"""

from meltingpot.configs.substrates import predator_prey as base_config
from meltingpot.utils.substrates import specs
from ml_collections import config_dict

build = base_config.build

ASCII_MAP = """
/;__________,;_______,/
;]aa'X'XX''a|!a''''aA[,
!a''aaaaaa'X[]''aa&''A|
!X'aaAaaaaa''''aaaaa''|
!'&'aaaaaa''Aa'aaaaaa'|
!a'''X''''X'''a''''''a|
!aa''aaa''''''''''''aa|
L~+''aaa''=~~+XXXaA=~~J
;_]'''a'X'[_,L~~~~~J;_,
!XX'''a'X'''[_______]'|
!'''''a''''''XX'''''''|
!'r^^^^^^l'''X'A'''A''|
!'zv#****#^l'''''A''''|
!'''<#***##j'''A'''A''|
!''r###**#>''''''A'''X|
!''zv##***#^l''A'''A''|
!''''zvvvvvvj''''A'&''|
L+'''''''''''''''''''=J
/L~~~~~~~~~~~~~~~~~~~J/
"""

# `prefab` determines which prefab game object to use for each `char` in the
# ascii map.
CHAR_PREFAB_MAP = {
    "*": {"type": "all", "list": ["safe_grass", "spawn_point_prey"]},
    "&": {"type": "all", "list": ["tiled_floor", "spawn_point_prey"]},
    "X": {"type": "all", "list": ["tiled_floor", "spawn_point_predator"]},
    "a": {"type": "all", "list": ["tiled_floor", "apple"]},
    "A": {"type": "all", "list": ["tiled_floor", "floor_acorn"]},
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
    "#": "safe_grass",
    "<": "safe_grass_w_edge",
    "^": "safe_grass_n_edge",
    ">": "safe_grass_e_edge",
    "v": "safe_grass_s_edge",
    "l": "safe_grass_ne_corner",
    "j": "safe_grass_se_corner",
    "z": "safe_grass_sw_corner",
    "r": "safe_grass_nw_corner",
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
      "WORLD.RGB": specs.rgb(152, 184),
  })

  # The roles assigned to each player.
  config.default_player_roles = ("predator",) * 5 + ("prey",) * 8

  return config
