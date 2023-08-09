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
"""Configuration for predator_prey__random_forest.

Example video: https://youtu.be/ZYkXwvn5_Sc

See predator_prey.py for a detailed description applicable to all predator_prey
substrates.

In this variant there are only acorns, no apples. And, there is no fully safe
tall grass. The tall grass that there is on this map is never large enough for
prey to be fully safe from predation. The grass merely provides an obstacle that
predators must navigate around while chasing prey.
"""

from meltingpot.configs.substrates import predator_prey as base_config
from meltingpot.utils.substrates import map_helpers
from meltingpot.utils.substrates import specs
from ml_collections import config_dict

build = base_config.build

ASCII_MAP = """
/;___________________,/
;]XAXXXXXXXAXXXXXXXAX[,
!XXXXXXXXXXXXXXXXXXXXX|
!''''M'M''MMM''M'M''''|
!'M''M'MM''Q''MM'M''M'|
!'MQ'M''MMMMMMM''M'QM'|
!''''''QM'''''MQ''''''|
!M'MMMMMM@@@@@MMMMMM'M|
!M''''''@@@@@@@''''''M|
!Q'MMQ''@@@A@@@''QMM'Q|
!M''''''@@@@@@@''''''M|
!M'MMMMMM@@@@@MMMMMM'M|
!''''''QM'''''MQ''''''|
!'MQ'M''MMMMMMM''M'QM'|
!'M''M'MM''Q''MM'M''M'|
!''''M'M''MMM''M'M''''|
!XXXXXXXXXXXXXXXXXXXXX|
L+XAXXXXXXXAXXXXXXXAX=J
/L~~~~~~~~~~~~~~~~~~~J/
"""

prey_spawn_point = {"type": "all", "list": ["tiled_floor", "spawn_point_prey"]}
predator_spawn_point = {"type": "all", "list": ["tiled_floor",
                                                "spawn_point_predator"]}
acorn = {"type": "all", "list": ["tiled_floor", "floor_acorn"]}

# `prefab` determines which prefab game object to use for each `char` in the
# ascii map.
CHAR_PREFAB_MAP = {
    "@": prey_spawn_point,
    "*": {"type": "all", "list": ["safe_grass", "spawn_point_prey"]},
    "&": {"type": "all", "list": ["tiled_floor", "apple", "spawn_point_prey"]},
    "X": predator_spawn_point,
    "a": {"type": "all", "list": ["tiled_floor", "apple"]},
    "A": acorn,
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
    "Q": map_helpers.a_or_b_with_odds(acorn, "tiled_floor", odds=(1, 2)),
    "M": map_helpers.a_or_b_with_odds("safe_grass", "tiled_floor", odds=(1, 2)),
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
