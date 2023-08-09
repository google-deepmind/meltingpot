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
"""Configuration for Externality Mushrooms: Dense.

Example video: https://youtu.be/MwHhg7sa0xs

See base config: externality_mushrooms.py. Here the map is such that mushrooms
may grow anywhere on the map and most of the map can become full of mushrooms.
This may sometimes make it necessary to actively avoid or destroy undesirable
mushrooms.
"""

from meltingpot.configs.substrates import externality_mushrooms as base_config
from meltingpot.utils.substrates import specs
from ml_collections import config_dict

build = base_config.build

ASCII_MAP = """
/_____________________+
'#####################`
!                     |
! R             G     |
!        R            |
!                     |
!           G         |
!   B     O           |
!                  B  |
!        R            |
!                     |
!    B        G       |
!                     |
(---------------------)
"""

# Map a character to the prefab it represents in the ASCII map.
CHAR_PREFAB_MAP = {
    " ": {"type": "all", "list": ["dirt", "spawn_point", "potential_mushroom"]},
    "R": {"type": "all", "list": ["dirt", "red_mushroom"]},
    "G": {"type": "all", "list": ["dirt", "green_mushroom"]},
    "B": {"type": "all", "list": ["dirt", "blue_mushroom"]},
    "O": {"type": "all", "list": ["dirt", "orange_mushroom"]},
    # fence prefabs
    "/": {"type": "all", "list": ["dirt", "nw_wall_corner"]},
    "'": {"type": "all", "list": ["dirt", "nw_inner_wall_corner"]},
    "+": {"type": "all", "list": ["dirt", "ne_wall_corner"]},
    "`": {"type": "all", "list": ["dirt", "ne_inner_wall_corner"]},
    ")": {"type": "all", "list": ["dirt", "se_wall_corner"]},
    "(": {"type": "all", "list": ["dirt", "sw_wall_corner"]},
    "_": {"type": "all", "list": ["dirt", "wall_north"]},
    "|": {"type": "all", "list": ["dirt", "wall_east"]},
    "-": {"type": "all", "list": ["dirt", "wall_south"]},
    "!": {"type": "all", "list": ["dirt", "wall_west"]},
    "#": {"type": "all", "list": ["dirt", "wall_shadow_s"]},
    ">": {"type": "all", "list": ["dirt", "wall_shadow_se"]},
    "<": {"type": "all", "list": ["dirt", "wall_shadow_sw"]},
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
      "READY_TO_SHOOT": specs.OBSERVATION["READY_TO_SHOOT"],
      # Debug only (do not use the following observations in policies).
      "WORLD.RGB": specs.rgb(112, 184),
  })

  config.default_player_roles = ("default",) * 5

  return config
