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
"""Configuration for the substrate: fruit_market_concentric_rivers.

Example video: https://youtu.be/djmylRv1i_w

This substrate has three concentric rings of water that confer a small stamina
cost to players who step on them.
"""

from meltingpot.configs.substrates import fruit_market as base_config
from meltingpot.utils.substrates import specs
from ml_collections import config_dict as configdict

build = base_config.build

ASCII_MAP = """
xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
x/___________________________+x
x'###########################`x
x!~~~~~~~~~~~~~~~~~~~~~~~~~~~|x
x!~~~~~~~~~~~~~~~~~~~~~~~~~~~|x
x!~~~LLLLLLLLLLLLLLLLLLLLL~~~|x
x!~~~L~~~~~~~~~~~~~~~~~~~L~~~|x
x!~~~L~~~~~~~~~~~~~~~~~~~L~~~|x
x!~~~L~~LLLLLLLLLLLLLLL~~L~~~|x
x!~~~L~~L~~~~~~~~~~~~~L~~L~~~|x
x!~~~L~~L~~~~~~~~~~~~~L~~L~~~|x
x!~~~L~~L~~LLLLLLLLL~~L~~L~~~|x
x!~~~L~~L~~LP~P~P~PL~~L~~L~~~|x
x!~~~L~~L~~L~P~P~P~L~~L~~L~~~|x
x!~~~L~~L~~L~~P~P~~L~~L~~L~~~|x
x!~~~L~~L~~L~P~P~P~L~~L~~L~~~|x
x!~~~L~~L~~L~~P~P~~L~~L~~L~~~|x
x!~~~L~~L~~L~P~P~P~L~~L~~L~~~|x
x!~~~L~~L~~LP~P~P~PL~~L~~L~~~|x
x!~~~L~~L~~LLLLLLLLL~~L~~L~~~|x
x!~~~L~~L~~~~~~~~~~~~~L~~L~~~|x
x!~~~L~~L~~~~~~~~~~~~~L~~L~~~|x
x!~~~L~~LLLLLLLLLLLLLLL~~L~~~|x
x!~~~L~~~~~~~~~~~~~~~~~~~L~~~|x
x!~~~L~~~~~~~~~~~~~~~~~~~L~~~|x
x!~~~LLLLLLLLLLLLLLLLLLLLL~~~|x
x!~~~~~~~~~~~~~~~~~~~~~~~~~~~|x
x!~~~~~~~~~~~~~~~~~~~~~~~~~~~|x
x!~~~~~~~~~~~~~~~~~~~~~~~~~~~|x
x(---------------------------)x
x<###########################>x
"""

# Map a character to the prefab it represents in the ASCII map.
CHAR_PREFAB_MAP = {
    # wall prefabs
    "/": {"type": "all", "list": ["ground", "nw_wall_corner"]},
    "'": {"type": "all", "list": ["ground", "nw_inner_wall_corner"]},
    "+": {"type": "all", "list": ["ground", "ne_wall_corner"]},
    "`": {"type": "all", "list": ["ground", "ne_inner_wall_corner"]},
    ")": {"type": "all", "list": ["ground", "se_wall_corner"]},
    "(": {"type": "all", "list": ["ground", "sw_wall_corner"]},
    "_": {"type": "all", "list": ["ground", "wall_north"]},
    "|": {"type": "all", "list": ["ground", "wall_east"]},
    "-": {"type": "all", "list": ["ground", "wall_south"]},
    "!": {"type": "all", "list": ["ground", "wall_west"]},
    "#": {"type": "all", "list": ["ground", "wall_shadow_s"]},
    ">": {"type": "all", "list": ["ground", "wall_shadow_se"]},
    "<": {"type": "all", "list": ["ground", "wall_shadow_sw"]},

    # non-wall prefabs
    "L": "river",
    "P": {"type": "all", "list": ["ground", "potential_tree", "spawn_point"]},
    "~": {"type": "all", "list": ["ground", "potential_tree"]},
    "x": "ground",
}


def get_config():
  """Configuration for this substrate."""
  config = base_config.get_config()
  # Specify the number of players to particate in each episode (optional).
  config.recommended_num_players = 16
  # Override the map layout settings.
  config.layout = configdict.ConfigDict()
  config.layout.ascii_map = ASCII_MAP
  config.layout.char_prefab_map = CHAR_PREFAB_MAP

  # The specs of the environment (from a single-agent perspective).
  config.action_spec = specs.action(len(base_config.ACTION_SET))
  config.timestep_spec = specs.timestep({
      "RGB": specs.OBSERVATION["RGB"],
      "READY_TO_SHOOT": specs.OBSERVATION["READY_TO_SHOOT"],
      "STAMINA": specs.float64(),
      "INVENTORY": specs.int64(2),
      "MY_OFFER": specs.int64(2),
      "OFFERS": specs.int64(102),
      "HUNGER": specs.float64(),
      # Debug only (do not use the following observations in policies).
      "WORLD.RGB": specs.rgb(248, 248,),
  })

  # The roles assigned to each player.
  config.valid_roles = frozenset({"apple_farmer", "banana_farmer"})
  config.default_player_roles = ("apple_farmer",) * 8 + ("banana_farmer",) * 8
  return config
