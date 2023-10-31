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
"""Configuration for Territory: Rooms.

Example video: https://youtu.be/4URkGR9iv9k

See _Territory: Open_ for the general description of the mechanics at play in
this substrate.

In this substrate, _Territory: Rooms_, players start in segregated rooms which
strongly suggest a partition they could adhere to. They can break down the walls
of their regions and invade each other's ``natural territory'', but the
destroyed resource walls are then lost forever. A peaceful partition is possible
at the start of the episode, and the policy to achieve it is easy to implement.
But if any agent gets too greedy and invades, it buys itself a chance of large
rewards, but also chances inflicting significant chaos and deadweight loss on
everyone if its actions spark wider conflict. The reason it can spiral out of
control is that once an agent's neighbor has left their natural territory then
it becomes rational to invade the space, leaving one's own territory undefended,
creating more opportunity for mischief by others.
"""

from meltingpot.configs.substrates import territory as base_config
from meltingpot.utils.substrates import specs
from ml_collections import config_dict

build = base_config.build

ASCII_MAP = """
JRRRRRLJRRRRRLJRRRRRL
R,,,,,RR,,,,,RR,,,,,R
R,,,,,RR,,,,,RR,,,,,R
R,,P,,RR,,P,,RR,,P,,R
R,,,,,RR,,,,,RR,,,,,R
R,,,,,RR,,,,,RR,,,,,R
TRRRRRFTRRRRRFTRRRRRF
JRRRRRLJRRRRRLJRRRRRL
R,,,,,RR,,,,,RR,,,,,R
R,,,,,RR,,,,,RR,,,,,R
R,,P,,RR,,P,,RR,,P,,R
R,,,,,RR,,,,,RR,,,,,R
R,,,,,RR,,,,,RR,,,,,R
TRRRRRFTRRRRRFTRRRRRF
JRRRRRLJRRRRRLJRRRRRL
R,,,,,RR,,,,,RR,,,,,R
R,,,,,RR,,,,,RR,,,,,R
R,,P,,RR,,P,,RR,,P,,R
R,,,,,RR,,,,,RR,,,,,R
R,,,,,RR,,,,,RR,,,,,R
TRRRRRFTRRRRRFTRRRRRF
"""

# `prefab` determines which prefab game object to use for each `char` in the
# ascii map.
resource_associated_prefabs = ["floor", "resource_texture", "resource",
                               "reward_indicator", "damage_indicator"]
resource = {"type": "all", "list": resource_associated_prefabs}
spawn_point_associated_prefabs = ["floor", "spawn_point"]
spawn_point = {"type": "all", "list": spawn_point_associated_prefabs}
CHAR_PREFAB_MAP = {
    "P": spawn_point,
    ",": "floor",
    "W": "wall",
    "F": {"type": "all", "list": ["wall", "wall_highlight_nw"]},
    "T": {"type": "all", "list": ["wall", "wall_highlight_ne"]},
    "J": {"type": "all", "list": ["wall", "wall_highlight_se"]},
    "L": {"type": "all", "list": ["wall", "wall_highlight_sw"]},
    "R": resource,
}


def get_config():
  """Default configuration."""
  config = base_config.get_config()

  # Override the map layout settings.
  config.layout = config_dict.ConfigDict()
  config.layout.ascii_map = ASCII_MAP
  config.layout.char_prefab_map = CHAR_PREFAB_MAP
  config.layout.topology = "TORUS"

  # The specs of the environment (from a single-agent perspective).
  config.timestep_spec = specs.timestep({
      "RGB": specs.OBSERVATION["RGB"],
      "READY_TO_SHOOT": specs.OBSERVATION["READY_TO_SHOOT"],
      # Debug only (do not use the following observations in policies).
      "WORLD.RGB": specs.rgb(168, 168),
  })

  # The roles assigned to each player.
  config.default_player_roles = ("default",) * 9

  return config
