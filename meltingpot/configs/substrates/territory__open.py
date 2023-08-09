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
"""Configuration for Territory: Open.

Example video: https://youtu.be/F1OO6LFIZHI

Players can claim a resource in two ways: (1) by touching it, and (2) by using a
"claiming beam", different from the zapping beam, which they also have.
Claimed resources are colored in the unique color of the player that claimed
them. Unclaimed resources are gray. Players cannot walk through resources, they
are like walls.

Once a resource has been claimed a countdown begins. After 100 timesteps, the
claimed resource becomes active. This is visualized by a white and gray plus
sign appearing on top. Active resources provide reward stochastically to the
player that claimed them at a rate of 0.01 per timestep. Thus the more resources
a player claims and can hold until they become active, the more reward they
obtain.

The claiming beam is of length 2. It can color through a resource to
simultaneously color a second resource on the other side. If two players stand
on opposite sides of a wall of resources of width 2 and one player claims all
the way across to the other side (closer to the other player than themselves)
then the player on the other side might reasonably perceive that as a somewhat
aggressive action. Less aggressive of course than the other option both players
have: using their zapping beam. If any resource is zapped twice then it gets
permanently destroyed. It no longer functions as a wall or a resource, allowing
players to pass through.

Like resources, when players are hit by a zapping beam they also get removed
from the game and never regenerate. Once a player has been zapped out it is
gone. All resources it claimed are immediately returned to the unclaimed state.
"""

from meltingpot.configs.substrates import territory as base_config
from meltingpot.utils.substrates import specs
from ml_collections import config_dict

build = base_config.build

ASCII_MAP = """
F=====================================T
|,,RRRRR,,RR,,RR,,,,,,RR,,,,,,RR,,,,,,|
|,,,,,RR,,,,,,RR,,,,,,RR,,,,,,,,,,,,,,|
|,,,,,RR,,,,,,RR,,,,,,,,,,,,,,,,,,,,,,|
|,RR,,RR,,,,,,RR,,,,,,,,,,R,,,RR,,,RR,|
|,,,,,RR,,,,,,RR,,,,,,,,,,R,,,RR,,,,,,|
|,,,,,RR,,,,,,,,,,RRRR,,,,R,,,,,,,,,,,|
|,,RR,RR,,,,,,,,,,,,,,,,,,R,,,,,,,,,,,|
|,,,,,RR,,,,,,,RR,,,,,,,,,R,,,,,,,,,,,|
|,,,,,RRRR,,,,,,,,,,,,,,,,,,,,,RR,,,,,|
|,,,,,,,,,,,,,,,,,,,,RR,,,,,,,,,,,,,,,|
|,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,|
|,,RRRR,,,RRRRRR,,,,,,,,,,,RR,,,,R,,,,|
|,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,R,,,,|
|,,,,,,,,,,,,,,,,RR,,,,,,,,,,,,,,,,P,,|
|,,,,RR,,,,,,,,,,,,,,,,RR,,,,,,,P,,,,,|
|,,,,,,,,,RR,,,,,,,,,,,,,,,,,,,,,P,,P,|
|,,,,,,,,,,,,,,,,,,,,,,,,,,,P,,P,,,,,,|
|,,,,,,,,,,,,,,,,,,,,,,,,,,,,,P,,,P,,,|
|,,P,,,,P,,,P,,P,,,P,,,,P,P,,P,,P,,P,,|
|,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,|
|,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,|
L=====================================J
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
    "F": {"type": "all", "list": ["wall", "wall_highlight_nw"]},
    "|": {"type": "all", "list": ["wall", "wall_highlight_e_w"]},
    "=": {"type": "all", "list": ["wall", "wall_highlight_n_s"]},
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
  config.layout.topology = "BOUNDED"

  # The specs of the environment (from a single-agent perspective).
  config.timestep_spec = specs.timestep({
      "RGB": specs.OBSERVATION["RGB"],
      "READY_TO_SHOOT": specs.OBSERVATION["READY_TO_SHOOT"],
      # Debug only (do not use the following observations in policies).
      "WORLD.RGB": specs.rgb(184, 312),
  })

  # The roles assigned to each player.
  config.default_player_roles = ("default",) * 9

  return config
