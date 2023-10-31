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

Each player has their own unique color. Players aim to claim territory by
painting walls in their color. Wet paint (dull version of each player's color)
provides no reward. 25 steps after painting a wall, if no other paint was
applied since, the paint dries and changes to the brighter version of the
claiming player's color. All dry paint on a wall yields reward stochastically
to the claiming player with a fixed rate. The more walls a player has claimed
the more reward they may expect to achieve per timestep.

Players can claim a wall in two ways: (1) by touching it with the paintbrush
they carry, and (2) by flinging paint forward. Claimed walls are colored in
the unique color of the player that claimed them. Unclaimed walls are gray.

Once a wall has been claimed a countdown begins. After 25 timesteps the paint
on the wall dries and the wall is said to be `active'. This is visualized by
the paint color brightening. Active walls provide reward stochastically to the
player that claimed them at a rate of 0.01 per timestep. Thus the more resources
a player claims and can hold until the paint dries and they become active, the
more reward they obtain.

Players may fling paint a distance of 2. This allows then to paint over a wall
to simultaneously paint both a near wall and a second wall on the other side of
it. If two players stand on opposite sides of a wall of width 2 and one player
claims all the way across to the other side (closer to the other player than
themselves) then the player on the other side might reasonably perceive that as
a somewhat aggressive action. It is still less aggressive of course than the
other option both players have: using their zapping beam to attack one another
or their claimed walls. If any wall is zapped twice then it is permanently
destroyed. It no longer functions as a wall and it is no longer claimable. Once
a wall has been destroyed then players may freely walk over it.

Players, like walls, are also removed from the game once they are hit twice by a
zapping game. Like walls, players also never regenerate. This is different from
other substrates where being hit by a zapping beam does not cause permanent
removal. In territory, once a player has been zapped out they are gone for good.
All walls they claimed immediately return to the unclaimed state.

In territory__open, all nine players spawn in an open space near one another and
at some distance away from all the resource walls. Some parts of the map are
more rich in resource walls than others.
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
