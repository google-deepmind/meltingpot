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
"""Configuration for Territory: Inside Out.

Example video: https://youtu.be/LdbIjnHaisU

See _Territory: Open_ for the general description of the mechanics at play in
this substrate.

In this substrate, _Territory: Inside Out_,  players start on the outside of
a randomly generated maze of resource walls. They must move from their starting
locations inward toward the center of the map to claim territory. In so doing
they will quickly encounter their coplayers who will be doing the same thing
from their own starting locations. In order to get high scores, agents must be
able to rapidly negotiate tacit agreements with one another concerning the
borders between their respective territories. Since the spatial arrangement of
the resource walls differs from episode to episode, so too does the negotiation
problem to be solved.
"""

from meltingpot.configs.substrates import territory as base_config
from meltingpot.utils.substrates import map_helpers
from meltingpot.utils.substrates import specs
from ml_collections import config_dict

build = base_config.build

ASCII_MAP = """
F=====================T
|,,,,,,,,,,P,,,,,,,,,,|
|,P,,,,QQ,,,,,QQ,,,,P,|
|,,RRR,,,,RRR,,,,RRR,,|
|,,R,RAAAAR,RAAAAR,R,,|
|,,RRR,BB,RRR,BB,RRR,,|
|,,,A,,BB,,A,,BB,,A,,,|
|,Q,ABBRRBBABBRRBBA,Q,|
|,Q,ABBRRBBABBRRBBA,Q,|
|,,,A,,BB,,A,,BB,,A,,,|
|,,RRR,BB,RRR,BB,RRR,,|
|P,R,RAAAAR,RAAAAR,R,P|
|,,RRR,BB,RRR,BB,RRR,,|
|,,,A,,BB,,A,,BB,,A,,,|
|,Q,ABBRRBBABBRRBBA,Q,|
|,Q,ABBRRBBABBRRBBA,Q,|
|,,,A,,BB,,A,,BB,,A,,,|
|,,RRR,BB,RRR,BB,RRR,,|
|,,R,RAAAAR,RAAAAR,R,,|
|,,RRR,,,,RRR,,,,RRR,,|
|,P,,,,QQ,,,,,QQ,,,,P,|
|,,,,,,,,,,P,,,,,,,,,,|
L=====================J
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
    "Q": map_helpers.a_or_b_with_odds(spawn_point, "floor", odds=(1, 6)),
    ",": "floor",
    "F": {"type": "all", "list": ["wall", "wall_highlight_nw"]},
    "|": {"type": "all", "list": ["wall", "wall_highlight_e_w"]},
    "=": {"type": "all", "list": ["wall", "wall_highlight_n_s"]},
    "T": {"type": "all", "list": ["wall", "wall_highlight_ne"]},
    "J": {"type": "all", "list": ["wall", "wall_highlight_se"]},
    "L": {"type": "all", "list": ["wall", "wall_highlight_sw"]},
    "R": resource,
    "A": map_helpers.a_or_b_with_odds(resource, "floor", odds=(2, 1)),
    "B": map_helpers.a_or_b_with_odds(resource, "floor", odds=(1, 3)),
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
      "WORLD.RGB": specs.rgb(184, 184),
  })

  # The roles assigned to each player.
  config.default_player_roles = ("default",) * 5

  return config
