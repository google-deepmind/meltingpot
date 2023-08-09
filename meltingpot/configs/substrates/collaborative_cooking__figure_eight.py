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
"""Configuration for Collaborative Cooking: Figure Eight.

Example video: https://youtu.be/hUCbOL5l-Gw

The recipe they must follow is for tomato soup:
1.   Add three tomatoes to the cooking pot.
2.   Wait for the soup to cook (status bar completion).
3.   Bring a bowl to the pot and pour the soup from the pot into the bowl.
4.   Deliver the bowl of soup at the goal location.

This substrate is a pure common interest game. All players share all rewards.

Players have a `5 x 5` observation window.

Map:
Figure Eight: The map is a figure eight shaped maze that generates numerous
places where players might get in one another's way, blocking critical paths.
While it is technically possible for a single player to complete the task on
their own it is very unlikely that poor performing partners would get out of its
way, so in practice, collaboration is essential.
"""

from meltingpot.configs.substrates import collaborative_cooking as base_config
from meltingpot.utils.substrates import specs
from ml_collections import config_dict

build = base_config.build

# Figure Eight: Strong performance on this map requires two stages of teamwork.
ASCII_MAP = """
################
####C#C##C#C####
# P          P #
## ########## ##
#    P   P     #
## ########## ##
#    P   P     #
### #ODTTOD# ###
################
"""


def get_config():
  """Default configuration."""
  config = base_config.get_config()

  # Override the map layout settings.
  config.layout = config_dict.ConfigDict()
  config.layout.ascii_map = ASCII_MAP
  # The specs of the environment (from a single-agent perspective).
  config.timestep_spec = specs.timestep({
      "RGB": specs.rgb(40, 40),
      # Debug only (do not use the following observations in policies).
      "WORLD.RGB": specs.rgb(72, 128),
  })

  # The roles assigned to each player.
  config.valid_roles = frozenset({"default"})
  config.default_player_roles = ("default",) * 6

  return config
