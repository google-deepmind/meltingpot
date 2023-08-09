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
"""Configuration for Collaborative Cooking: Ring.

Example video: https://youtu.be/j5v7B9pfG9I

The recipe they must follow is for tomato soup:
1.   Add three tomatoes to the cooking pot.
2.   Wait for the soup to cook (status bar completion).
3.   Bring a bowl to the pot and pour the soup from the pot into the bowl.
4.   Deliver the bowl of soup at the goal location.

This substrate is a pure common interest game. All players share all rewards.

Players have a `5 x 5` observation window.

Map:
Coordination Ring: A layout with two equally successful movement strategies –
(1) both players moving clockwise, and (2) both players moving anti-clockwise.
If players do not coordinate, they will block each other’s movement.
"""

from meltingpot.configs.substrates import collaborative_cooking as base_config
from meltingpot.utils.substrates import specs
from ml_collections import config_dict

build = base_config.build

# Coordination Ring: Another tight layout requiring significant movement
# coordination between the players, this time in terms of moving clockwise vs
# counterclockwise.
ASCII_MAP = """
xx###C#xx
xx#   Cxx
xxDP# #xx
xxO P #xx
xx#OT##xx
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
      "WORLD.RGB": specs.rgb(40, 72),
  })

  # The roles assigned to each player.
  config.valid_roles = frozenset({"default"})
  config.default_player_roles = ("default",) * 2

  return config
