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
"""Configuration for Collaborative Cooking: Circuit.

Example video: https://youtu.be/2nXe5OPvJ7g

The recipe they must follow is for tomato soup:
1.   Add three tomatoes to the cooking pot.
2.   Wait for the soup to cook (status bar completion).
3.   Bring a bowl to the pot and pour the soup from the pot into the bowl.
4.   Deliver the bowl of soup at the goal location.

This substrate is a pure common interest game. All players share all rewards.

Players have a `5 x 5` observation window.

Map:
Counter Circuit: Players are able to cook and deliver soups by themselves
through walking around the entire circuit. However, there exists a more optimal
coordinated strategy whereby players pass tomatoes across the counter.
Additionally, there are the clockwise and anti-clockwise strategies as in the
Ring layout.
"""

from ml_collections import config_dict

from meltingpot.python.configs.substrates import collaborative_cooking as base_config
from meltingpot.python.utils.substrates import specs

build = base_config.build

# Counter Circuit: Another layout where it is possible for agents to work
# independently but more efficient if they work together, with one agent passing
# tomatoes to the other.
ASCII_MAP = """
x###CC###
x#P     #
xD #### T
x#     P#
x###OO###
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
