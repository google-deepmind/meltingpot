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
"""Configuration for Collaborative Cooking: Forced.

Example video: https://youtu.be/FV_xZuSCRmM

The recipe they must follow is for tomato soup:
1.   Add three tomatoes to the cooking pot.
2.   Wait for the soup to cook (status bar completion).
3.   Bring a bowl to the pot and pour the soup from the pot into the bowl.
4.   Deliver the bowl of soup at the goal location.

This substrate is a pure common interest game. All players share all rewards.

Players have a `5 x 5` observation window.

Map:
Forced Coordination: One player is in the left room and second player is in the
right room. Consequently, both players are forced to work together in order to
cook and deliver soup. The player in the left room can only pass tomatoes and
dishes, while the player on the right can only cook the soup and deliver it
(using the items provided by the first player).
"""

from meltingpot.configs.substrates import collaborative_cooking as base_config
from meltingpot.utils.substrates import specs
from ml_collections import config_dict

build = base_config.build

# Forced Coordination: A two-room layout in which agents cannot complete the
# task alone and therefore must work together, with one player passing tomatoes
# and plates to the other, and the other player loading the pot and delivering
# soups.
ASCII_MAP = """
xx###C#xx
xxO #PCxx
xxOP# #xx
xxD # #xx
xx###T#xx
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
