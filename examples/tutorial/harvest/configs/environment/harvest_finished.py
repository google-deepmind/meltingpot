# Copyright 2020 DeepMind Technologies Limited.
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
"""Configuration for finished tutorial level: Harvest."""

from meltingpot.utils.substrates import shapes
from ml_collections import config_dict

SPAWN_POINT = {
    "name": "spawn_point",
    "components": [
        {
            "component": "StateManager",
            "kwargs": {
                "initialState": "spawnPoint",
                "stateConfigs": [{
                    "state": "spawnPoint",
                    "groups": ["spawnPoints"],
                }],
            }
        },
        {
            "component": "Transform",
        },
    ]
}

AVATAR = {
    "name": "avatar",
    "components": [
        {
            "component": "StateManager",
            "kwargs": {
                "initialState": "player",
                "stateConfigs": [
                    {"state": "player",
                     "layer": "upperPhysical",
                     "contact": "avatar",
                     "sprite": "Avatar",},
                ]
            }
        },
        {
            "component": "Transform",
        },
        {
            "component": "Appearance",
            "kwargs": {
                "renderMode": "ascii_shape",
                "spriteNames": ["Avatar"],
                "spriteShapes": [shapes.CUTE_AVATAR],
                "palettes": [{}],  # Will be overridden
                "noRotates": [True],
            }
        },
        {
            "component": "Avatar",
            "kwargs": {
                "aliveState": "player",
                "waitState": "playerWait",
                "spawnGroup": "spawnPoints",
                "view": {
                    "left": 3,
                    "right": 3,
                    "forward": 5,
                    "backward": 1,
                    "centered": False,
                }
            }
        },
    ]
}


WALL = {
    "name": "wall",
    "components": [
        {
            "component": "StateManager",
            "kwargs": {
                "initialState": "wall",
                "stateConfigs": [{
                    "state": "wall",
                    "layer": "upperPhysical",
                    "sprite": "Wall",
                }],
            }
        },
        {
            "component": "Transform",
        },
        {
            "component": "Appearance",
            "kwargs": {
                "renderMode": "ascii_shape",
                "spriteNames": ["Wall",],
                "spriteShapes": [shapes.WALL],
                "palettes": [shapes.WALL_PALETTE],
                "noRotates": [True],
            }
        },
        {
            "component": "BeamBlocker",
            "kwargs": {
                "beamType": "gift"
            }
        },
        {
            "component": "BeamBlocker",
            "kwargs": {
                "beamType": "zap"
            }
        },
    ]
}

APPLE = {
    "name": "apple",
    "components": [
        {
            "component": "StateManager",
            "kwargs": {
                "initialState": "apple",
                "stateConfigs": [{
                    "state": "apple",
                    "layer": "lowerPhysical",
                    "sprite": "Apple",
                }, {
                    "state": "appleWait",
                }],
            }
        },
        {
            "component": "Transform",
        },
        {
            "component": "Appearance",
            "kwargs": {
                "renderMode": "ascii_shape",
                "spriteNames": ["Apple",],
                "spriteShapes": [shapes.LEGACY_APPLE],
                "palettes": [shapes.GREEN_COIN_PALETTE],
                "noRotates": [True],
            }
        },
        {
            "component": "Edible",
            "kwargs": {
                "liveState": "apple",
                "waitState": "appleWait",
                "rewardForEating": 1.0,
            }
        },
        {
            "component": "DensityRegrow",
            "kwargs": {
                "liveState": "apple",
                "waitState": "appleWait",
                "baseRate": 0.01,
            }
        },
    ]
}


def get_config():
  """Default configuration for the Harvest level."""
  config = config_dict.ConfigDict()

  # Basic configuration.
  config.individual_observation_names = ["RGB"]
  config.global_observation_names = ["WORLD.RGB"]

  ascii_map = """
**********************
*      AAA       AAA *
* AAA   A   AAA   A  *
*AAAAA  _  AAAAA  _  *
* AAA       AAA      *
*      AAA       AAA *
*     AAAAA  _  AAAAA*
*      AAA       AAA *
*  A         A       *
* AAA   _   AAA   _  *
**********************
  """

  # Lua script configuration.
  config.lab2d_settings = {
      "levelName":
          "harvest_finished",
      "levelDirectory":
          "examples/tutorial/harvest/levels",
      "maxEpisodeLengthFrames":
          1000,
      "numPlayers":
          5,
      "spriteSize":
          8,
      "simulation": {
          "map": ascii_map,
          "prefabs": {
              "avatar": AVATAR,
              "spawn_point": SPAWN_POINT,
              "wall": WALL,
              "apple": APPLE,
          },
          "charPrefabMap": {
              "_": "spawn_point",
              "*": "wall",
              "A": "apple"
          },
          "playerPalettes": [],
      },
  }

  return config
