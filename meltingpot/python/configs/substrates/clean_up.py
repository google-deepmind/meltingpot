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
"""Configuration for Clean Up.

Example video: https://youtu.be/jOeIunFtTS0

Clean Up is a seven player game. Players are rewarded for collecting apples. In
Clean Up, apples grow in an orchard at a rate inversely related to the
cleanliness of a nearby river. The river accumulates pollution at a constant
rate. The apple growth rate in the orchard drops to zero once the pollution
accumulates past a threshold value. Players have an additional action allowing
them to clean a small amount of pollution from the river in front of themselves.
They must physically leave the apple orchard to clean the river. Thus, players
must maintain a public good of high orchard regrowth rate through effortful
contributions. This is a public good provision problem because the benefit of a
healthy orchard is shared by all, but the costs incurred to ensure it exists are
born by individuals.

Players are also able to zap others with a beam that removes any player hit by
it from the game for 50 steps.

Clean Up was first described in Hughes et al. (2018).

Hughes, E., Leibo, J.Z., Phillips, M., Tuyls, K., Duenez-Guzman, E.,
Castaneda, A.G., Dunning, I., Zhu, T., McKee, K., Koster, R. and Roff, H., 2018,
Inequity aversion improves cooperation in intertemporal social dilemmas. In
Proceedings of the 32nd International Conference on Neural Information
Processing Systems (pp. 3330-3340).
"""

from typing import Any, Dict

from ml_collections import config_dict
from meltingpot.python.utils.substrates import colors
from meltingpot.python.utils.substrates import game_object_utils
from meltingpot.python.utils.substrates import shapes
from meltingpot.python.utils.substrates import specs

PrefabConfig = game_object_utils.PrefabConfig


ASCII_MAP = """
WWWWWWWWWWWWWWWWWWWWWWWWWWWWWW
WHFFFHFFHFHFHFHFHFHFHHFHFFFHFW
WHFHFHFFHFHFHFHFHFHFHHFHFFFHFW
WHFFHFFHHFHFHFHFHFHFHHFHFFFHFW
WHFHFHFFHFHFHFHFHFHFHHFHFFFHFW
WHFFFFFFHFHFHFHFHFHFHHFHFFFHFW
W               HFHHHHHH     W
W   P    P          SSS      W
W     P     P   P   SS   P   W
W             P   PPSS       W
W   P    P          SS    P  W
W               P   SS P     W
W     P           P SS       W
W           P       SS  P    W
W  P             P PSS       W
W B B B B B B B B B SSB B B BW
WBBBBBBBBBBBBBBBBBBBBBBBBBBBBW
WBBBBBBBBBBBBBBBBBBBBBBBBBBBBW
WBBBBBBBBBBBBBBBBBBBBBBBBBBBBW
WBBBBBBBBBBBBBBBBBBBBBBBBBBBBW
WWWWWWWWWWWWWWWWWWWWWWWWWWWWWW
"""

# Map a character to the prefab it represents in the ASCII map.
CHAR_PREFAB_MAP = {
    "W": "wall",
    "P": "spawn_point",
    "B": "potential_apple",
    "S": "river",
    "H": {"type": "all", "list": ["river", "potential_dirt"]},
    "F": {"type": "all", "list": ["river", "actual_dirt"]},
}

_COMPASS = ["N", "E", "S", "W"]

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
            "component": "Appearance",
            "kwargs": {
                "renderMode": "ascii_shape",
                "spriteNames": ["Wall"],
                "spriteShapes": [shapes.WALL],
                "palettes": [{"*": (95, 95, 95, 255),
                              "&": (100, 100, 100, 255),
                              "@": (109, 109, 109, 255),
                              "#": (152, 152, 152, 255)}],
                "noRotates": [False]
            }
        },
        {
            "component": "Transform",
        },
        {
            "component": "BeamBlocker",
            "kwargs": {
                "beamType": "zapHit"
            }
        },
        {
            "component": "BeamBlocker",
            "kwargs": {
                "beamType": "cleanHit"
            }
        },
    ]
}

SPAWN_POINT = {
    "name": "spawnPoint",
    "components": [
        {
            "component": "StateManager",
            "kwargs": {
                "initialState": "spawnPoint",
                "stateConfigs": [{
                    "state": "spawnPoint",
                    "layer": "logic",
                    "groups": ["spawnPoints"]
                }],
            }
        },
        {
            "component": "Transform",
            "kwargs": {
                "position": (0, 0),
                "orientation": "N"
            }
        },
    ]
}

POTENTIAL_APPLE = {
    "name": "potentialApple",
    "components": [
        {
            "component": "StateManager",
            "kwargs": {
                "initialState": "appleWait",
                "stateConfigs": [
                    {
                        "state": "apple",
                        "sprite": "Apple",
                        "layer": "lowerPhysical",
                    },
                    {
                        "state": "appleWait"
                    }],
            }
        },
        {
            "component": "Transform",
            "kwargs": {
                "position": (0, 0),
                "orientation": "N"
            }
        },
        {
            "component": "Appearance",
            "kwargs": {
                "renderMode": "ascii_shape",
                "spriteNames": ["Apple"],
                "spriteShapes": [shapes.APPLE],
                "palettes": [{"*": (102, 255, 0, 255),
                              "@": (230, 255, 0, 255),
                              "&": (117, 255, 26, 255),
                              "#": (255, 153, 0, 255),
                              "x": (0, 0, 0, 0)}],
                "noRotates": [False]
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
            "component": "AppleGrow",
            "kwargs": {
                "maxAppleGrowthRate": 0.05,
                "thresholdDepletion": 0.4,
                "thresholdRestoration": 0.0,
            }
        }
    ]
}


def create_dirt_prefab(initial_state):
  """Create a dirt prefab with the given initial state."""
  dirt_prefab = {
      "name": "DirtContainer",
      "components": [
          {
              "component": "StateManager",
              "kwargs": {
                  "initialState": initial_state,
                  "stateConfigs": [
                      {
                          "state": "dirtWait",
                          "layer": "logic",
                      },
                      {
                          "state": "dirt",
                          "layer": "lowerPhysical",
                          "sprite": "Dirt",
                      },
                  ],
              }
          },
          {
              "component": "Transform",
              "kwargs": {
                  "position": (0, 0),
                  "orientation": "N"
              }
          },
          {
              "component": "Appearance",
              "kwargs": {
                  "spriteNames": ["Dirt"],
                  # This color is greenish, and quite transparent to expose the
                  # animated water below.
                  "spriteRGBColors": [(2, 230, 80, 50)],
              }
          },
          {
              "component": "DirtTracker",
              "kwargs": {
                  "activeState": "dirt",
                  "inactiveState": "dirtWait",
              }
          },
          {
              "component": "DirtCleaning",
              "kwargs": {}
          },
      ]
  }
  return dirt_prefab

# Primitive action components.
# pylint: disable=bad-whitespace
# pyformat: disable
NOOP        = {"move": 0, "turn":  0, "fireZap": 0, "fireClean": 0}
FORWARD     = {"move": 1, "turn":  0, "fireZap": 0, "fireClean": 0}
STEP_RIGHT  = {"move": 2, "turn":  0, "fireZap": 0, "fireClean": 0}
BACKWARD    = {"move": 3, "turn":  0, "fireZap": 0, "fireClean": 0}
STEP_LEFT   = {"move": 4, "turn":  0, "fireZap": 0, "fireClean": 0}
TURN_LEFT   = {"move": 0, "turn": -1, "fireZap": 0, "fireClean": 0}
TURN_RIGHT  = {"move": 0, "turn":  1, "fireZap": 0, "fireClean": 0}
FIRE_ZAP    = {"move": 0, "turn":  0, "fireZap": 1, "fireClean": 0}
FIRE_CLEAN  = {"move": 0, "turn":  0, "fireZap": 0, "fireClean": 1}
# pyformat: enable
# pylint: enable=bad-whitespace

ACTION_SET = (
    NOOP,
    FORWARD,
    BACKWARD,
    STEP_LEFT,
    STEP_RIGHT,
    TURN_LEFT,
    TURN_RIGHT,
    FIRE_ZAP,
    FIRE_CLEAN
)

TARGET_SPRITE_SELF = {
    "name": "Self",
    "shape": shapes.CUTE_AVATAR,
    "palette": shapes.get_palette((50, 100, 200)),
    "noRotate": True,
}


def get_water():
  """Get an animated water game object."""
  layer = "background"
  water = {
      "name": "water_{}".format(layer),
      "components": [
          {
              "component": "StateManager",
              "kwargs": {
                  "initialState": "water_1",
                  "stateConfigs": [
                      {"state": "water_1",
                       "layer": layer,
                       "sprite": "water_1",
                       "groups": ["water"]},
                      {"state": "water_2",
                       "layer": layer,
                       "sprite": "water_2",
                       "groups": ["water"]},
                      {"state": "water_3",
                       "layer": layer,
                       "sprite": "water_3",
                       "groups": ["water"]},
                      {"state": "water_4",
                       "layer": layer,
                       "sprite": "water_4",
                       "groups": ["water"]},
                  ]
              }
          },
          {"component": "Transform",},
          {
              "component": "Appearance",
              "kwargs": {
                  "renderMode": "ascii_shape",
                  "spriteNames": ["water_1", "water_2", "water_3", "water_4"],
                  "spriteShapes": [shapes.WATER_1, shapes.WATER_2,
                                   shapes.WATER_3, shapes.WATER_4],
                  "palettes": [shapes.WATER_PALETTE] * 4,
              }
          },
          {
              "component": "Animation",
              "kwargs": {
                  "states": ["water_1", "water_2", "water_3", "water_4"],
                  "gameFramesPerAnimationFrame": 2,
                  "loop": True,
                  "randomStartFrame": True,
                  "group": "water",
              }
          },
      ]
  }
  return water


def create_prefabs() -> PrefabConfig:
  """Returns the prefabs.

  Prefabs are a dictionary mapping names to template game objects that can
  be cloned and placed in multiple locations accoring to an ascii map.
  """
  prefabs = {
      "wall": WALL,
      "spawn_point": SPAWN_POINT,
      "potential_apple": POTENTIAL_APPLE,
      "river": get_water(),
      "potential_dirt": create_dirt_prefab("dirtWait"),
      "actual_dirt": create_dirt_prefab("dirt"),
  }
  return prefabs


def create_scene():
  """Create the scene object, a non-physical object to hold global logic."""
  scene = {
      "name": "scene",
      "components": [
          {
              "component": "StateManager",
              "kwargs": {
                  "initialState": "scene",
                  "stateConfigs": [{
                      "state": "scene",
                  }],
              }
          },
          {
              "component": "Transform",
              "kwargs": {
                  "position": (0, 0),
                  "orientation": "N"
              },
          },
          {
              "component": "RiverMonitor",
              "kwargs": {},
          },
          {
              "component": "DirtSpawner",
              "kwargs": {
                  "dirtSpawnProbability": 0.5,
                  "delayStartOfDirtSpawning": 50,
              },
          },
      ]
  }
  return scene


def create_avatar_object(player_idx: int,
                         target_sprite_self: Dict[str, Any]) -> Dict[str, Any]:
  """Create an avatar object that always sees itself as blue."""
  # Lua is 1-indexed.
  lua_index = player_idx + 1

  # Setup the self vs other sprite mapping.
  source_sprite_self = "Avatar" + str(lua_index)
  custom_sprite_map = {source_sprite_self: target_sprite_self["name"]}

  live_state_name = "player{}".format(lua_index)
  avatar_object = {
      "name": "avatar",
      "components": [
          {
              "component": "StateManager",
              "kwargs": {
                  "initialState": live_state_name,
                  "stateConfigs": [
                      # Initial player state.
                      {"state": live_state_name,
                       "layer": "upperPhysical",
                       "sprite": source_sprite_self,
                       "contact": "avatar",
                       "groups": ["players"]},

                      # Player wait type for times when they are zapped out.
                      {"state": "playerWait",
                       "groups": ["playerWaits"]},
                  ]
              }
          },
          {
              "component": "Transform",
              "kwargs": {
                  "position": (0, 0),
                  "orientation": "N"
              }
          },
          {
              "component": "Appearance",
              "kwargs": {
                  "renderMode": "ascii_shape",
                  "spriteNames": [source_sprite_self],
                  "spriteShapes": [shapes.CUTE_AVATAR],
                  "palettes": [shapes.get_palette(colors.palette[player_idx])],
                  "noRotates": [True]
              }
          },
          {
              "component": "AdditionalSprites",
              "kwargs": {
                  "renderMode": "ascii_shape",
                  "customSpriteNames": [target_sprite_self["name"]],
                  "customSpriteShapes": [target_sprite_self["shape"]],
                  "customPalettes": [target_sprite_self["palette"]],
                  "customNoRotates": [target_sprite_self["noRotate"]],
              }
          },
          {
              "component": "Avatar",
              "kwargs": {
                  "index": lua_index,
                  "aliveState": live_state_name,
                  "waitState": "playerWait",
                  "spawnGroup": "spawnPoints",
                  "actionOrder": ["move",
                                  "turn",
                                  "fireZap",
                                  "fireClean"],
                  "actionSpec": {
                      "move": {"default": 0, "min": 0, "max": len(_COMPASS)},
                      "turn": {"default": 0, "min": -1, "max": 1},
                      "fireZap": {"default": 0, "min": 0, "max": 1},
                      "fireClean": {"default": 0, "min": 0, "max": 1},
                  },
                  "view": {
                      "left": 5,
                      "right": 5,
                      "forward": 9,
                      "backward": 1,
                      "centered": False
                  },
                  "spriteMap": custom_sprite_map,
              }
          },
          {
              "component": "Zapper",
              "kwargs": {
                  "cooldownTime": 10,
                  "beamLength": 3,
                  "beamRadius": 1,
                  "framesTillRespawn": 50,
                  "penaltyForBeingZapped": 0,
                  "rewardForZapping": 0,
                  "removeHitPlayer": True,
              }
          },
          {
              "component": "ReadyToShootObservation",
          },
          {
              "component": "Cleaner",
              "kwargs": {
                  "cooldownTime": 2,
                  "beamLength": 3,
                  "beamRadius": 1,
              }
          },
          {
              "component": "Taste",
              "kwargs": {
                  "role": "free",
                  "rewardAmount": 1,
              }
          },
          {
              "component": "LocationObserver",
              "kwargs": {
                  "objectIsAvatar": True,
                  "alsoReportOrientation": True
              }
          },
      ]
  }
  return avatar_object


def create_avatar_objects(num_players):
  """Returns list of avatar objects of length 'num_players'."""
  avatar_objects = []
  for player_idx in range(0, num_players):
    game_object = create_avatar_object(player_idx,
                                       TARGET_SPRITE_SELF)
    avatar_objects.append(game_object)

  return avatar_objects


def create_lab2d_settings(num_players: int) -> Dict[str, Any]:
  """Returns the lab2d settings."""
  lab2d_settings = {
      "levelName": "clean_up",
      "levelDirectory":
          "meltingpot/lua/levels",
      "numPlayers": num_players,
      "maxEpisodeLengthFrames": 1000,
      "spriteSize": 8,
      "topology": "BOUNDED",  # Choose from ["BOUNDED", "TORUS"],
      "simulation": {
          "map": ASCII_MAP,
          "gameObjects": create_avatar_objects(num_players),
          "prefabs": create_prefabs(),
          "charPrefabMap": CHAR_PREFAB_MAP,
          "scene": create_scene(),
      },
  }
  return lab2d_settings


def get_config():
  """Default configuration for training on the clean_up level."""
  config = config_dict.ConfigDict()

  # Basic configuration.
  config.num_players = 7

  # Lua script configuration.
  config.lab2d_settings = create_lab2d_settings(config.num_players)

  # Action set configuration.
  config.action_set = ACTION_SET
  # Observation format configuration.
  config.individual_observation_names = [
      "RGB",
      "READY_TO_SHOOT",
      "POSITION",
      "ORIENTATION",
  ]
  config.global_observation_names = [
      "WORLD.RGB",
  ]

  # The specs of the environment (from a single-agent perspective).
  config.action_spec = specs.action(len(ACTION_SET))
  config.timestep_spec = specs.timestep({
      "RGB": specs.OBSERVATION["RGB"],
      "READY_TO_SHOOT": specs.OBSERVATION["READY_TO_SHOOT"],
      "POSITION": specs.OBSERVATION["POSITION"],
      "ORIENTATION": specs.OBSERVATION["ORIENTATION"],
      "WORLD.RGB": specs.rgb(168, 240),
  })

  return config
