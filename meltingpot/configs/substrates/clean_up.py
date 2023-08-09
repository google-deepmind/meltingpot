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
"""Configuration for Clean Up.

Example video: https://youtu.be/TqiJYxOwdxw

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

from typing import Any, Dict, Mapping, Sequence

from meltingpot.utils.substrates import colors
from meltingpot.utils.substrates import game_object_utils
from meltingpot.utils.substrates import shapes
from meltingpot.utils.substrates import specs
from ml_collections import config_dict

PrefabConfig = game_object_utils.PrefabConfig

# Warning: setting `_ENABLE_DEBUG_OBSERVATIONS = True` may cause slowdown.
_ENABLE_DEBUG_OBSERVATIONS = False

ASCII_MAP = """
WWWWWWWWWWWWWWWWWWWWWWWWWWWWWW
WHFFFHFFHFHFHFHFHFHFHHFHFFFHFW
WHFHFHFFHFHFHFHFHFHFHHFHFFFHFW
WHFFHFFHHFHFHFHFHFHFHHFHFFFHFW
WHFHFHFFHFHFHFHFHFHFHHFHFFFHFW
WHFFFFFFHFHFHFHFHFHFHHFHFFFHFW
W==============+~FHHHHHHf====W
W   P    P      ===+~SSf     W
W     P     P   P  <~Sf  P   W
W             P   P<~S>      W
W   P    P         <~S>   P  W
W               P  <~S>P     W
W     P           P<~S>      W
W           P      <~S> P    W
W  P             P <~S>      W
W^T^T^T^T^T^T^T^T^T;~S,^T^T^TW
WBBBBBBBBBBBBBBBBBBBssBBBBBBBW
WBBBBBBBBBBBBBBBBBBBBBBBBBBBBW
WBBBBBBBBBBBBBBBBBBBBBBBBBBBBW
WBBBBBBBBBBBBBBBBBBBBBBBBBBBBW
WWWWWWWWWWWWWWWWWWWWWWWWWWWWWW
"""

# Map a character to the prefab it represents in the ASCII map.
CHAR_PREFAB_MAP = {
    "W": "wall",
    " ": "sand",
    "P": {"type": "all", "list": ["sand", "spawn_point"]},
    "B": {"type": "all", "list": ["grass", "potential_apple"]},
    "s": {"type": "all", "list": ["grass", "shadow_n"]},
    "+": {"type": "all", "list": ["sand", "shadow_e", "shadow_n"]},
    "f": {"type": "all", "list": ["sand", "shadow_w", "shadow_n"]},
    ";": {"type": "all", "list": ["sand", "grass_edge", "shadow_e"]},
    ",": {"type": "all", "list": ["sand", "grass_edge", "shadow_w"]},
    "^": {"type": "all", "list": ["sand", "grass_edge",]},
    "=": {"type": "all", "list": ["sand", "shadow_n",]},
    ">": {"type": "all", "list": ["sand", "shadow_w",]},
    "<": {"type": "all", "list": ["sand", "shadow_e",]},
    "~": {"type": "all", "list": ["river", "shadow_w",]},
    "T": {"type": "all", "list": ["sand", "grass_edge", "potential_apple"]},
    "S": "river",
    "H": {"type": "all", "list": ["river", "potential_dirt"]},
    "F": {"type": "all", "list": ["river", "actual_dirt"]},
}

_COMPASS = ["N", "E", "S", "W"]

SAND = {
    "name": "sand",
    "components": [
        {
            "component": "StateManager",
            "kwargs": {
                "initialState": "sand",
                "stateConfigs": [{
                    "state": "sand",
                    "layer": "background",
                    "sprite": "Sand",
                }],
            }
        },
        {
            "component": "Appearance",
            "kwargs": {
                "renderMode": "ascii_shape",
                "spriteNames": ["Sand"],
                "spriteShapes": [shapes.GRAINY_FLOOR],
                "palettes": [{"+": (222, 221, 189, 255),
                              "*": (219, 218, 186, 255)}],
                "noRotates": [False]
            }
        },
        {
            "component": "Transform",
        },
    ]
}

GRASS = {
    "name": "grass",
    "components": [
        {
            "component": "StateManager",
            "kwargs": {
                "initialState": "grass",
                "stateConfigs": [{
                    "state": "grass",
                    "layer": "background",
                    "sprite": "Grass",
                }],
            }
        },
        {
            "component": "Appearance",
            "kwargs": {
                "renderMode": "ascii_shape",
                "spriteNames": ["Grass"],
                "spriteShapes": [shapes.GRASS_STRAIGHT],
                "palettes": [{"*": (164, 189, 75, 255),
                              "@": (182, 207, 95, 255),
                              "x": (0, 0, 0, 0)}],
                "noRotates": [False]
            }
        },
        {
            "component": "Transform",
        },
    ]
}

GRASS_EDGE = {
    "name": "grass_edge",
    "components": [
        {
            "component": "StateManager",
            "kwargs": {
                "initialState": "grass_edge",
                "stateConfigs": [{
                    "state": "grass_edge",
                    "layer": "lowerPhysical",
                    "sprite": "GrassEdge",
                }],
            }
        },
        {
            "component": "Appearance",
            "kwargs": {
                "renderMode": "ascii_shape",
                "spriteNames": ["GrassEdge"],
                "spriteShapes": [shapes.GRASS_STRAIGHT_N_EDGE],
                "palettes": [{"*": (164, 189, 75, 255),
                              "@": (182, 207, 95, 255),
                              "x": (0, 0, 0, 0)}],
                "noRotates": [False]
            }
        },
        {
            "component": "Transform",
        },
    ]
}

SHADOW_W = {
    "name": "shadow_w",
    "components": [
        {
            "component": "StateManager",
            "kwargs": {
                "initialState": "shadow_w",
                "stateConfigs": [{
                    "state": "shadow_w",
                    "layer": "upperPhysical",
                    "sprite": "ShadowW",
                }],
            }
        },
        {
            "component": "Appearance",
            "kwargs": {
                "renderMode": "ascii_shape",
                "spriteNames": ["ShadowW"],
                "spriteShapes": [shapes.SHADOW_W],
                "palettes": [shapes.SHADOW_PALETTE],
                "noRotates": [False]
            }
        },
        {
            "component": "Transform",
        },
    ]
}

SHADOW_E = {
    "name": "shadow_e",
    "components": [
        {
            "component": "StateManager",
            "kwargs": {
                "initialState": "shadow_e",
                "stateConfigs": [{
                    "state": "shadow_e",
                    "layer": "upperPhysical",
                    "sprite": "ShadowE",
                }],
            }
        },
        {
            "component": "Appearance",
            "kwargs": {
                "renderMode": "ascii_shape",
                "spriteNames": ["ShadowE"],
                "spriteShapes": [shapes.SHADOW_E],
                "palettes": [shapes.SHADOW_PALETTE],
                "noRotates": [False]
            }
        },
        {
            "component": "Transform",
        },
    ]
}

SHADOW_N = {
    "name": "shadow_n",
    "components": [
        {
            "component": "StateManager",
            "kwargs": {
                "initialState": "shadow_n",
                "stateConfigs": [{
                    "state": "shadow_n",
                    "layer": "overlay",
                    "sprite": "ShadowN",
                }],
            }
        },
        {
            "component": "Appearance",
            "kwargs": {
                "renderMode": "ascii_shape",
                "spriteNames": ["ShadowN"],
                "spriteShapes": [shapes.SHADOW_N],
                "palettes": [shapes.SHADOW_PALETTE],
                "noRotates": [False]
            }
        },
        {
            "component": "Transform",
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
                    "layer": "superOverlay",
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
                        "layer": "upperPhysical",
                    },
                    {
                        "state": "appleWait"
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
                "spriteNames": ["Apple"],
                "spriteShapes": [shapes.APPLE],
                "palettes": [{
                    "x": (0, 0, 0, 0),
                    "*": (212, 80, 57, 255),
                    "#": (173, 66, 47, 255),
                    "o": (43, 127, 53, 255),
                    "|": (79, 47, 44, 255)}],
                "noRotates": [True]
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
                          "layer": "upperPhysical",
                          "sprite": "Dirt",
                      },
                  ],
              }
          },
          {
              "component": "Transform",
          },
          {
              "component": "Appearance",
              "kwargs": {
                  "spriteNames": ["Dirt"],
                  # This color is greenish, and quite transparent to expose the
                  # animated water below.
                  "spriteRGBColors": [(2, 245, 80, 50)],
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

# Remove the first entry from human_readable_colors after using it for the self
# color to prevent it from being used again as another avatar color.
human_readable_colors = list(colors.human_readable)
TARGET_SPRITE_SELF = {
    "name": "Self",
    "shape": shapes.CUTE_AVATAR,
    "palette": shapes.get_palette(human_readable_colors.pop(0)),
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
                  "palettes": [{
                      "@": (66, 173, 212, 255),
                      "*": (35, 133, 168, 255),
                      "o": (34, 129, 163, 255),
                      "~": (33, 125, 158, 255),}] * 4,
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
      "sand": SAND,
      "grass": GRASS,
      "grass_edge": GRASS_EDGE,
      "shadow_w": SHADOW_W,
      "shadow_e": SHADOW_E,
      "shadow_n": SHADOW_N,
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
          {
              "component": "StochasticIntervalEpisodeEnding",
              "kwargs": {
                  "minimumFramesPerEpisode": 1000,
                  "intervalLength": 100,  # Set equal to unroll length.
                  "probabilityTerminationPerInterval": 0.2
              }
          },
          {
              "component": "GlobalData",
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
      "name": f"avatar{lua_index}",
      "components": [
          {
              "component": "StateManager",
              "kwargs": {
                  "initialState": live_state_name,
                  "stateConfigs": [
                      # Initial player state.
                      {"state": live_state_name,
                       "layer": "superOverlay",
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
          },
          {
              "component": "Appearance",
              "kwargs": {
                  "renderMode": "ascii_shape",
                  "spriteNames": [source_sprite_self],
                  "spriteShapes": [shapes.CUTE_AVATAR],
                  "palettes": [shapes.get_palette(
                      human_readable_colors[player_idx])],
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
              "component": "AllNonselfCumulants",
          },
      ]
  }
  # Signals needed for puppeteers.
  metrics = [
      {
          "name": "NUM_OTHERS_WHO_CLEANED_THIS_STEP",
          "type": "Doubles",
          "shape": [],
          "component": "AllNonselfCumulants",
          "variable": "num_others_who_cleaned_this_step",
      },
  ]
  if _ENABLE_DEBUG_OBSERVATIONS:
    avatar_object["components"].append({
        "component": "LocationObserver",
        "kwargs": {"objectIsAvatar": True, "alsoReportOrientation": True},
    })
    # Debug metrics
    metrics.append({
        "name": "PLAYER_CLEANED",
        "type": "Doubles",
        "shape": [],
        "component": "Cleaner",
        "variable": "player_cleaned",
    })
    metrics.append({
        "name": "PLAYER_ATE_APPLE",
        "type": "Doubles",
        "shape": [],
        "component": "Taste",
        "variable": "player_ate_apple",
    })
    metrics.append({
        "name": "NUM_OTHERS_PLAYER_ZAPPED_THIS_STEP",
        "type": "Doubles",
        "shape": [],
        "component": "Zapper",
        "variable": "num_others_player_zapped_this_step",
    })
    metrics.append({
        "name": "NUM_OTHERS_WHO_ATE_THIS_STEP",
        "type": "Doubles",
        "shape": [],
        "component": "AllNonselfCumulants",
        "variable": "num_others_who_ate_this_step",
    })

  # Add the metrics reporter.
  avatar_object["components"].append({
      "component": "AvatarMetricReporter",
      "kwargs": {"metrics": metrics}
  })

  return avatar_object


def create_avatar_objects(num_players):
  """Returns list of avatar objects of length 'num_players'."""
  avatar_objects = []
  for player_idx in range(0, num_players):
    game_object = create_avatar_object(player_idx,
                                       TARGET_SPRITE_SELF)
    avatar_objects.append(game_object)

  return avatar_objects


def get_config():
  """Default configuration for the clean_up level."""
  config = config_dict.ConfigDict()

  # Action set configuration.
  config.action_set = ACTION_SET
  # Observation format configuration.
  config.individual_observation_names = [
      "RGB",
      "READY_TO_SHOOT",
      # Global switching signals for puppeteers.
      "NUM_OTHERS_WHO_CLEANED_THIS_STEP",
  ]
  config.global_observation_names = [
      "WORLD.RGB",
  ]

  # The specs of the environment (from a single-agent perspective).
  config.action_spec = specs.action(len(ACTION_SET))
  config.timestep_spec = specs.timestep({
      "RGB": specs.OBSERVATION["RGB"],
      "READY_TO_SHOOT": specs.OBSERVATION["READY_TO_SHOOT"],
      # Global switching signals for puppeteers.
      "NUM_OTHERS_WHO_CLEANED_THIS_STEP": specs.float64(),
      # Debug only (do not use the following observations in policies).
      "WORLD.RGB": specs.rgb(168, 240),
  })

  # The roles assigned to each player.
  config.valid_roles = frozenset({"default"})
  config.default_player_roles = ("default",) * 7

  return config


def build(
    roles: Sequence[str],
    config: config_dict.ConfigDict,
) -> Mapping[str, Any]:
  """Build the clean_up substrate given roles."""
  del config
  num_players = len(roles)
  # Build the rest of the substrate definition.
  substrate_definition = dict(
      levelName="clean_up",
      levelDirectory="meltingpot/lua/levels",
      numPlayers=num_players,
      # Define upper bound of episode length since episodes end stochastically.
      maxEpisodeLengthFrames=5000,
      spriteSize=8,
      topology="BOUNDED",  # Choose from ["BOUNDED", "TORUS"],
      simulation={
          "map": ASCII_MAP,
          "gameObjects": create_avatar_objects(num_players),
          "scene": create_scene(),
          "prefabs": create_prefabs(),
          "charPrefabMap": CHAR_PREFAB_MAP,
      },
    )
  return substrate_definition
