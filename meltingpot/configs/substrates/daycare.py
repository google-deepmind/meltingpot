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
"""Configuration for daycare."""

import copy
from typing import Any, Dict, Mapping, Sequence

from meltingpot.utils.substrates import colors
from meltingpot.utils.substrates import shapes
from meltingpot.utils.substrates import specs
from ml_collections import config_dict as configdict

# Warning: setting `_ENABLE_DEBUG_OBSERVATIONS = True` may cause slowdown.
_ENABLE_DEBUG_OBSERVATIONS = False

_COMPASS = ["N", "E", "S", "W"]

ASCII_MAP = """
/__________________+
!~~~~~~~~~~~~~~~~~~|
!~~~~~~~~~~~~~~~~~~|
!~~~~~~~~~~~~~~~~~~|
!~~~~~~~~~~~~~~~~~~|
!~~~~~~~PPP~~~~~~~~|
!~~~~~~~PPP~~~~~~~~|
!~~~~~~~PPP~~~~~~~~|
!~~~~~~~~~~~~~~~~~~|
!~~~~~~~~~~~~~~~~~~|
!~~~~~~~~~~~~~~~~~~|
!~~~~~~~~~~~~~~~~~~|
(------------------)
"""

# Map a character to the prefab it represents in the ASCII map.
CHAR_PREFAB_MAP = {
    # wall prefabs
    "/": "nw_wall_corner",
    "+": "ne_wall_corner",
    ")": "se_wall_corner",
    "(": "sw_wall_corner",
    "_": "wall_north",
    "|": "wall_east",
    "-": "wall_south",
    "!": "wall_west",

    # non-wall prefabs
    "P": {"type": "all", "list": ["ground", "spawn_point"]},
    "~": {"type": "all", "list": ["ground", "tree", "fruit"]},
}
INVISIBLE = (0, 0, 0, 0)

NW_WALL_CORNER = {
    "name": "nw_wall_corner",
    "components": [
        {
            "component": "StateManager",
            "kwargs": {
                "initialState": "nw_wall_corner",
                "stateConfigs": [{
                    "state": "nw_wall_corner",
                    "layer": "superOverlay",
                    "sprite": "NwWallCorner",
                }],
            }
        },
        {"component": "Transform"},
        {
            "component": "Appearance",
            "kwargs": {
                "renderMode": "ascii_shape",
                "spriteNames": ["NwWallCorner"],
                "spriteShapes": [shapes.BRICK_WALL_NW_CORNER],
                "palettes": [shapes.BRICK_WALL_PALETTE],
                "noRotates": [False]
            }
        },
    ]
}

NE_WALL_CORNER = {
    "name": "ne_wall_corner",
    "components": [
        {
            "component": "StateManager",
            "kwargs": {
                "initialState": "ne_wall_corner",
                "stateConfigs": [{
                    "state": "ne_wall_corner",
                    "layer": "superOverlay",
                    "sprite": "NeWallCorner",
                }],
            }
        },
        {"component": "Transform"},
        {
            "component": "Appearance",
            "kwargs": {
                "renderMode": "ascii_shape",
                "spriteNames": ["NeWallCorner"],
                "spriteShapes": [shapes.BRICK_WALL_NE_CORNER],
                "palettes": [shapes.BRICK_WALL_PALETTE],
                "noRotates": [False]
            }
        },
    ]
}

SE_WALL_CORNER = {
    "name": "se_wall_corner",
    "components": [
        {
            "component": "StateManager",
            "kwargs": {
                "initialState": "se_wall_corner",
                "stateConfigs": [{
                    "state": "se_wall_corner",
                    "layer": "superOverlay",
                    "sprite": "SeWallCorner",
                }],
            }
        },
        {"component": "Transform"},
        {
            "component": "Appearance",
            "kwargs": {
                "renderMode": "ascii_shape",
                "spriteNames": ["SeWallCorner"],
                "spriteShapes": [shapes.BRICK_WALL_SE_CORNER],
                "palettes": [shapes.BRICK_WALL_PALETTE],
                "noRotates": [False]
            }
        },
    ]
}

SW_WALL_CORNER = {
    "name": "sw_wall_corner",
    "components": [
        {
            "component": "StateManager",
            "kwargs": {
                "initialState": "sw_wall_corner",
                "stateConfigs": [{
                    "state": "sw_wall_corner",
                    "layer": "superOverlay",
                    "sprite": "SwWallCorner",
                }],
            }
        },
        {"component": "Transform"},
        {
            "component": "Appearance",
            "kwargs": {
                "renderMode": "ascii_shape",
                "spriteNames": ["SwWallCorner"],
                "spriteShapes": [shapes.BRICK_WALL_SW_CORNER],
                "palettes": [shapes.BRICK_WALL_PALETTE],
                "noRotates": [False]
            }
        },
    ]
}

NW_INNER_WALL_CORNER = {
    "name": "nw_inner_wall_corner",
    "components": [
        {
            "component": "StateManager",
            "kwargs": {
                "initialState": "nw_inner_wall_corner",
                "stateConfigs": [{
                    "state": "nw_inner_wall_corner",
                    "layer": "superOverlay",
                    "sprite": "NwInnerWallCorner",
                }],
            }
        },
        {"component": "Transform"},
        {
            "component": "Appearance",
            "kwargs": {
                "renderMode": "ascii_shape",
                "spriteNames": ["NwInnerWallCorner"],
                "spriteShapes": [shapes.BRICK_WALL_INNER_NW_CORNER],
                "palettes": [shapes.BRICK_WALL_PALETTE],
                "noRotates": [False]
            }
        },
    ]
}

NE_INNER_WALL_CORNER = {
    "name": "ne_inner_wall_corner",
    "components": [
        {
            "component": "StateManager",
            "kwargs": {
                "initialState": "ne_inner_wall_corner",
                "stateConfigs": [{
                    "state": "ne_inner_wall_corner",
                    "layer": "superOverlay",
                    "sprite": "NeInnerWallCorner",
                }],
            }
        },
        {"component": "Transform"},
        {
            "component": "Appearance",
            "kwargs": {
                "renderMode": "ascii_shape",
                "spriteNames": ["NeInnerWallCorner"],
                "spriteShapes": [shapes.BRICK_WALL_INNER_NE_CORNER],
                "palettes": [{"b": (166, 162, 139, 255),
                              "c": (110, 108, 92, 255),
                              "o": (78, 78, 78, 255)}],
                "noRotates": [False]
            }
        },
    ]
}

SE_INNER_WALL_CORNER = {
    "name": "se_inner_wall_corner",
    "components": [
        {
            "component": "StateManager",
            "kwargs": {
                "initialState": "se_inner_wall_corner",
                "stateConfigs": [{
                    "state": "se_inner_wall_corner",
                    "layer": "superOverlay",
                    "sprite": "SeInnerWallCorner",
                }],
            }
        },
        {"component": "Transform"},
        {
            "component": "Appearance",
            "kwargs": {
                "renderMode": "ascii_shape",
                "spriteNames": ["SeInnerWallCorner"],
                "spriteShapes": [shapes.BRICK_WALL_INNER_SE_CORNER],
                "palettes": [{"b": (166, 162, 139, 255),
                              "c": (110, 108, 92, 255),
                              "o": (78, 78, 78, 255)}],
                "noRotates": [False]
            }
        },
    ]
}

SW_INNER_WALL_CORNER = {
    "name": "sw_inner_wall_corner",
    "components": [
        {
            "component": "StateManager",
            "kwargs": {
                "initialState": "sw_inner_wall_corner",
                "stateConfigs": [{
                    "state": "sw_inner_wall_corner",
                    "layer": "superOverlay",
                    "sprite": "SwInnerWallCorner",
                }],
            }
        },
        {"component": "Transform"},
        {
            "component": "Appearance",
            "kwargs": {
                "renderMode": "ascii_shape",
                "spriteNames": ["SwInnerWallCorner"],
                "spriteShapes": [shapes.BRICK_WALL_INNER_SW_CORNER],
                "palettes": [shapes.BRICK_WALL_PALETTE],
                "noRotates": [False]
            }
        },
    ]
}

WALL_NORTH = {
    "name": "wall_north",
    "components": [
        {
            "component": "StateManager",
            "kwargs": {
                "initialState": "wall_north",
                "stateConfigs": [{
                    "state": "wall_north",
                    "layer": "superOverlay",
                    "sprite": "WallNorth",
                }],
            }
        },
        {"component": "Transform"},
        {
            "component": "Appearance",
            "kwargs": {
                "renderMode": "ascii_shape",
                "spriteNames": ["WallNorth"],
                "spriteShapes": [shapes.BRICK_WALL_NORTH],
                "palettes": [shapes.BRICK_WALL_PALETTE],
                "noRotates": [False]
            }
        },
    ]
}

WALL_EAST = {
    "name": "wall_east",
    "components": [
        {
            "component": "StateManager",
            "kwargs": {
                "initialState": "wall_east",
                "stateConfigs": [{
                    "state": "wall_east",
                    "layer": "superOverlay",
                    "sprite": "WallEast",
                }],
            }
        },
        {"component": "Transform"},
        {
            "component": "Appearance",
            "kwargs": {
                "renderMode": "ascii_shape",
                "spriteNames": ["WallEast"],
                "spriteShapes": [shapes.BRICK_WALL_EAST],
                "palettes": [shapes.BRICK_WALL_PALETTE],
                "noRotates": [False]
            }
        },
    ]
}

WALL_SOUTH = {
    "name": "wall_south",
    "components": [
        {
            "component": "StateManager",
            "kwargs": {
                "initialState": "wall_south",
                "stateConfigs": [{
                    "state": "wall_south",
                    "layer": "superOverlay",
                    "sprite": "WallSouth",
                }],
            }
        },
        {"component": "Transform"},
        {
            "component": "Appearance",
            "kwargs": {
                "renderMode": "ascii_shape",
                "spriteNames": ["WallSouth"],
                "spriteShapes": [shapes.BRICK_WALL_SOUTH],
                "palettes": [shapes.BRICK_WALL_PALETTE],
                "noRotates": [False]
            }
        },
    ]
}


WALL_WEST = {
    "name": "wall_west",
    "components": [
        {
            "component": "StateManager",
            "kwargs": {
                "initialState": "wall_west",
                "stateConfigs": [{
                    "state": "wall_west",
                    "layer": "superOverlay",
                    "sprite": "WallWest",
                }],
            }
        },
        {"component": "Transform"},
        {
            "component": "Appearance",
            "kwargs": {
                "renderMode": "ascii_shape",
                "spriteNames": ["WallWest"],
                "spriteShapes": [shapes.BRICK_WALL_WEST],
                "palettes": [shapes.BRICK_WALL_PALETTE],
                "noRotates": [False]
            }
        },
    ]
}

GROUND = {
    "name": "ground",
    "components": [
        {
            "component": "StateManager",
            "kwargs": {
                "initialState": "ground",
                "stateConfigs": [{
                    "state": "ground",
                    "layer": "background",
                    "sprite": "groundSprite",
                }],
            }
        },
        {"component": "Transform"},
        {
            "component": "Appearance",
            "kwargs": {
                "renderMode": "ascii_shape",
                "spriteNames": ["groundSprite"],
                "spriteShapes": [shapes.DIRT_PATTERN],
                "palettes": [{"X": (155, 118, 83, 255),
                              "x": (149, 114, 80, 255),}],
                "noRotates": [True]
            }
        },
    ]
}


def get_fruit_tree_palette(fruit_type):
  """Return a palette with the correct colored fruit."""
  palette = copy.deepcopy(shapes.TREE_PALETTE)
  if fruit_type == "ripe_apple":
    palette["Z"] = (255, 0, 0, 255)
  elif fruit_type == "ripe_banana":
    palette["Z"] = (255, 255, 53, 255)
  elif fruit_type == "unripe_apple":
    palette["Z"] = (128, 0, 0, 255)
  elif fruit_type == "unripe_banana":
    palette["Z"] = (153, 153, 0, 255)
  return palette

TREE = {
    "name": "tree",
    "components": [
        {
            "component": "StateManager",
            "kwargs": {
                "initialState": "treeWait",
                "stateConfigs": [
                    {"state": "treeWait"},
                    {
                        "state": "appleTree",
                        "layer": "lowerPhysical",
                        "sprite": "appleTreeSprite",
                    },
                    {
                        "state": "appleShrub",
                        "layer": "lowerPhysical",
                        "sprite": "appleShrubSprite",
                    },
                    {
                        "state": "bananaTree",
                        "layer": "lowerPhysical",
                        "sprite": "bananaTreeSprite",
                    },
                    {
                        "state": "bananaShrub",
                        "layer": "lowerPhysical",
                        "sprite": "bananaShrubSprite",
                    },
                ],
            }
        },
        {"component": "Transform"},
        {
            "component": "Appearance",
            "kwargs": {
                "renderMode": "ascii_shape",
                "spriteNames": ["appleTreeSprite",
                                "bananaTreeSprite",
                                "appleShrubSprite",
                                "bananaShrubSprite",],
                "spriteShapes": [shapes.EMPTY_TREE,
                                 shapes.EMPTY_TREE,
                                 shapes.EMPTY_SHRUB,
                                 shapes.EMPTY_SHRUB],
                "palettes": [get_fruit_tree_palette("ripe_apple"),
                             get_fruit_tree_palette("ripe_banana"),
                             get_fruit_tree_palette("ripe_apple"),
                             get_fruit_tree_palette("ripe_banana")],
                "noRotates": [True] * 4,
            }
        },
        {
            "component": "TreeType",
            "kwargs": {
                "probabilities": {
                    "empty": 0.8,
                    "appleTree": 0.15,
                    "appleShrub": 0.01,
                    "bananaTree": 0.03,
                    # lower probability that child can pick up what they like
                    "bananaShrub": 0.01,
                }
            }
        },
    ]
}

FRUIT = {
    "name": "fruit",
    "components": [
        {
            "component": "StateManager",
            "kwargs": {
                "initialState": "fruitWait",
                "stateConfigs": [
                    {"state": "fruitWait"},
                    {"state": "fruitEaten"},
                    {
                        "state": "applePicked",
                        "layer": "overlay",
                        "sprite": "appleSprite",
                    },
                    {
                        "state": "appleInTree",
                        "layer": "upperPhysical",
                        "sprite": "appleInTreeSprite",
                    },
                    {
                        "state": "appleInShrub",
                        "layer": "upperPhysical",
                        "sprite": "appleInShrubSprite",
                    },
                    {
                        "state": "bananaPicked",
                        "layer": "overlay",
                        "sprite": "bananaSprite",
                    },
                    {
                        "state": "bananaInTree",
                        "layer": "upperPhysical",
                        "sprite": "bananaInTreeSprite",
                    },
                    {
                        "state": "bananaInShrub",
                        "layer": "upperPhysical",
                        "sprite": "bananaInShrubSprite",
                    },
                ],
            }
        },
        {"component": "Transform"},
        {
            "component": "Appearance",
            "kwargs": {
                "renderMode": "ascii_shape",
                "spriteNames": ["appleSprite",
                                "appleInTreeSprite",
                                "appleInShrubSprite",
                                "bananaSprite",
                                "bananaInTreeSprite",
                                "bananaInShrubSprite",
                                ],
                "spriteShapes": [shapes.HD_APPLE,
                                 shapes.FRUIT_IN_TREE,
                                 shapes.FRUIT_IN_SHRUB,
                                 shapes.HD_APPLE,
                                 shapes.FRUIT_IN_TREE,
                                 shapes.FRUIT_IN_SHRUB,
                                 ],
                "palettes": [shapes.get_palette((255, 0, 0, 255)),
                             get_fruit_tree_palette("ripe_apple"),
                             get_fruit_tree_palette("ripe_apple"),
                             shapes.get_palette((255, 255, 53, 255)),
                             get_fruit_tree_palette("ripe_banana"),
                             get_fruit_tree_palette("ripe_banana"),
                             ],
                "noRotates": [True] * 6,
            }
        },
        {
            "component": "Graspable",
        },
        {
            "component": "FruitType",
            "kwargs": {"framesTillAppleRespawn": 50}
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
                    "layer": "alternateLogic",
                    "groups": ["spawnPoints"]
                }],
            }
        },
        {"component": "Transform"},
    ]
}


# PREFABS is a dictionary mapping names to template game objects that can
# be cloned and placed in multiple locations accoring to an ascii map.
PREFABS = {
    "nw_wall_corner": NW_WALL_CORNER,
    "ne_wall_corner": NE_WALL_CORNER,
    "se_wall_corner": SE_WALL_CORNER,
    "sw_wall_corner": SW_WALL_CORNER,
    "nw_inner_wall_corner": NW_INNER_WALL_CORNER,
    "ne_inner_wall_corner": NE_INNER_WALL_CORNER,
    "se_inner_wall_corner": SE_INNER_WALL_CORNER,
    "sw_inner_wall_corner": SW_INNER_WALL_CORNER,
    "wall_north": WALL_NORTH,
    "wall_east": WALL_EAST,
    "wall_south": WALL_SOUTH,
    "wall_west": WALL_WEST,
    # non-wall prefabs
    "spawn_point": SPAWN_POINT,
    "ground": GROUND,
    "tree": TREE,
    "fruit": FRUIT,
}

# Primitive action components.
# pylint: disable=bad-whitespace
# pyformat: disable
NOOP       = {"move": 0, "turn":  0, "eat": 0, "grasp": 0}
FORWARD    = {"move": 1, "turn":  0, "eat": 0, "grasp": 0}
STEP_RIGHT = {"move": 2, "turn":  0, "eat": 0, "grasp": 0}
BACKWARD   = {"move": 3, "turn":  0, "eat": 0, "grasp": 0}
STEP_LEFT  = {"move": 4, "turn":  0, "eat": 0, "grasp": 0}
TURN_LEFT  = {"move": 0, "turn": -1, "eat": 0, "grasp": 0}
TURN_RIGHT = {"move": 0, "turn":  1, "eat": 0, "grasp": 0}
EAT     = {"move": 0, "turn":  0, "eat": 1, "grasp": 0}
GRASP      = {"move": 0, "turn":  0, "eat": 0, "grasp": 1}

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
    EAT,
    GRASP,
)


def create_scene():
  """Creates the global scene."""
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
          {"component": "Transform"},
      ]
  }

  return scene


def _create_avatar_object(player_idx: int, is_child: bool) -> Dict[str, Any]:
  """Create an avatar object."""
  # Lua is 1-indexed.
  lua_index = player_idx + 1
  live_state_name = "player{}".format(lua_index)
  avatar_sprite_name = "avatarSprite{}".format(lua_index)

  if is_child:
    color_palette = shapes.get_palette(colors.palette[3])
    sprite = shapes.CUTE_AVATAR_CHILD
    can_grasp_tree = False
    # child gets reward for eating bananas
    apple_reward = 0
    banana_reward = 1
    grasp_success_probability = 0.3
    # child sees trees as shrubs
    custom_sprite_map = {"appleTreeSprite": "appleShrubSprite",
                         "appleInTreeSprite": "appleInShrubSprite",
                         "bananaTreeSprite": "bananaShrubSprite",
                         "bananaInTreeSprite": "bananaInShrubSprite"}
  else:
    color_palette = shapes.get_palette(colors.palette[0])
    sprite = shapes.CUTE_AVATAR
    can_grasp_tree = True
    apple_reward = 1
    banana_reward = 1
    grasp_success_probability = 1
    # parent sees bananas as apples
    custom_sprite_map = {"bananaTreeSprite": "appleTreeSprite",
                         "bananaShrubSprite": "appleShrubSprite",
                         "bananaInTreeSprite": "appleInTreeSprite",
                         "bananaInShrubSprite": "appleInShrubSprite",
                         "bananaSprite": "appleSprite",}

  avatar_object = {
      "name": "avatar",
      "components": [
          {
              "component": "StateManager",
              "kwargs": {
                  "initialState":
                      live_state_name,
                  "stateConfigs": [
                      # Initial player state.
                      {
                          "state": live_state_name,
                          "layer": "superOverlay",
                          "sprite": avatar_sprite_name,
                          "contact": "avatar",
                          "groups": ["players"]
                      },
                      # Player wait type for times when they are zapped out.
                      {
                          "state": "playerWait",
                          "groups": ["playerWaits"]
                      },
                  ]
              }
          },
          {
              "component": "Transform"
          },
          {
              "component": "Appearance",
              "kwargs": {
                  "renderMode": "ascii_shape",
                  "spriteNames": [avatar_sprite_name],
                  "spriteShapes": [sprite],
                  "palettes": [color_palette],
                  "noRotates": [True]
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
                                  "eat",
                                  "grasp"],
                  "actionSpec": {
                      "move": {"default": 0, "min": 0, "max": len(_COMPASS)},
                      "turn": {"default": 0, "min": -1, "max": 1},
                      "eat": {"default": 0, "min": 0, "max": 1},
                      "grasp": {"default": 0, "min": 0, "max": 1},
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
              "component": "Role",
              "kwargs": {
                  "isChild": is_child,
              }
          },
          {
              "component": "Eating",
              "kwargs": {
                  "bananaReward": banana_reward,
                  "appleReward": apple_reward,
              }
          },
          {
              "component": "PlayerGrasp",
              "kwargs": {
                  "shape": shapes.GRASP_SHAPE,
                  "palette": color_palette,
                  "canGraspTree": can_grasp_tree,
                  "graspSuccessProbability": grasp_success_probability,
                  "attentiveParentPseudoreward": 0.0,
                  "droppingParentPseudoreward": 0.0,
                  "tryingChildPseudoreward": 0.0,
                  "tryingChildBananaPseudoreward": 0.0,
              }
          },
          {
              "component": "AvatarRespawn",
              "kwargs": {
                  "framesTillRespawn": 100,
              }
          },
          {
              "component": "Hunger",
              "kwargs": {
                  "framesTillHungry": 200,
              }
          },
          {
              "component": "HungerObserver",
              "kwargs": {
                  "needComponent": "Hunger",
              },
          },
      ]
  }
  if _ENABLE_DEBUG_OBSERVATIONS:
    avatar_object["components"].append({
        "component": "LocationObserver",
        "kwargs": {"objectIsAvatar": True, "alsoReportOrientation": True},
    })

  return avatar_object


def _build_child_objects(player_idx: int):
  """Build child avatar objects."""
  avatar_object = _create_avatar_object(
      player_idx, is_child=True)
  game_objects = []
  game_objects.append(avatar_object)
  return game_objects


def _build_parent_objects(player_idx: int):
  """Build parent avatar objects."""
  avatar_object = _create_avatar_object(
      player_idx, is_child=False)
  game_objects = []
  game_objects.append(avatar_object)
  return game_objects


def create_avatar_objects(roles: Sequence[str]):
  """Returns list of avatar objects of length 'num_players'."""
  avatar_objects_and_helpers = []
  for player_idx, role in enumerate(roles):
    if role == "child":
      avatar_objects_and_helpers.extend(_build_child_objects(player_idx))
    elif role == "parent":
      avatar_objects_and_helpers.extend(_build_parent_objects(player_idx))
    elif role == "default":
      # Parents and children are alternating, parents in even positions.
      if player_idx % 2 == 0:
        avatar_objects_and_helpers.extend(_build_parent_objects(player_idx))
      else:
        avatar_objects_and_helpers.extend(_build_child_objects(player_idx))
    else:
      raise ValueError(f"Unrecognized role: {role}")

  return avatar_objects_and_helpers


def get_config():
  """Default configuration for the daycare substrate."""
  config = configdict.ConfigDict()

  # Specify the number of players to particate in each episode (optional).
  config.recommended_num_players = 2

  # Action set configuration.
  config.action_set = ACTION_SET
  # Observation format configuration.
  config.individual_observation_names = [
      "RGB",
      "HUNGER",
  ]
  config.global_observation_names = [
      "WORLD.RGB",
  ]

  # The specs of the environment (from a single-agent perspective).
  config.action_spec = specs.action(len(ACTION_SET))
  config.timestep_spec = specs.timestep({
      "RGB": specs.OBSERVATION["RGB"],
      "HUNGER": specs.float64(),
      # Debug only (do not use the following observations in policies).
      "WORLD.RGB": specs.rgb(104, 160,),
  })

  # The roles assigned to each player.
  config.valid_roles = frozenset({"child", "parent"})
  config.default_player_roles = ("child", "parent")
  return config


def build(
    roles: Sequence[str],
    config: configdict.ConfigDict,
) -> Mapping[str, Any]:
  """Build this substrate given player roles."""
  del config
  substrate_definition = dict(
      levelName="daycare",
      levelDirectory="meltingpot/lua/levels",
      numPlayers=len(roles),
      maxEpisodeLengthFrames=1000,
      spriteSize=8,
      topology="BOUNDED",  # Choose from ["BOUNDED", "TORUS"],
      simulation=dict(
          map=ASCII_MAP,
          gameObjects=create_avatar_objects(roles),
          scene=create_scene(),
          prefabs=PREFABS,
          charPrefabMap=CHAR_PREFAB_MAP,
      ),
  )
  return substrate_definition
