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
"""Configuration for the Fruit Market substrate.

This substrate is used to study the dynamics of trade and bargaining of goods
that have different value to different players.

The substrate consists of an open space where two types of trees exist: apple
trees and banana trees. Trees can be harvested by players by stepping on their
location and wait until they harvest the fruit from the tree. A harvested fruit
(apple or banana) goes into a player's inventory. Players can carry any number
of apples or bananas. Harvested fruit can be consumed for reward. Players have
two actions to consume fruit of the two types from their inventory.

Players can be of two types: apple farmer & banana farmer. Apple farmers have a
higher probability of harvesting from apple trees than banana trees, but receive
more reward for consuming bananas. Banana farmers are the opposite.

Players have a hunger meter which can be replenished by consuming a fruit.
Players have an action to consume an apple from their inventory, and another to
consume a banana. If the hunger meter reaches zero the player pays a
substantial cost in stamina.

Crossing water also imposes a cost in stamina.

Players also have trading actions of the form "I offer X apples for Y bananas"
and the converse "I offer Z bananas for W apples". When players are within a
trading radius of each other and have corresponding offers (`X = W` and `Y = Z`)
and enough fruit in their inventories to satisfy it, the trade occurs and the
appropriate number of apples and bananas are exchanged and placed in their
inventories.
"""

import copy
from typing import Any, Dict, Generator, Mapping, Sequence

from meltingpot.utils.substrates import game_object_utils
from meltingpot.utils.substrates import shapes
from ml_collections import config_dict as configdict

PrefabConfig = game_object_utils.PrefabConfig

# Warning: setting `_ENABLE_DEBUG_OBSERVATIONS = True` may cause slowdown.
_ENABLE_DEBUG_OBSERVATIONS = False

MAX_OFFER_QUANTITY = 3

_COMPASS = ["N", "E", "S", "W"]
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
                    "layer": "upperPhysical",
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
                "spriteShapes": [shapes.FENCE_NW_CORNER],
                "palettes": [shapes.FENCE_PALETTE],
                "noRotates": [False]
            }
        },
        {"component": "BeamBlocker", "kwargs": {"beamType": "hold"}},
        {"component": "BeamBlocker", "kwargs": {"beamType": "shove"}},
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
                    "layer": "upperPhysical",
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
                "spriteShapes": [shapes.FENCE_NE_CORNER],
                "palettes": [shapes.FENCE_PALETTE],
                "noRotates": [False]
            }
        },
        {"component": "BeamBlocker", "kwargs": {"beamType": "hold"}},
        {"component": "BeamBlocker", "kwargs": {"beamType": "shove"}},
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
                    "layer": "upperPhysical",
                    "sprite": "ne_inner_wall_corner",
                }],
            }
        },
        {"component": "Transform"},
        {
            "component": "Appearance",
            "kwargs": {
                "renderMode": "ascii_shape",
                "spriteNames": ["ne_inner_wall_corner"],
                "spriteShapes": [shapes.FENCE_INNER_NE_CORNER],
                "palettes": [shapes.FENCE_PALETTE],
                "noRotates": [False]
            }
        },
        {"component": "BeamBlocker", "kwargs": {"beamType": "hold"}},
        {"component": "BeamBlocker", "kwargs": {"beamType": "shove"}},
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
                    "layer": "upperPhysical",
                    "sprite": "nw_inner_wall_corner",
                }],
            }
        },
        {"component": "Transform"},
        {
            "component": "Appearance",
            "kwargs": {
                "renderMode": "ascii_shape",
                "spriteNames": ["nw_inner_wall_corner"],
                "spriteShapes": [shapes.FENCE_INNER_NW_CORNER],
                "palettes": [shapes.FENCE_PALETTE],
                "noRotates": [False]
            }
        },
        {"component": "BeamBlocker", "kwargs": {"beamType": "hold"}},
        {"component": "BeamBlocker", "kwargs": {"beamType": "shove"}},
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
                    "layer": "upperPhysical",
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
                "spriteShapes": [shapes.FENCE_SE_CORNER],
                "palettes": [shapes.FENCE_PALETTE],
                "noRotates": [False]
            }
        },
        {"component": "BeamBlocker", "kwargs": {"beamType": "hold"}},
        {"component": "BeamBlocker", "kwargs": {"beamType": "shove"}},
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
                    "layer": "upperPhysical",
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
                "spriteShapes": [shapes.FENCE_SW_CORNER],
                "palettes": [shapes.FENCE_PALETTE],
                "noRotates": [False]
            }
        },
        {"component": "BeamBlocker", "kwargs": {"beamType": "hold"}},
        {"component": "BeamBlocker", "kwargs": {"beamType": "shove"}},
    ]
}

WALL_SHADOW_SW = {
    "name": "wall_shadow_sw",
    "components": [
        {
            "component": "StateManager",
            "kwargs": {
                "initialState": "wall_shadow_sw",
                "stateConfigs": [{
                    "state": "wall_shadow_sw",
                    "layer": "upperPhysical",
                    "sprite": "wall_shadow_sw",
                }],
            }
        },
        {"component": "Transform"},
        {
            "component": "Appearance",
            "kwargs": {
                "renderMode": "ascii_shape",
                "spriteNames": ["wall_shadow_sw"],
                "spriteShapes": [shapes.FENCE_SHADOW_SW],
                "palettes": [shapes.FENCE_PALETTE],
                "noRotates": [False]
            }
        },
        {"component": "BeamBlocker", "kwargs": {"beamType": "hold"}},
        {"component": "BeamBlocker", "kwargs": {"beamType": "shove"}},
    ]
}

WALL_SHADOW_S = {
    "name": "wall_shadow_s",
    "components": [
        {
            "component": "StateManager",
            "kwargs": {
                "initialState": "wall_shadow_s",
                "stateConfigs": [{
                    "state": "wall_shadow_s",
                    "layer": "upperPhysical",
                    "sprite": "wall_shadow_s",
                }],
            }
        },
        {"component": "Transform"},
        {
            "component": "Appearance",
            "kwargs": {
                "renderMode": "ascii_shape",
                "spriteNames": ["wall_shadow_s"],
                "spriteShapes": [shapes.FENCE_SHADOW_S],
                "palettes": [shapes.FENCE_PALETTE],
                "noRotates": [False]
            }
        },
        {"component": "BeamBlocker", "kwargs": {"beamType": "hold"}},
        {"component": "BeamBlocker", "kwargs": {"beamType": "shove"}},
    ]
}

WALL_SHADOW_SE = {
    "name": "wall_shadow_se",
    "components": [
        {
            "component": "StateManager",
            "kwargs": {
                "initialState": "wall_shadow_se",
                "stateConfigs": [{
                    "state": "wall_shadow_se",
                    "layer": "upperPhysical",
                    "sprite": "wall_shadow_se",
                }],
            }
        },
        {"component": "Transform"},
        {
            "component": "Appearance",
            "kwargs": {
                "renderMode": "ascii_shape",
                "spriteNames": ["wall_shadow_se"],
                "spriteShapes": [shapes.FENCE_SHADOW_SE],
                "palettes": [shapes.FENCE_PALETTE],
                "noRotates": [False]
            }
        },
        {"component": "BeamBlocker", "kwargs": {"beamType": "hold"}},
        {"component": "BeamBlocker", "kwargs": {"beamType": "shove"}},
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
                    "layer": "upperPhysical",
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
                "spriteShapes": [shapes.FENCE_N],
                "palettes": [shapes.FENCE_PALETTE],
                "noRotates": [False]
            }
        },
        {"component": "BeamBlocker", "kwargs": {"beamType": "hold"}},
        {"component": "BeamBlocker", "kwargs": {"beamType": "shove"}},
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
                    "layer": "upperPhysical",
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
                "spriteShapes": [shapes.FENCE_E],
                "palettes": [shapes.FENCE_PALETTE],
                "noRotates": [False]
            }
        },
        {"component": "BeamBlocker", "kwargs": {"beamType": "hold"}},
        {"component": "BeamBlocker", "kwargs": {"beamType": "shove"}},
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
                    "layer": "upperPhysical",
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
                "spriteShapes": [shapes.FENCE_S],
                "palettes": [shapes.FENCE_PALETTE],
                "noRotates": [False]
            }
        },
        {"component": "BeamBlocker", "kwargs": {"beamType": "hold"}},
        {"component": "BeamBlocker", "kwargs": {"beamType": "shove"}},
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
                    "layer": "upperPhysical",
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
                "spriteShapes": [shapes.FENCE_W],
                "palettes": [shapes.FENCE_PALETTE],
                "noRotates": [False]
            }
        },
        {"component": "BeamBlocker", "kwargs": {"beamType": "hold"}},
        {"component": "BeamBlocker", "kwargs": {"beamType": "shove"}},
    ]
}

STONE_WALL = {
    "name": "stone_wall",
    "components": [
        {
            "component": "StateManager",
            "kwargs": {
                "initialState": "stoneWall",
                "stateConfigs": [{
                    "state": "stoneWall",
                    "layer": "upperPhysical",
                    "sprite": "StoneWall",
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
                "spriteNames": ["StoneWall"],
                "spriteShapes": [shapes.WALL],
                "palettes": [{"*": (95, 95, 95, 255),
                              "&": (100, 100, 100, 255),
                              "@": (109, 109, 109, 255),
                              "#": (152, 152, 152, 255)}],
                "noRotates": [False]
            }
        },
        {"component": "BeamBlocker", "kwargs": {"beamType": "hold"}},
        {"component": "BeamBlocker", "kwargs": {"beamType": "shove"}},
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
                      "@": (52, 193, 209, 255),
                      "*": (34, 166, 181, 255),
                      "o": (32, 155, 168, 255),
                      "~": (31, 148, 161, 255)}] * 4,
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
          {
              "component": "TraversalCost",
              "kwargs": {
                  "penaltyAmount": 0,  # No reward cost from crossing water.
                  "alsoReduceStamina": True,  # Crossing water depletes stamina.
                  "staminaPenaltyAmount": 1,  # Stamina lost per step on water.
              }
          },
      ]
  }
  return water


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
                "palettes": [{"X": (207, 199, 184, 255),
                              "x": (199, 192, 177, 255),}],
                "noRotates": [True]
            }
        },
    ]
}


def get_fruit_tree_palette(fruit_type):
  """Return a palette with the correct colored fruit."""
  apple_palette = copy.deepcopy(shapes.APPLE_TREE_PALETTE)
  banana_palette = copy.deepcopy(shapes.BANANA_TREE_PALETTE)
  if fruit_type == "ripe_apple":
    apple_palette["o"] = (199, 33, 8, 255)
    return apple_palette
  elif fruit_type == "ripe_banana":
    banana_palette["o"] = (222, 222, 13, 255)
    return banana_palette
  elif fruit_type == "unripe_apple":
    apple_palette["o"] = (124, 186, 58, 255)
    return apple_palette
  elif fruit_type == "unripe_banana":
    banana_palette["o"] = (37, 115, 45, 255)
    return banana_palette


def get_potential_tree(probability_empty: float = 0.9,
                       probability_apple: float = 0.05,
                       probability_banana: float = 0.05) -> PrefabConfig:
  """Return a prefab for a potential tree."""
  assert probability_empty + probability_apple + probability_banana == 1.0, (
      "Probabilities must sum to 1.0.")
  spawn_probabilities = {"empty": probability_empty,
                         "apple": probability_apple,
                         "banana": probability_banana}
  prefab = {
      "name": "potential_tree",
      "components": [
          {
              "component": "StateManager",
              "kwargs": {
                  "initialState": "treeWait",
                  "stateConfigs": [
                      {"state": "treeWait"},
                      {
                          "state": "appleTreeHarvestable",
                          "layer": "lowerPhysical",
                          "sprite": "appleTreeHarvestableSprite",
                      },
                      {
                          "state": "bananaTreeHarvestable",
                          "layer": "lowerPhysical",
                          "sprite": "bananaTreeHarvestableSprite",
                      },
                      {
                          "state": "appleTreeUnripe",
                          "layer": "lowerPhysical",
                          "sprite": "appleTreeUnripeSprite",
                      },
                      {
                          "state": "bananaTreeUnripe",
                          "layer": "lowerPhysical",
                          "sprite": "bananaTreeUnripeSprite",
                      },
                  ],
              }
          },
          {"component": "Transform"},
          {
              "component": "Appearance",
              "kwargs": {
                  "renderMode": "ascii_shape",
                  "spriteNames": ["appleTreeHarvestableSprite",
                                  "bananaTreeHarvestableSprite",
                                  "appleTreeUnripeSprite",
                                  "bananaTreeUnripeSprite"],
                  "spriteShapes": [shapes.APPLE_TREE_STOUT,
                                   shapes.BANANA_TREE,
                                   shapes.APPLE_TREE_STOUT,
                                   shapes.BANANA_TREE],
                  "palettes": [get_fruit_tree_palette("ripe_apple"),
                               get_fruit_tree_palette("ripe_banana"),
                               get_fruit_tree_palette("unripe_apple"),
                               get_fruit_tree_palette("unripe_banana")],
                  "noRotates": [True,
                                True,
                                True,
                                True]
              }
          },
          {
              "component": "FruitType",
              "kwargs": {
                  "probabilities": spawn_probabilities,
              }
          },
          {
              "component": "Harvestable",
              "kwargs": {
                  "regrowthTime": 50,
              }
          },
          {
              "component": "PreventStaminaRecoveryHere",
          },
      ]
  }
  return prefab


# Primitive action components.
# pylint: disable=bad-whitespace
# pyformat: disable
NOOP        = {"move": 0, "turn":  0, "eat_apple": 0, "eat_banana": 0, "offer_apple": 0, "offer_banana": 0, "offer_cancel": 0, "hold": 0, "shove":  0}
FORWARD     = {"move": 1, "turn":  0, "eat_apple": 0, "eat_banana": 0, "offer_apple": 0, "offer_banana": 0, "offer_cancel": 0, "hold": 0, "shove":  0}
STEP_RIGHT  = {"move": 2, "turn":  0, "eat_apple": 0, "eat_banana": 0, "offer_apple": 0, "offer_banana": 0, "offer_cancel": 0, "hold": 0, "shove":  0}
BACKWARD    = {"move": 3, "turn":  0, "eat_apple": 0, "eat_banana": 0, "offer_apple": 0, "offer_banana": 0, "offer_cancel": 0, "hold": 0, "shove":  0}
STEP_LEFT   = {"move": 4, "turn":  0, "eat_apple": 0, "eat_banana": 0, "offer_apple": 0, "offer_banana": 0, "offer_cancel": 0, "hold": 0, "shove":  0}
TURN_LEFT   = {"move": 0, "turn": -1, "eat_apple": 0, "eat_banana": 0, "offer_apple": 0, "offer_banana": 0, "offer_cancel": 0, "hold": 0, "shove":  0}
TURN_RIGHT  = {"move": 0, "turn":  1, "eat_apple": 0, "eat_banana": 0, "offer_apple": 0, "offer_banana": 0, "offer_cancel": 0, "hold": 0, "shove":  0}
EAT_APPLE   = {"move": 0, "turn":  0, "eat_apple": 1, "eat_banana": 0, "offer_apple": 0, "offer_banana": 0, "offer_cancel": 0, "hold": 0, "shove":  0}
EAT_BANANA  = {"move": 0, "turn":  0, "eat_apple": 0, "eat_banana": 1, "offer_apple": 0, "offer_banana": 0, "offer_cancel": 0, "hold": 0, "shove":  0}
HOLD        = {"move": 0, "turn":  0, "eat_apple": 0, "eat_banana": 0, "offer_apple": 0, "offer_banana": 0, "offer_cancel": 0, "hold": 1, "shove":  0}
# Notice that SHOVE includes both `hold` and `shove` parts.
SHOVE       = {"move": 0, "turn":  0, "eat_apple": 0, "eat_banana": 0, "offer_apple": 0, "offer_banana": 0, "offer_cancel": 0, "hold": 1, "shove":  1}
PULL        = {"move": 0, "turn":  0, "eat_apple": 0, "eat_banana": 0, "offer_apple": 0, "offer_banana": 0, "offer_cancel": 0, "hold": 1, "shove": -1}
# pyformat: enable
# pylint: enable=bad-whitespace

offer_actions = []
# Add the cancel action
cancel_action = {"move": 0, "turn": 0, "eat_apple": 0, "eat_banana": 0,
                 "offer_apple": 0, "offer_banana": 0, "offer_cancel": 1,
                 "hold": 0, "shove": 0}
offer_actions.append(cancel_action)

for a in range(-MAX_OFFER_QUANTITY, MAX_OFFER_QUANTITY):
  for b in range(-MAX_OFFER_QUANTITY, MAX_OFFER_QUANTITY):
    offer_action = {"move": 0, "turn": 0, "eat_apple": 0, "eat_banana": 0,
                    "offer_apple": a, "offer_banana": b, "offer_cancel": 0,
                    "hold": 0, "shove": 0}
    if a > 0 and b < 0:
      offer_actions.append(offer_action)
    elif a < 0 and b > 0:
      offer_actions.append(offer_action)

ACTION_SET = (
    NOOP,
    FORWARD,
    BACKWARD,
    STEP_LEFT,
    STEP_RIGHT,
    TURN_LEFT,
    TURN_RIGHT,
    EAT_APPLE,
    EAT_BANANA,
    HOLD,
    SHOVE,
    PULL,
    *offer_actions,
)


def create_prefabs() -> PrefabConfig:
  """Returns the prefabs.

  Prefabs are a dictionary mapping names to template game objects that can
  be cloned and placed in multiple locations accoring to an ascii map.
  """
  prefabs = {
      # wall prefabs
      "nw_wall_corner": NW_WALL_CORNER,
      "nw_inner_wall_corner": NW_INNER_WALL_CORNER,
      "ne_wall_corner": NE_WALL_CORNER,
      "ne_inner_wall_corner": NE_INNER_WALL_CORNER,
      "se_wall_corner": SE_WALL_CORNER,
      "sw_wall_corner": SW_WALL_CORNER,
      "wall_north": WALL_NORTH,
      "wall_east": WALL_EAST,
      "wall_south": WALL_SOUTH,
      "wall_west": WALL_WEST,
      "wall_shadow_sw": WALL_SHADOW_SW,
      "wall_shadow_s": WALL_SHADOW_S,
      "wall_shadow_se": WALL_SHADOW_SE,
      "stone_wall": STONE_WALL,

      # non-wall prefabs
      "spawn_point": SPAWN_POINT,
      "river": get_water(),
      "ground": GROUND,
      "potential_tree": get_potential_tree(),
      "high_probability_tree": get_potential_tree(
          probability_empty=0.1,
          probability_apple=0.45,
          probability_banana=0.45,
      ),
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
              "component": "TradeManager",
          },
      ]
  }
  return scene


def _create_stamina_overlay(player_idx: int,
                            max_stamina_bar_states: int,
                            ) -> Generator[Dict[str, Any], None, None]:
  """Create stamina marker overlay objects."""
  # Lua is 1-indexed.
  lua_idx = player_idx + 1

  stamina_bar_state_configs = [
      # Invisible inactive (dead) overlay type.
      {"state": "staminaBarWait"},
  ]
  stamina_bar_sprite_names = []
  stamina_bar_sprite_shapes = []

  # Each player's stamina bars must be in their own layer so they do not
  # interact/collide with other players' stamina bars.
  stamina_bar_layer = f"superOverlay_{player_idx}"

  # Declare one state per level of the stamina bar.
  for i in range(max_stamina_bar_states):
    sprite_name = f"sprite_for_level_{i}"
    stamina_bar_state_configs.append(
        {"state": f"level_{i}",
         "layer": stamina_bar_layer,
         "sprite": sprite_name})
    stamina_bar_sprite_names.append(sprite_name)
    xs = "\nxxxxxxxx"
    blank_space = xs * 7
    number_of_rs = max(6 - i, 0)
    number_of_ys = i if i < 7 else 12 - i
    number_of_gs = max(i - 6, 0)
    if i >= 13:
      level = blank_space + xs
    else:
      level = (blank_space + "\nx" + "G" * number_of_gs + "Y" * number_of_ys +
               "R" * number_of_rs + "x")
    empty = "\n".join(["x" * 8] * 8)
    # Replace the east/south/west sprites with invisible sprites so the only
    # stamina bar rendered is the one in the direction that the current player
    # is facing.
    stamina_bar_sprite_shapes.append((level, empty, empty, empty))

  # Create a stamina bar for each compass direction. Only the direction the
  # current player is facing is visible.
  for direction in ("N", "E", "S", "W"):
    yield {
        "name": "avatar_stamina_bar",
        "components": [
            {
                "component": "StateManager",
                "kwargs": {
                    "initialState": "staminaBarWait",
                    "stateConfigs": stamina_bar_state_configs
                }
            },
            {
                "component": "Transform",
            },
            {
                "component": "Appearance",
                "kwargs": {
                    "renderMode": "ascii_shape",
                    "spriteNames": stamina_bar_sprite_names,
                    "spriteShapes": stamina_bar_sprite_shapes,
                    "palettes": [{"G": (62, 137, 72, 255),
                                  "Y": (255, 216, 97, 255),
                                  "R": (162, 38, 51, 255),
                                  "x": INVISIBLE,}] * max_stamina_bar_states,
                    "noRotates": [True] * max_stamina_bar_states
                }
            },
            {
                "component": "StaminaBar",
                "kwargs": {
                    "playerIndex": lua_idx,
                    "waitState": "staminaBarWait",
                    "layer": stamina_bar_layer,
                    "direction": direction
                }
            },
        ]
    }


def create_avatar_object(player_idx: int,
                         specialty: str,
                         max_stamina_bar_states: int) -> Dict[str, Any]:
  """Create an avatar object."""
  # Lua is 1-indexed.
  lua_index = player_idx + 1

  source_sprite_self = "Avatar" + str(lua_index)
  grappling_sprite = "AvatarGrappling" + str(lua_index)
  grappled_sprite = "AvatarGrappled" + str(lua_index)

  live_state_name = "player{}".format(lua_index)
  grappling_state_name = f"player{lua_index}_grappling"
  grappled_state_name = f"player{lua_index}_grappled"

  map_specialty_to_sprite_color = {
      "apple": (199, 55, 47),  # apple red
      "banana": (255, 225, 53),  # banana yellow
  }
  avatar_color = map_specialty_to_sprite_color[specialty]

  avatar_palette = shapes.get_palette(avatar_color)
  avatar_palette["P"] = (196, 77, 190, 200)
  avatar_palette["p"] = (184, 72, 178, 150)

  map_specialty_to_complement = {
      "apple": "banana",
      "banana": "apple"
  }
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
                      {"state": grappling_state_name,
                       "layer": "upperPhysical",
                       "sprite": grappling_sprite,
                       "contact": "avatar",
                       "groups": ["players"]},
                      {"state": grappled_state_name,
                       "layer": "upperPhysical",
                       "sprite": grappled_sprite,
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
                  "spriteNames": [source_sprite_self, grappling_sprite,
                                  grappled_sprite],
                  "spriteShapes": [shapes.CUTE_AVATAR,
                                   shapes.CUTE_AVATAR_ARMS_UP,
                                   shapes.MAGIC_GRAPPLED_AVATAR],
                  "palettes": [avatar_palette] * 3,
                  "noRotates": [True] * 3,
              }
          },
          {
              "component": "Avatar",
              "kwargs": {
                  "index": lua_index,
                  "aliveState": live_state_name,
                  "additionalLiveStates": [grappled_state_name,
                                           grappling_state_name],
                  "waitState": "playerWait",
                  "spawnGroup": "spawnPoints",
                  "actionOrder": [
                      # Basic movement actions
                      "move",
                      "turn",
                      # Trade actions
                      "eat_apple",
                      "eat_banana",
                      "offer_apple",
                      "offer_banana",
                      "offer_cancel",
                      # Grappling actions
                      "hold",
                      "shove",
                  ],
                  "actionSpec": {
                      # Basic movement actions
                      "move": {"default": 0, "min": 0, "max": len(_COMPASS)},
                      "turn": {"default": 0, "min": -1, "max": 1},
                      # Trade actions
                      "eat_apple": {"default": 0, "min": 0, "max": 1},
                      "eat_banana": {"default": 0, "min": 0, "max": 1},
                      "offer_apple": {"default": 0, "min": -MAX_OFFER_QUANTITY,
                                      "max": MAX_OFFER_QUANTITY},
                      "offer_banana": {"default": 0, "min": -MAX_OFFER_QUANTITY,
                                       "max": MAX_OFFER_QUANTITY},
                      "offer_cancel": {"default": 0, "min": 0, "max": 1},
                      # Grappling actions
                      "hold": {"default": 0, "min": 0, "max": 1},
                      "shove": {"default": 0, "min": -1, "max": 1},
                  },
                  "view": {
                      "left": 5,
                      "right": 5,
                      "forward": 9,
                      "backward": 1,
                      "centered": False
                  },
              }
          },
          {
              "component": "Inventory",
          },
          {
              "component": "Eating",
          },
          {
              "component": "Specialization",
              "kwargs": {
                  "specialty": specialty,  # "apple" or "banana"
                  "strongAmount": 2,
                  "weakAmount": 2,
                  "strongProbability": 1,
                  "weakProbability": 0.04,
              }
          },
          {
              "component": "Trading",
              "kwargs": {
                  "maxOfferQuantity": 3,  # The highest possible offer.
                  "radius": 4,  # Size of neighborhood where trade is possible.
              }
          },
          {
              "component": "Taste",
              "kwargs": {
                  "mostTastyFruit": map_specialty_to_complement[specialty],
                  "mostTastyReward": 8,
                  "defaultReward": 1,
              }
          },
          {
              "component": "PeriodicNeed",  # The hunger mechanic
              "kwargs": {
                  # Hunger threshold reached after `delay` steps without eating.
                  "delay": 50,
                  # No reward cost of hunger exceeding threshold.
                  "reward": 0,
              },
          },
          {
              "component": "Grappling",
              "kwargs": {
                  "shape": shapes.MAGIC_BEAM,
                  "palette": shapes.MAGIC_BEAM_PALETTE,
                  "liveState": live_state_name,
                  "grappledState": grappled_state_name,
                  "grapplingState": grappling_state_name,
              }
          },
          {
              "component": "ReadyToShootObservation",
              "kwargs": {
                  # In this case READY_TO_SHOOT will be 1 if hold is allowed and
                  # will be 0 if not.
                  "zapperComponent": "Grappling",
              }
          },
          {
              "component": "Stamina",
              "kwargs": {
                  "maxStamina": max_stamina_bar_states,
                  "classConfig": {
                      "name": "player",
                      "greenFreezeTime": 0,
                      "yellowFreezeTime": 2,
                      "redFreezeTime": 6,
                      # `decrementRate` = 0.5 means decrease stamina on every
                      # other costly step. `decrementRate` = 1 means decrease
                      # stamina on every costly step.
                      "decrementRate": 0.5,
                  },
                  "amountInvisible": 6,
                  "amountGreen": 6,
                  "amountYellow": 6,
                  "amountRed": 1,
                  "costlyActions": ["move"],
              }
          },
          {
              "component": "StaminaModulatedByNeed",
              "kwargs": {
                  # Reduce stamina by `lossPerStepBeyondThreshold` per timestep
                  # after hunger exceeds its threshold.
                  "lossPerStepBeyondThreshold": 1,
              }
          },
          {
              "component": "StaminaObservation",
              "kwargs": {
                  "staminaComponent": "Stamina",
              }
          },
          {
              "component": "InventoryObserver",
          },
          {
              "component": "MyOfferObserver",
          },
          {
              "component": "AllOffersObserver",
              "kwargs": {
                  "flatten": True,
              }
          },
          {
              "component": "HungerObserver",
              "kwargs": {
                  "needComponent": "PeriodicNeed",
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


def create_avatar_objects(roles: Sequence[str],
                          max_stamina_bar_states: int = 19):
  """Returns list of avatar objects of length 'num_players'."""
  avatar_objects = []
  for player_idx, role in enumerate(roles):
    if role == "default":
      # If no role was passed then set even numbered players to be banana
      # farmers and odd numbered players to be apple farmers.
      if player_idx % 2 == 1:
        specialty = "apple"
      else:
        specialty = "banana"
    elif role == "apple_farmer":
      specialty = "apple"
    elif role == "banana_farmer":
      specialty = "banana"
    else:
      raise ValueError(f"Unsupported role: {role}")
    game_object = create_avatar_object(player_idx,
                                       specialty,
                                       max_stamina_bar_states - 1)
    stamina_bar_objects = _create_stamina_overlay(player_idx,
                                                  max_stamina_bar_states)
    avatar_objects.append(game_object)
    avatar_objects.extend(stamina_bar_objects)

  return avatar_objects


def get_config():
  """Default configuration for the Fruit Market game."""
  config = configdict.ConfigDict()

  # Specify the number of players to particate in each episode (optional).
  config.recommended_num_players = 16

  # Action set configuration.
  config.action_set = ACTION_SET
  # Observation format configuration.
  config.individual_observation_names = [
      "RGB",
      "READY_TO_SHOOT",
      "STAMINA",
      "INVENTORY",
      "MY_OFFER",
      "OFFERS",
      "HUNGER",
  ]
  config.global_observation_names = [
      "WORLD.RGB",
  ]

  return config


def build(
    roles: Sequence[str],
    config: configdict.ConfigDict,
) -> Mapping[str, Any]:
  """Build this substrate given player roles."""
  substrate_definition = dict(
      levelName="trade",
      levelDirectory="meltingpot/lua/levels",
      numPlayers=len(roles),
      maxEpisodeLengthFrames=1000,
      spriteSize=8,
      topology="BOUNDED",  # Choose from ["BOUNDED", "TORUS"],
      simulation={
          "map": config.layout.ascii_map,
          "gameObjects": create_avatar_objects(roles),
          "prefabs": create_prefabs(),
          "charPrefabMap": config.layout.char_prefab_map,
          "scene": create_scene(),
      },
  )
  return substrate_definition
