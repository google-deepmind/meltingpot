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
"""Common configuration for all predator_prey__* substrates.

There are two roles: predators and prey. The prey try to eat apples and acorns.
The predators try to eat prey.

Apples are worth 1 reward and can be eaten immediately. Acorns are worth 18
reward but they take a long time to eat. It is not possible to move while eating
so a prey player is especially vulnerable while they eat it.

Predators can also eat other predators, though they get no reward for doing so.
However, a predator might eat another predator anyway in order to remove a
competitor who might otherwise eat its prey.

When prey travel together in tight groups they can defend themselves from being
eaten by predators. When a predator tries to eat its prey then all other prey
who are not currently eating an acorn within a radius of 3 are counted. If there
are more prey than predators within the radius then the predator cannot eat the
prey.

So prey are safer in groups. However, they are also tempted to depart from their
group and strike out alone since that way they are more likely to be the one to
come first to any food they find.

Both predators and prey have limited stamina. They can only move at top speed
for a limited number of consecutive steps, after which they must slow down.
Stamina is visible to all with a colored bar above each player's head. If the
bar over a particular player's head is invisible or green then they can move at
top speed. If it is red then they have depleted their stamina and can only move
slowly until they let it recharge. Stamina is recharged by standing still for
some number of consecutive time steps, how many depends on how much stamina was
depleted. Predators have a faster top speed than prey but they tire more
quickly.

Prey but not predators can cross tall grass (green grassy locations). Prey must
still be careful on grass though since predators can still reach one cell over
the border to eat prey on the edge of safety.

Both predators and prey respawn 200 steps after being eaten.
"""

from typing import Any, Dict, Generator, Mapping, Sequence

from meltingpot.utils.substrates import colors
from meltingpot.utils.substrates import shapes
from meltingpot.utils.substrates import specs
from ml_collections import config_dict as configdict

# Warning: setting `_ENABLE_DEBUG_OBSERVATIONS = True` may cause slowdown.
_ENABLE_DEBUG_OBSERVATIONS = False

_COMPASS = ("N", "E", "S", "W")
ITEMS = ("empty", "acorn")
INVISIBLE = (0, 0, 0, 0)

SPRITES = {}
PALETTES = {}

SPRITES["empty"] = """
xxxxxxxx
xxxxxxxx
xxxxxxxx
xxxxxxxx
xxxxxxxx
xxxxxxxx
xxxxxxxx
xxxxxxxx
"""

PALETTES["empty"] = {"x": INVISIBLE,}

SPRITES["acorn"] = """
xxxxxxxx
xxxxxxxx
xxxxxxxx
xxxxxxxx
xxxxxxxx
xxxxxxBB
xxxxxxbb
xxxxxxxx
"""

PALETTES["acorn"] = {
    "x": INVISIBLE,
    "B": [158, 85, 25, 255],
    "b": [92, 29, 19, 255],
}

PREDATOR_EAT_SPRITE = """
*x*x*x*x
*x*x*x**
x***&x**
xx*&xx**
xx*&xx*&
xx*&xx*x
xx&&xx&x
xxxxxxxx
"""

APPLE_SPRITE = """
xxxxxxxx
xxo|*xxx
x*#|**xx
x#***#xx
x#####xx
xx###xxx
xxxxxxxx
xxxxxxxx
"""


def create_inventory(player_index: int):
  """Return prefab for the inventory of the player at index `player_index`."""
  lua_idx = player_index + 1
  prefab = {
      "name": "inventory",
      "components": [
          {
              "component": "StateManager",
              "kwargs": {
                  "initialState": "wait",
                  "stateConfigs": (
                      [{"state": "wait"}] +
                      [{"state": item, "sprite": item, "layer": "overlay"}
                       for item in ITEMS]),
              }
          },
          {
              "component": "Transform",
          },
          {
              "component": "Appearance",
              "kwargs": {
                  "renderMode": "ascii_shape",
                  "spriteNames": ITEMS,
                  "spriteShapes": [SPRITES[item] for item in ITEMS],
                  "palettes": [PALETTES[item] for item in ITEMS],
                  "noRotates": [False]
              }
          },
          {
              "component": "AvatarConnector",
              "kwargs": {
                  "playerIndex": lua_idx,
                  "aliveState": "empty",
                  "waitState": "wait"
              }
          },
          {
              "component": "Inventory",
              "kwargs": {
                  "playerIndex": lua_idx,
              }
          },
      ]
  }
  return prefab


def create_base_prefab(name, layer="upperPhysical"):
  """Returns a base prefab with a given name on the given layer."""
  return {
      "name": f"{name}",
      "components": [
          {
              "component": "StateManager",
              "kwargs": {
                  "initialState": f"{name}",
                  "stateConfigs": [{
                      "state": f"{name}",
                      "layer": layer,
                      "sprite": f"{name}",
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
                  "spriteNames": [name],
                  "spriteShapes": [SPRITES[name]],
                  "palettes": [PALETTES[name]],
                  "noRotates": [True]
              }
          }]
  }

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
                    "layer": "upperPhysical",
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
                    "layer": "upperPhysical",
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
                "palettes": [shapes.BRICK_WALL_PALETTE],
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
                    "layer": "upperPhysical",
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
                "palettes": [shapes.BRICK_WALL_PALETTE],
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
                    "layer": "upperPhysical",
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
                "spriteShapes": [shapes.BRICK_WALL_WEST],
                "palettes": [shapes.BRICK_WALL_PALETTE],
                "noRotates": [False]
            }
        },
    ]
}

FILL = {
    "name": "fill",
    "components": [
        {
            "component": "StateManager",
            "kwargs": {
                "initialState": "fill",
                "stateConfigs": [{
                    "state": "fill",
                    "layer": "upperPhysical",
                    "sprite": "fill",
                }],
            }
        },
        {"component": "Transform"},
        {
            "component": "Appearance",
            "kwargs": {
                "renderMode": "ascii_shape",
                "spriteNames": ["fill"],
                "spriteShapes": [shapes.FILL],
                "palettes": [shapes.BRICK_WALL_PALETTE],
                "noRotates": [False]
            }
        },
    ]
}

TILED_FLOOR = {
    "name": "tiled_floor",
    "components": [
        {
            "component": "StateManager",
            "kwargs": {
                "initialState": "tiled_floor",
                "stateConfigs": [{
                    "state": "tiled_floor",
                    "layer": "background",
                    "sprite": "tiled_floor",
                }],
            }
        },
        {"component": "Transform"},
        {
            "component": "Appearance",
            "kwargs": {
                "renderMode": "ascii_shape",
                "spriteNames": ["tiled_floor"],
                "spriteShapes": [shapes.TILED_FLOOR_GREY],
                "palettes": [{"o": (204, 199, 192, 255),
                              "-": (194, 189, 182, 255),}],
                "noRotates": [False]
            }
        },
    ]
}

SAFE_GRASS = {
    "name":
        "safe_grass",
    "components": [
        {
            "component": "StateManager",
            "kwargs": {
                "initialState":
                    "safe_grass",
                "stateConfigs": [{
                    "state": "safe_grass",
                    "layer": "midPhysical",
                    "sprite": "safe_grass",
                }],
            }
        },
        {
            "component": "Transform"
        },
        {
            "component": "Appearance",
            "kwargs": {
                "renderMode": "ascii_shape",
                "spriteNames": ["safe_grass"],
                "spriteShapes": [shapes.GRASS_STRAIGHT],
                "palettes": [shapes.GRASS_PALETTE],
                "noRotates": [False]
            }
        },
    ]
}

SAFE_GRASS_N_EDGE = {
    "name":
        "safe_grass_n_edge",
    "components": [
        {
            "component": "StateManager",
            "kwargs": {
                "initialState":
                    "safe_grass_n_edge",
                "stateConfigs": [{
                    "state": "safe_grass_n_edge",
                    "layer": "midPhysical",
                    "sprite": "safe_grass_n_edge",
                }],
            }
        },
        {
            "component": "Transform"
        },
        {
            "component": "Appearance",
            "kwargs": {
                "renderMode": "ascii_shape",
                "spriteNames": ["safe_grass_n_edge"],
                "spriteShapes": [shapes.GRASS_STRAIGHT_N_EDGE],
                "palettes": [shapes.GRASS_PALETTE],
                "noRotates": [False]
            }
        },
    ]
}

SAFE_GRASS_E_EDGE = {
    "name":
        "safe_grass_e_edge",
    "components": [
        {
            "component": "StateManager",
            "kwargs": {
                "initialState":
                    "safe_grass_e_edge",
                "stateConfigs": [{
                    "state": "safe_grass_e_edge",
                    "layer": "midPhysical",
                    "sprite": "safe_grass_e_edge",
                }],
            }
        },
        {
            "component": "Transform"
        },
        {
            "component": "Appearance",
            "kwargs": {
                "renderMode": "ascii_shape",
                "spriteNames": ["safe_grass_e_edge"],
                "spriteShapes": [shapes.GRASS_STRAIGHT_E_EDGE],
                "palettes": [shapes.GRASS_PALETTE],
                "noRotates": [False]
            }
        },
    ]
}

SAFE_GRASS_S_EDGE = {
    "name":
        "safe_grass_s_edge",
    "components": [
        {
            "component": "StateManager",
            "kwargs": {
                "initialState":
                    "safe_grass_s_edge",
                "stateConfigs": [{
                    "state": "safe_grass_s_edge",
                    "layer": "midPhysical",
                    "sprite": "safe_grass_s_edge",
                }],
            }
        },
        {
            "component": "Transform"
        },
        {
            "component": "Appearance",
            "kwargs": {
                "renderMode": "ascii_shape",
                "spriteNames": ["safe_grass_s_edge"],
                "spriteShapes": [shapes.GRASS_STRAIGHT_S_EDGE],
                "palettes": [shapes.GRASS_PALETTE],
                "noRotates": [False]
            }
        },
    ]
}

SAFE_GRASS_W_EDGE = {
    "name":
        "safe_grass_w_edge",
    "components": [
        {
            "component": "StateManager",
            "kwargs": {
                "initialState":
                    "safe_grass_w_edge",
                "stateConfigs": [{
                    "state": "safe_grass_w_edge",
                    "layer": "midPhysical",
                    "sprite": "safe_grass_w_edge",
                }],
            }
        },
        {
            "component": "Transform"
        },
        {
            "component": "Appearance",
            "kwargs": {
                "renderMode": "ascii_shape",
                "spriteNames": ["safe_grass_w_edge"],
                "spriteShapes": [shapes.GRASS_STRAIGHT_W_EDGE],
                "palettes": [shapes.GRASS_PALETTE],
                "noRotates": [False]
            }
        },
    ]
}

SAFE_GRASS_NW_CORNER = {
    "name":
        "safe_grass_nw_corner",
    "components": [
        {
            "component": "StateManager",
            "kwargs": {
                "initialState":
                    "safe_grass_nw_corner",
                "stateConfigs": [{
                    "state": "safe_grass_nw_corner",
                    "layer": "midPhysical",
                    "sprite": "safe_grass_nw_corner",
                }],
            }
        },
        {
            "component": "Transform"
        },
        {
            "component": "Appearance",
            "kwargs": {
                "renderMode": "ascii_shape",
                "spriteNames": ["safe_grass_nw_corner"],
                "spriteShapes": [shapes.GRASS_STRAIGHT_NW_CORNER],
                "palettes": [shapes.GRASS_PALETTE],
                "noRotates": [False]
            }
        },
    ]
}

SAFE_GRASS_NE_CORNER = {
    "name":
        "safe_grass_ne_corner",
    "components": [
        {
            "component": "StateManager",
            "kwargs": {
                "initialState":
                    "safe_grass_ne_corner",
                "stateConfigs": [{
                    "state": "safe_grass_ne_corner",
                    "layer": "midPhysical",
                    "sprite": "safe_grass_ne_corner",
                }],
            }
        },
        {
            "component": "Transform"
        },
        {
            "component": "Appearance",
            "kwargs": {
                "renderMode": "ascii_shape",
                "spriteNames": ["safe_grass_ne_corner"],
                "spriteShapes": [shapes.GRASS_STRAIGHT_NE_CORNER],
                "palettes": [shapes.GRASS_PALETTE],
                "noRotates": [False]
            }
        },
    ]
}

SAFE_GRASS_SE_CORNER = {
    "name":
        "safe_grass_se_corner",
    "components": [
        {
            "component": "StateManager",
            "kwargs": {
                "initialState":
                    "safe_grass_se_corner",
                "stateConfigs": [{
                    "state": "safe_grass_se_corner",
                    "layer": "midPhysical",
                    "sprite": "safe_grass_se_corner",
                }],
            }
        },
        {
            "component": "Transform"
        },
        {
            "component": "Appearance",
            "kwargs": {
                "renderMode": "ascii_shape",
                "spriteNames": ["safe_grass_se_corner"],
                "spriteShapes": [shapes.GRASS_STRAIGHT_SE_CORNER],
                "palettes": [shapes.GRASS_PALETTE],
                "noRotates": [False]
            }
        },
    ]
}

SAFE_GRASS_SW_CORNER = {
    "name":
        "safe_grass_sw_corner",
    "components": [
        {
            "component": "StateManager",
            "kwargs": {
                "initialState":
                    "safe_grass_sw_corner",
                "stateConfigs": [{
                    "state": "safe_grass_sw_corner",
                    "layer": "midPhysical",
                    "sprite": "safe_grass_sw_corner",
                }],
            }
        },
        {
            "component": "Transform"
        },
        {
            "component": "Appearance",
            "kwargs": {
                "renderMode": "ascii_shape",
                "spriteNames": ["safe_grass_sw_corner"],
                "spriteShapes": [shapes.GRASS_STRAIGHT_SW_CORNER],
                "palettes": [shapes.GRASS_PALETTE],
                "noRotates": [False]
            }
        },
    ]
}


def create_apple(apple_reward: float = 1.0):
  """Return a prefab object defining an apple, which can be eaten by prey."""
  prefab = {
      "name":
          "edibleApple",
      "components": [
          {
              "component": "StateManager",
              "kwargs": {
                  "initialState":
                      "apple",
                  "stateConfigs": [
                      {
                          "state": "apple",
                          "layer": "lowerPhysical",
                          "sprite": "apple",
                      },
                      {
                          "state": "appleWait",
                          "layer": "logic",
                      },
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
                  "spriteNames": ["apple"],
                  "spriteShapes": [APPLE_SPRITE],
                  "palettes": [{
                      "*": (106, 184, 83, 255),
                      "#": (96, 166, 75, 255),
                      "o": (61, 130, 62, 255),
                      "|": (115, 62, 57, 255),
                      "x": INVISIBLE,
                  }],
                  "noRotates": [True],
              }
          },
          {
              "component": "AppleEdible",
              "kwargs": {
                  "liveState": "apple",
                  "waitState": "appleWait",
                  "rewardForEating": apple_reward,
              }
          },
          {
              "component": "FixedRateRegrow",
              "kwargs": {
                  "name": "AppleFixedRateRegrow",
                  "liveState": "apple",
                  "waitState": "appleWait",
                  "regrowRate": 0.007,
              }
          },
      ]
  }
  return prefab

FLOOR_ACORN = {
    "name":
        "floorAcorn",
    "components": [
        {
            "component": "StateManager",
            "kwargs": {
                "initialState":
                    "floorAcorn",
                "stateConfigs": [
                    {
                        "state": "floorAcorn",
                        "layer": "lowerPhysical",
                        "sprite": "floorAcorn",
                    },
                    {
                        "state": "acornWait",
                        "layer": "logic",
                    },
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
                "spriteNames": ["floorAcorn"],
                "spriteShapes": [shapes.ACORN],
                "palettes": [{
                    "*": (158, 85, 25, 255),
                    "@": (158, 85, 25, 140),
                    "o": (92, 29, 19, 255),
                    "x": INVISIBLE,
                }],
                "noRotates": [True],
            }
        },
        {
            "component": "AcornPickUppable",
            "kwargs": {
                "liveState": "floorAcorn",
                "waitState": "acornWait",
            }
        },
        {
            "component": "FixedRateRegrow",
            "kwargs": {
                "name": "AcornFixedRateRegrow",
                "liveState": "floorAcorn",
                "waitState": "acornWait",
                "regrowRate": 0.01,
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
                    "layer": "alternateLogic",
                    "groups": ["spawnPoints"]
                }],
            }
        },
        {"component": "Transform"},
    ]
}


def create_spawn_point_prefab(team):
  """Return a team-specific spawn-point prefab."""
  prefab = {
      "name": "spawn_point",
      "components": [
          {
              "component": "StateManager",
              "kwargs": {
                  "initialState": "playerSpawnPoint",
                  "stateConfigs": [{
                      "state": "playerSpawnPoint",
                      "layer": "alternateLogic",
                      "groups": ["{}SpawnPoints".format(team)],
                  }],
              }
          },
          {"component": "Transform",},
          {
              "component": "Appearance",
              "kwargs": {
                  "renderMode": "invisible",
                  "spriteNames": [],
                  "spriteRGBColors": []
              }
          },
      ]
  }
  return prefab


def create_prefabs(apple_reward: float = 1.0):
  """Returns the prefabs dictionary."""
  prefabs = {
      "spawn_point_prey": create_spawn_point_prefab("prey"),
      "spawn_point_predator": create_spawn_point_prefab("predator"),
      "apple": create_apple(apple_reward=apple_reward),
      "floor_acorn": FLOOR_ACORN,
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
      "fill": FILL,
      "tiled_floor": TILED_FLOOR,
      "safe_grass": SAFE_GRASS,
      "safe_grass_n_edge": SAFE_GRASS_N_EDGE,
      "safe_grass_e_edge": SAFE_GRASS_E_EDGE,
      "safe_grass_s_edge": SAFE_GRASS_S_EDGE,
      "safe_grass_w_edge": SAFE_GRASS_W_EDGE,
      "safe_grass_ne_corner": SAFE_GRASS_NE_CORNER,
      "safe_grass_se_corner": SAFE_GRASS_SE_CORNER,
      "safe_grass_sw_corner": SAFE_GRASS_SW_CORNER,
      "safe_grass_nw_corner": SAFE_GRASS_NW_CORNER,
  }
  return prefabs

# Primitive action components.
# pylint: disable=bad-whitespace
# pyformat: disable
NOOP       = {"move": 0, "turn":  0, "interact": 0}
FORWARD    = {"move": 1, "turn":  0, "interact": 0}
STEP_RIGHT = {"move": 2, "turn":  0, "interact": 0}
BACKWARD   = {"move": 3, "turn":  0, "interact": 0}
STEP_LEFT  = {"move": 4, "turn":  0, "interact": 0}
TURN_LEFT  = {"move": 0, "turn": -1, "interact": 0}
TURN_RIGHT = {"move": 0, "turn":  1, "interact": 0}
INTERACT   = {"move": 0, "turn":  0, "interact": 1}
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
    INTERACT,
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


def _create_avatar_object(player_idx: int, is_predator: bool,
                          max_stamina: int) -> Dict[str, Any]:
  """Create an avatar object."""
  # Lua is 1-indexed.
  lua_index = player_idx + 1
  live_state_name = "player{}".format(lua_index)
  avatar_sprite_name = "avatarSprite{}".format(lua_index)

  if is_predator:
    spawn_group = "predatorSpawnPoints"
    color_palette = shapes.PRED1_PALETTE
    sprite = shapes.PERSISTENCE_PREDATOR
  if not is_predator:
    spawn_group = "preySpawnPoints"
    alert_state_name = live_state_name + "alert"
    sit_state_name = live_state_name + "sit"
    prep_to_eat_state_name = live_state_name + "prepToEat"
    first_bite_state_name = live_state_name + "firstBite"
    second_bite_state_name = live_state_name + "secondBite"
    last_bite_state_name = live_state_name + "lastBite"
    alert_sprite_name = "avatarAlertSprite{}".format(lua_index)
    sit_sprite_name = "avatarSitSprite{}".format(lua_index)
    prep_to_eat_sprite_name = "avatarPrepToEatSprite{}".format(lua_index)
    first_bite_sprite_name = "avatarFirstBiteSprite{}".format(lua_index)
    second_bite_sprite_name = "avatarSecondBiteSprite{}".format(lua_index)
    last_bite_sprite_name = "avatarLastBiteSprite{}".format(lua_index)
    color_palette = {**shapes.get_palette(colors.palette[player_idx]
                                          ), ** PALETTES["acorn"]}
    sprite = shapes.CUTE_AVATAR
    alert_sprite = shapes.CUTE_AVATAR_ALERT
    sit_sprite = shapes.CUTE_AVATAR_SIT
    prep_to_eat_sprite = shapes.CUTE_AVATAR_EAT
    first_bite_sprite = shapes.CUTE_AVATAR_FIRST_BITE
    second_bite_sprite = shapes.CUTE_AVATAR_SECOND_BITE
    last_bite_sprite = shapes.CUTE_AVATAR_LAST_BITE

  interact_palette = {
      "P": colors.palette[player_idx] + (255,),
      "&": (10, 10, 10, 50),
      "*": (230, 230, 230, 255),
      "x": INVISIBLE,
  }

  if is_predator:
    role_name = "predator"
    green_freeze_time = 0
    yellow_freeze_time = 1
    red_freeze_time = 6
  else:
    role_name = "prey"
    green_freeze_time = 1
    yellow_freeze_time = 2
    red_freeze_time = 4

  live_state_name = "player{}".format(lua_index)
  avatar_object = {
      "name": f"avatar{lua_index}",
      "components": [
          {
              "component": "Transform",
          },
          {
              "component": "Role",
              "kwargs": {
                  "isPredator": is_predator,
              }
          },
          {
              "component": "Stamina",
              "kwargs": {
                  "maxStamina": max_stamina,
                  "classConfig": {"name": role_name,
                                  "greenFreezeTime": green_freeze_time,
                                  "yellowFreezeTime": yellow_freeze_time,
                                  "redFreezeTime": red_freeze_time},
                  "amountInvisible": 6,
                  "amountGreen": 6,
                  "amountYellow": 6,
                  "amountRed": 1,
                  "costlyActions": ["move", "turn", "interact"],
              }
          },
          {
              "component": "StaminaObservation",
              "kwargs": {
                  "staminaComponent": "Stamina",
              }
          },
          # The `RewardForStaminaLevel` component defines a pseudoreward which
          # is useful for training background populations to rapidly learn how
          # to control their stamina. For the default "real" substrate, the
          # reward it defines should always be zero.
          {
              "component": "RewardForStaminaLevel",
              "kwargs": {
                  "rewardValue": 0.0,
                  "bands": [],
              }
          },
      ]
  }
  if is_predator:
    avatar_object["components"].extend([
        {
            "component": "StateManager",
            "kwargs": {
                "initialState": live_state_name,
                "stateConfigs": [
                    # Initial player state.
                    {
                        "state": live_state_name,
                        "layer": "upperPhysical",
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
            "component": "PredatorInteractBeam",
            "kwargs": {
                "cooldownTime": 5,
                "shapes": [PREDATOR_EAT_SPRITE, shapes.FILL],
                "palettes": [interact_palette] * 2,
            }
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
                "additionalLiveStates": [],
                "waitState": "playerWait",
                "spawnGroup": spawn_group,
                "actionOrder": ["move",
                                "turn",
                                "interact"],
                "actionSpec": {
                    "move": {"default": 0, "min": 0, "max": len(_COMPASS)},
                    "turn": {"default": 0, "min": -1, "max": 1},
                    "interact": {"default": 0, "min": 0, "max": 1},
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
            "component": "AvatarEdible",
        },
        {
            "component": "AvatarRespawn",
            "kwargs": {
                "framesTillRespawn": 200,
            }
        },
        ])
  if not is_predator:
    avatar_object["components"].extend([
        {
            "component": "StateManager",
            "kwargs": {
                "initialState": live_state_name,
                "stateConfigs": [
                    # Initial player state.
                    {
                        "state": live_state_name,
                        "layer": "upperPhysical",
                        "sprite": avatar_sprite_name,
                        "contact": "avatar",
                        "groups": ["players"]
                    },
                    # Player wait type for times when they are zapped out.
                    {
                        "state": "playerWait",
                        "groups": ["playerWaits"]
                    },
                    {
                        "state": alert_state_name,
                        "layer": "upperPhysical",
                        "sprite": alert_sprite_name,
                        "contact": "avatar",
                        "groups": ["players"]
                    },
                    {
                        "state": sit_state_name,
                        "layer": "upperPhysical",
                        "sprite": sit_sprite_name,
                        "contact": "avatar",
                        "groups": ["players"]
                    },
                    {
                        "state": prep_to_eat_state_name,
                        "layer": "upperPhysical",
                        "sprite": prep_to_eat_sprite_name,
                        "contact": "avatar",
                        "groups": ["players"]
                    },
                    {
                        "state": first_bite_state_name,
                        "layer": "upperPhysical",
                        "sprite": first_bite_sprite_name,
                        "contact": "avatar",
                        "groups": ["players"]
                    },
                    {
                        "state": second_bite_state_name,
                        "layer": "upperPhysical",
                        "sprite": second_bite_sprite_name,
                        "contact": "avatar",
                        "groups": ["players"]
                    },
                    {
                        "state": last_bite_state_name,
                        "layer": "upperPhysical",
                        "sprite": last_bite_sprite_name,
                        "contact": "avatar",
                        "groups": ["players"]
                    }
                ]
            }
        },
        {
            "component": "Avatar",
            "kwargs": {
                "index": lua_index,
                "aliveState": live_state_name,
                "additionalLiveStates": [alert_state_name,
                                         sit_state_name,
                                         prep_to_eat_state_name,
                                         first_bite_state_name,
                                         second_bite_state_name,
                                         last_bite_state_name],
                "waitState": "playerWait",
                "spawnGroup": spawn_group,
                "actionOrder": ["move", "turn", "interact"],
                "actionSpec": {
                    "move": {"default": 0, "min": 0, "max": len(_COMPASS)},
                    "turn": {"default": 0, "min": -1, "max": 1},
                    "interact": {"default": 0, "min": 0, "max": 1},
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
            "component": "InteractEatAcorn",
            "kwargs": {
                "cooldownTime": 5,
                "shapes": [PREDATOR_EAT_SPRITE, shapes.FILL],
                "palettes": [interact_palette],
                "isEating": False,
                "defaultState": live_state_name,
            }
        },
        {
            "component": "Appearance",
            "kwargs": {
                "renderMode": "ascii_shape",
                "spriteNames": [avatar_sprite_name, alert_sprite_name,
                                sit_sprite_name, prep_to_eat_sprite_name,
                                first_bite_sprite_name, second_bite_sprite_name,
                                last_bite_sprite_name],
                "spriteShapes": [sprite, alert_sprite, sit_sprite,
                                 prep_to_eat_sprite, first_bite_sprite,
                                 second_bite_sprite, last_bite_sprite],
                "palettes": [color_palette] * 7,
                "noRotates": [True] * 7
            }
        },
        {
            "component": "AvatarEatingAnimation",
            "kwargs": {
                "sit": sit_state_name,
                "prepToEat": prep_to_eat_state_name,
                "firstBite": first_bite_state_name,
                "secondBite": second_bite_state_name,
                "lastBite": last_bite_state_name,
                "downState": live_state_name,
                # On each of 3 eating frames, get one third of `acornReward`.
                "acornReward": 18,
                }
        },
        {
            "component": "AvatarEdible",
            "kwargs": {
                "groupRadius": 3,
                "predatorRewardForEating": 1.0,
            }
        },
        {
            "component": "AvatarRespawn",
            "kwargs": {
                "framesTillRespawn": 200,
            }
        },
        {
            "component": "AvatarAnimation",
            "kwargs": {
                "upState": alert_state_name,
                "downState": live_state_name,
            }
        },
        # The `AcornTaste` component defines pseudorewards which are useful for
        # training background populations. For the default "real" substrate,
        # the rewards it defines should always be zero,
        {
            "component": "AcornTaste",
            "kwargs": {
                "collectReward": 0.0,
                "eatReward": 0.0,
                "safeAcornConsumptionReward": 0.0,
            }
        },
    ])

  if _ENABLE_DEBUG_OBSERVATIONS:
    avatar_object["components"].append({
        "component": "LocationObserver",
        "kwargs": {"objectIsAvatar": True, "alsoReportOrientation": True},
    })

  return avatar_object


def _create_predator_obstacle(player_idx: int) -> Dict[str, Any]:
  # Lua is 1-indexed.
  lua_idx = player_idx + 1
  return {
      "name":
          "predator_obstacle",
      "components": [
          {
              "component": "StateManager",
              "kwargs":
                  {
                      "initialState": "obstacleWait",
                      "stateConfigs": [
                          {
                              "state": "obstacleWait"
                          },
                          # Block predators from entering any tile with a
                          # piece on layer 'midPhysical'.
                          {
                              "state": "obstacleLive",
                              "layer": "midPhysical",
                              "groups": ["obstacles"]
                          }
                      ]
                  }
          },
          {
              "component": "Transform",
          },
          {
              "component": "AvatarConnector",
              "kwargs": {
                  "playerIndex": lua_idx,
                  "aliveState": "obstacleLive",
                  "waitState": "obstacleWait"
              }
          },
      ]
  }


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
      level = (
          blank_space
          + "\nx"
          + "G" * number_of_gs
          + "Y" * number_of_ys
          + "R" * number_of_rs
          + "x"
      )
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


def _build_prey_objects(player_idx: int,
                        max_stamina_bar_states: int = 19):
  """Build a prey avatar and its associated stamina objects."""
  avatar_object = _create_avatar_object(
      player_idx, is_predator=False, max_stamina=max_stamina_bar_states - 1)
  stamina_bar_objects = _create_stamina_overlay(player_idx,
                                                max_stamina_bar_states)
  inventory_object = create_inventory(player_index=player_idx)
  game_objects = []
  game_objects.append(avatar_object)
  game_objects.extend(stamina_bar_objects)
  game_objects.append(inventory_object)
  return game_objects


def _build_predator_objects(player_idx: int,
                            max_stamina_bar_states: int = 19):
  """Build a predator avatar and its associated stamina objects and obstacle."""
  avatar_object = _create_avatar_object(
      player_idx, is_predator=True, max_stamina=max_stamina_bar_states - 1)
  stamina_bar_objects = _create_stamina_overlay(player_idx,
                                                max_stamina_bar_states)
  predator_obstacle = _create_predator_obstacle(player_idx)
  game_objects = []
  game_objects.append(avatar_object)
  game_objects.extend(stamina_bar_objects)
  game_objects.append(predator_obstacle)
  return game_objects


def get_config():
  """Default configuration."""
  config = configdict.ConfigDict()

  # Declare parameters here that we may want to override externally.
  # `apple_reward` should be 1.0 for the canonical version of this environment,
  # but to train background bots it is sometimes useful to use other values in
  # order to control the relative attractiveness of apples and acorns.
  config.apple_reward = 1.0

  # Action set configuration.
  config.action_set = ACTION_SET
  # Observation format configuration.
  config.individual_observation_names = [
      "RGB",
      "STAMINA",
  ]
  config.global_observation_names = [
      "WORLD.RGB",
  ]

  # The specs of the environment (from a single-agent perspective).
  config.action_spec = specs.action(len(ACTION_SET))

  # The roles assigned to each player.
  config.valid_roles = frozenset({"predator", "prey"})

  return config


def build(
    roles: Sequence[str],
    config: configdict.ConfigDict,
) -> Mapping[str, Any]:
  """Build predator_and_prey substrate given player roles."""
  # Build avatars.
  num_players = len(roles)
  avatar_objects_and_helpers = []
  for player_idx, role in enumerate(roles):
    if role == "prey":
      avatar_objects_and_helpers.extend(_build_prey_objects(player_idx))
    elif role == "predator":
      avatar_objects_and_helpers.extend(_build_predator_objects(player_idx))
    else:
      raise ValueError(f"Unrecognized role: {role}")

  # Build the rest of the substrate definition.
  substrate_definition = dict(
      levelName="predator_prey",
      levelDirectory="meltingpot/lua/levels",
      numPlayers=num_players,
      maxEpisodeLengthFrames=1000,
      spriteSize=8,
      topology="BOUNDED",  # Choose from ["BOUNDED", "TORUS"],
      simulation=dict(
          map=config.layout.ascii_map,
          gameObjects=avatar_objects_and_helpers,
          scene=create_scene(),
          prefabs=create_prefabs(apple_reward=config.apple_reward),
          charPrefabMap=config.layout.char_prefab_map,
      ),
  )

  return substrate_definition
