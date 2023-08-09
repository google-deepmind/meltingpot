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
"""Configuration for Externality Mushrooms.

Externality mushrooms is an immediate feedback collective action problem and
social dilemma. Unlike the other sequential social dilemmas in this suite, there
is no delay between the time when an agent takes an antisocial (or prosocial)
action and when its effect is felt by all other players. Thus it is a
sequential social dilemma in the sense of Leibo et al. 2017, but not an
intertemporal social dilemma in the sense of Hughes et al. 2018.

Three types of mushrooms are spread around the map and can be consumed for a
reward. Eating a red mushroom gives a reward of 1 to the individual who
ate the mushroom. Eating a green mushroom gives a reward of 2 and it gets
divided equally among all individuals. Eating a blue mushroom gives a reward of
3 and it gets divided among the individuals except the individual who ate the
mushroom. Mushrooms regrowth depends on the type of the mushrooms eaten by
individuals. Red mushrooms regrow with a probability of 0.25 when a mushroom of
any color is eaten. Green mushrooms regrow with a probability of 0.4 when a
green or blue mushroom is eaten. Blue mushrooms regrow with a probability of 0.6
when a blue mushroom is eaten. Each mushroom has a time period that it takes to
digest it. An individual who ate a mushroom gets frozen during the time they are
digesting it. Red mushrooms get digested instantly, green and blue mushrooms
take 5 and 10 steps to digest respectively. In addition, unharvested mushrooms
spoil (and get removed from the game) after a period of time. Red, green and
blue mushrooms spoil after 75, 100 and 200 time steps respectively.

References:

Leibo JZ, Zambaldi V, Lanctot M, Marecki J, Graepel T. Multi-agent Reinforcement
Learning in Sequential Social Dilemmas (2017). AAMAS.

Hughes E, Leibo JZ, Phillips MG, Tuyls K, Duenez-Guzman EA, Garcia Castaneda A,
Dunning I, Zhu T, McKee KR, Koster R, Roff H, Graepel T. Inequity aversion
improves cooperation in intertemporal social dilemmas (2018). NeurIPS.
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

_COMPASS = ["N", "E", "S", "W"]

MARKING_SPRITE = """
oxxxxxxo
xoxxxxox
xxoxxoxx
xxxooxxx
xxxooxxx
xxoxxoxx
xoxxxxox
oxxxxxxo
"""

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
                "palettes": [shapes.FENCE_PALETTE_BROWN],
                "noRotates": [False]
            }
        },
        {"component": "BeamBlocker", "kwargs": {"beamType": "zapHit"}},
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
                "palettes": [shapes.FENCE_PALETTE_BROWN],
                "noRotates": [False]
            }
        },
        {"component": "BeamBlocker", "kwargs": {"beamType": "zapHit"}},
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
                "palettes": [shapes.FENCE_PALETTE_BROWN],
                "noRotates": [False]
            }
        },
        {"component": "BeamBlocker", "kwargs": {"beamType": "zapHit"}},
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
                "palettes": [shapes.FENCE_PALETTE_BROWN],
                "noRotates": [False]
            }
        },
        {"component": "BeamBlocker", "kwargs": {"beamType": "zapHit"}},
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
                "palettes": [shapes.FENCE_PALETTE_BROWN],
                "noRotates": [False]
            }
        },
        {"component": "BeamBlocker", "kwargs": {"beamType": "zapHit"}},
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
                "palettes": [shapes.FENCE_PALETTE_BROWN],
                "noRotates": [False]
            }
        },
        {"component": "BeamBlocker", "kwargs": {"beamType": "zapHit"}},
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
                "palettes": [shapes.FENCE_PALETTE_BROWN],
                "noRotates": [False]
            }
        },
        {"component": "BeamBlocker", "kwargs": {"beamType": "zapHit"}},
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
                "palettes": [shapes.FENCE_PALETTE_BROWN],
                "noRotates": [False]
            }
        },
        {"component": "BeamBlocker", "kwargs": {"beamType": "zapHit"}},
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
                "palettes": [shapes.FENCE_PALETTE_BROWN],
                "noRotates": [False]
            }
        },
        {"component": "BeamBlocker", "kwargs": {"beamType": "zapHit"}},
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
                "palettes": [shapes.FENCE_PALETTE_BROWN],
                "noRotates": [False]
            }
        },
        {"component": "BeamBlocker", "kwargs": {"beamType": "zapHit"}},
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
                "palettes": [shapes.FENCE_PALETTE_BROWN],
                "noRotates": [False]
            }
        },
        {"component": "BeamBlocker", "kwargs": {"beamType": "zapHit"}},
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
                "palettes": [shapes.FENCE_PALETTE_BROWN],
                "noRotates": [False]
            }
        },
        {"component": "BeamBlocker", "kwargs": {"beamType": "zapHit"}},
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
                "palettes": [shapes.FENCE_PALETTE_BROWN],
                "noRotates": [False]
            }
        },
        {"component": "BeamBlocker", "kwargs": {"beamType": "zapHit"}},
    ]
}


def get_marking_palette(alpha: float) -> Mapping[str, Sequence[int]]:
  alpha_uint8 = int(alpha * 255)
  assert alpha_uint8 >= 0.0 and alpha_uint8 <= 255, "Color value out of range."
  return {"x": shapes.ALPHA, "o": (0, 0, 0, alpha_uint8)}

DIRT = {
    "name": "dirt",
    "components": [
        {
            "component": "StateManager",
            "kwargs": {
                "initialState": "dirt",
                "stateConfigs": [{
                    "state": "dirt",
                    "layer": "background",
                    "sprite": "Dirt",
                }],
            }
        },
        {
            "component": "Appearance",
            "kwargs": {
                "renderMode": "ascii_shape",
                "spriteNames": ["Dirt"],
                "spriteShapes": [shapes.DIRT_PATTERN],
                "palettes": [{
                    "x": (81, 70, 32, 255),
                    "X": (89, 77, 36, 255),
                }],
                "noRotates": [False]
            }
        },
        {
            "component": "Transform",
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


def create_mushroom(initial_state: str = "wait"):
  """Create a mushroom prefab object."""

  mushroom_prefab = {
      "name": "mushroom",
      "components": [
          {
              "component": "StateManager",
              "kwargs": {
                  "initialState": initial_state,
                  "stateConfigs": [
                      {
                          "state": "fullInternalityZeroExternality",
                          "layer": "lowerPhysical",
                          "sprite": "FullInternalityZeroExternality",
                          "groups": ["fullInternalityZeroExternality"],
                      },
                      {
                          "state": "halfInternalityHalfExternality",
                          "layer": "lowerPhysical",
                          "sprite": "HalfInternalityHalfExternality",
                          "groups": ["halfInternalityHalfExternality"],
                      },
                      {
                          "state": "zeroInternalityFullExternality",
                          "layer": "lowerPhysical",
                          "sprite": "ZeroInternalityFullExternality",
                          "groups": ["zeroInternalityFullExternality"],
                      },
                      {
                          "state": "negativeInternalityNegativeExternality",
                          "layer": "lowerPhysical",
                          "sprite": "NegativeInternalityNegativeExternality",
                          "groups": ["negativeInternalityNegativeExternality"],
                      },
                      {
                          "state": "wait",
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
                  "renderMode": "ascii_shape",
                  "spriteNames": ["FullInternalityZeroExternality",
                                  "HalfInternalityHalfExternality",
                                  "ZeroInternalityFullExternality",
                                  "NegativeInternalityNegativeExternality"],
                  "spriteShapes": [shapes.MUSHROOM] * 4,
                  "palettes": [
                      shapes.MUSHROOM_RED_PALETTE,
                      shapes.MUSHROOM_GREEN_PALETTE,
                      shapes.MUSHROOM_BLUE_PALETTE,
                      shapes.MUSHROOM_ORANGE_PALETTE,
                  ],
                  "noRotates": [True] * 4
              }
          },
          {
              "component": "MushroomEating",
              "kwargs": {
                  "totalReward": {
                      "fullInternalityZeroExternality": 1,
                      "halfInternalityHalfExternality": 2,
                      "zeroInternalityFullExternality": 3,
                      "negativeInternalityNegativeExternality": -1.0,
                  },
                  "liveStates": ("fullInternalityZeroExternality",
                                 "halfInternalityHalfExternality",
                                 "zeroInternalityFullExternality",
                                 "negativeInternalityNegativeExternality"),
                  "numSporesReleasedWhenEaten": {
                      "fullInternalityZeroExternality": 3,
                      "halfInternalityHalfExternality": 3,
                      "zeroInternalityFullExternality": 3,
                      "negativeInternalityNegativeExternality": 1,
                  },
                  "digestionTimes": {
                      "fullInternalityZeroExternality": 0,
                      "halfInternalityHalfExternality": 10,
                      "zeroInternalityFullExternality": 15,
                      "negativeInternalityNegativeExternality": 15,
                  },
                  "destroyOnEating": {
                      "negativeInternalityNegativeExternality": {
                          "typeToDestroy": "fullInternalityZeroExternality",
                          "percentToDestroy": 0.25},
                  },
              },
          },
          {
              "component": "MushroomGrowable",
              "kwargs": {}
          },
          {
              "component": "Destroyable",
              "kwargs": {
                  "initialHealth": 1,
                  "waitState": "wait",
              }
          },
          {
              "component": "Perishable",
              "kwargs": {
                  "waitState": "wait",
                  "delayPerState": {
                      "fullInternalityZeroExternality": 200,
                      "halfInternalityHalfExternality": 100,
                      "zeroInternalityFullExternality": 75,
                      "negativeInternalityNegativeExternality": 1e7,
                  }
              }
          },
      ]
  }
  return mushroom_prefab


# Primitive action components.
# pylint: disable=bad-whitespace
# pyformat: disable
NOOP        = {"move": 0, "turn":  0, "fireZap": 0}
FORWARD     = {"move": 1, "turn":  0, "fireZap": 0}
STEP_RIGHT  = {"move": 2, "turn":  0, "fireZap": 0}
BACKWARD    = {"move": 3, "turn":  0, "fireZap": 0}
STEP_LEFT   = {"move": 4, "turn":  0, "fireZap": 0}
TURN_LEFT   = {"move": 0, "turn": -1, "fireZap": 0}
TURN_RIGHT  = {"move": 0, "turn":  1, "fireZap": 0}
FIRE_ZAP    = {"move": 0, "turn":  0, "fireZap": 1}
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
)

# Remove the first entry from human_readable_colors after using it for the self
# color to prevent it from being used again as another avatar color.
light_desaturated_avatar_palette = list(
    colors.light_desaturated_avatar_palette)
TARGET_SPRITE_SELF = {
    "name": "Self",
    "shape": shapes.CUTE_AVATAR,
    "palette": shapes.get_palette(light_desaturated_avatar_palette.pop(0)),
    "noRotate": True,
}


def create_prefabs() -> PrefabConfig:
  """Returns the prefabs.

  Prefabs are a dictionary mapping names to template game objects that can
  be cloned and placed in multiple locations accoring to an ascii map.
  """
  prefabs = {
      "dirt": DIRT,
      "spawn_point": SPAWN_POINT,
      "red_mushroom": create_mushroom(
          initial_state="fullInternalityZeroExternality"),
      "green_mushroom": create_mushroom(
          initial_state="halfInternalityHalfExternality"),
      "blue_mushroom": create_mushroom(
          initial_state="zeroInternalityFullExternality"),
      "orange_mushroom": create_mushroom(
          initial_state="negativeInternalityNegativeExternality"),
      "potential_mushroom": create_mushroom(initial_state="wait"),
      # fence prefabs
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
              "component": "MushroomRegrowth",
              "kwargs": {
                  "mushroomsToProbabilities": {
                      "fullInternalityZeroExternality": {
                          "fullInternalityZeroExternality": 0.25,
                          "halfInternalityHalfExternality": 0.0,
                          "zeroInternalityFullExternality": 0.0,
                          "negativeInternalityNegativeExternality": 0.0,
                      },
                      "halfInternalityHalfExternality": {
                          "fullInternalityZeroExternality": 0.25,
                          "halfInternalityHalfExternality": 0.4,
                          "zeroInternalityFullExternality": 0.0,
                          "negativeInternalityNegativeExternality": 0.0,
                      },
                      "zeroInternalityFullExternality": {
                          "fullInternalityZeroExternality": 0.25,
                          "halfInternalityHalfExternality": 0.4,
                          "zeroInternalityFullExternality": 0.6,
                          "negativeInternalityNegativeExternality": 0.0,
                      },
                      "negativeInternalityNegativeExternality": {
                          "fullInternalityZeroExternality": 0.0,
                          "halfInternalityHalfExternality": 0.0,
                          "zeroInternalityFullExternality": 0.0,
                          "negativeInternalityNegativeExternality": 1.0,
                      },
                  },
                  "minPotentialMushrooms": 1,
              }
          },
          {
              "component": "StochasticIntervalEpisodeEnding",
              "kwargs": {
                  "minimumFramesPerEpisode": 1000,
                  "intervalLength": 100,  # Set equal to unroll length.
                  "probabilityTerminationPerInterval": 0.2
              }
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
          },
          {
              "component": "Appearance",
              "kwargs": {
                  "renderMode": "ascii_shape",
                  "spriteNames": [source_sprite_self],
                  "spriteShapes": [shapes.CUTE_AVATAR],
                  "palettes": [
                      shapes.get_palette(
                          light_desaturated_avatar_palette[player_idx])
                  ],
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
                                  "fireZap"],
                  "actionSpec": {
                      "move": {"default": 0, "min": 0, "max": len(_COMPASS)},
                      "turn": {"default": 0, "min": -1, "max": 1},
                      "fireZap": {"default": 0, "min": 0, "max": 1},
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
                  "cooldownTime": 3,
                  "beamLength": 3,
                  "beamRadius": 1,
                  "framesTillRespawn": 50,
                  "penaltyForBeingZapped": 0,
                  "rewardForZapping": 0,
                  # GraduatedSanctionsMarking handles removal instead of Zapper.
                  "removeHitPlayer": False,
              }
          },
          {
              "component": "ReadyToShootObservation",
          },
          {
              "component": "Cumulants",
          },
      ]
  }
  if _ENABLE_DEBUG_OBSERVATIONS:
    avatar_object["components"].append({
        "component": "LocationObserver",
        "kwargs": {"objectIsAvatar": True, "alsoReportOrientation": True},
    })
    avatar_object["components"].append({
        "component": "AvatarMetricReporter",
        "kwargs": {
            "metrics": [
                {
                    "name": "ATE_MUSHROOM_FIZE",
                    "type": "Doubles",
                    "shape": [],
                    "component": "Cumulants",
                    "variable": "ate_mushroom_fize",
                },
                {
                    "name": "ATE_MUSHROOM_HIHE",
                    "type": "Doubles",
                    "shape": [],
                    "component": "Cumulants",
                    "variable": "ate_mushroom_hihe",
                },
                {
                    "name": "ATE_MUSHROOM_ZIFE",
                    "type": "Doubles",
                    "shape": [],
                    "component": "Cumulants",
                    "variable": "ate_mushroom_zife",
                },
                {
                    "name": "ATE_MUSHROOM_NINE",
                    "type": "Doubles",
                    "shape": [],
                    "component": "Cumulants",
                    "variable": "ate_mushroom_nine",
                },
                {
                    "name": "DESTROYED_MUSHROOM_FIZE",
                    "type": "Doubles",
                    "shape": [],
                    "component": "Cumulants",
                    "variable": "destroyed_mushroom_fize",
                },
                {
                    "name": "DESTROYED_MUSHROOM_HIHE",
                    "type": "Doubles",
                    "shape": [],
                    "component": "Cumulants",
                    "variable": "destroyed_mushroom_hihe",
                },
                {
                    "name": "DESTROYED_MUSHROOM_ZIFE",
                    "type": "Doubles",
                    "shape": [],
                    "component": "Cumulants",
                    "variable": "destroyed_mushroom_zife",
                },
                {
                    "name": "DESTROYED_MUSHROOM_NINE",
                    "type": "Doubles",
                    "shape": [],
                    "component": "Cumulants",
                    "variable": "destroyed_mushroom_nine",
                },
            ]
        },
    })

  return avatar_object


def create_marking_overlay(player_idx: int) -> Mapping[str, Any]:
  """Create a graduated sanctions marking overlay object."""
  # Lua is 1-indexed.
  lua_idx = player_idx + 1

  marking_object = {
      "name": "avatar_marking",
      "components": [
          {
              "component": "StateManager",
              "kwargs": {
                  "initialState": "avatarMarkingWait",
                  "stateConfigs": [
                      # Declare one state per level of the hit logic.
                      {"state": "level_1",
                       "layer": "superOverlay",
                       "sprite": "sprite_for_level_1"},
                      {"state": "level_2",
                       "layer": "superOverlay",
                       "sprite": "sprite_for_level_2"},

                      # Invisible inactive (zapped out) overlay type.
                      {"state": "avatarMarkingWait",
                       "groups": ["avatarMarkingWaits"]},
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
                  "spriteNames": ["sprite_for_level_1",
                                  "sprite_for_level_2"],
                  "spriteShapes": [MARKING_SPRITE,
                                   MARKING_SPRITE],
                  "palettes": [get_marking_palette(0.0),
                               get_marking_palette(1.0)],
                  "noRotates": [True] * 3
              }
          },
          {
              "component": "GraduatedSanctionsMarking",
              "kwargs": {
                  "playerIndex": lua_idx,
                  "waitState": "avatarMarkingWait",
                  "hitName": "zapHit",
                  "recoveryTime": 50,
                  "hitLogic": [
                      {"levelIncrement": 1,
                       "sourceReward": 0,
                       "targetReward": 0,
                       "freeze": 25},
                      {"levelIncrement": -1,
                       "sourceReward": 0,
                       "targetReward": 0,
                       "remove": True}
                  ],
              }
          },
      ]
  }
  return marking_object


def create_avatar_objects(num_players):
  """Returns list of avatar objects of length 'num_players'."""
  avatar_objects = []
  for player_idx in range(0, num_players):
    game_object = create_avatar_object(player_idx,
                                       TARGET_SPRITE_SELF)
    avatar_objects.append(game_object)

    marking_object = create_marking_overlay(player_idx)
    avatar_objects.append(marking_object)

  return avatar_objects


def get_config():
  """Default configuration for this substrate."""
  config = config_dict.ConfigDict()

  # Action set configuration.
  config.action_set = ACTION_SET
  # Observation format configuration.
  config.individual_observation_names = [
      "RGB",
      "READY_TO_SHOOT",
  ]
  config.global_observation_names = [
      "WORLD.RGB",
  ]

  # The specs of the environment (from a single-agent perspective).
  config.action_spec = specs.action(len(ACTION_SET))

  # The roles assigned to each player.
  config.valid_roles = frozenset({"default"})

  return config


def build(
    roles: Sequence[str],
    config: config_dict.ConfigDict,
) -> Mapping[str, Any]:
  """Build substrate definition given player roles."""
  num_players = len(roles)
  # Build the rest of the substrate definition.
  substrate_definition = dict(
      levelName="externality_mushrooms",
      levelDirectory="meltingpot/lua/levels",
      numPlayers=num_players,
      # Define upper bound of episode length since episodes end stochastically.
      maxEpisodeLengthFrames=5000,
      spriteSize=8,
      topology="BOUNDED",  # Choose from ["BOUNDED", "TORUS"],
      simulation={
          "map": config.layout.ascii_map,
          "gameObjects": create_avatar_objects(num_players),
          "scene": create_scene(),
          "prefabs": create_prefabs(),
          "charPrefabMap": config.layout.char_prefab_map,
      },
    )
  return substrate_definition
