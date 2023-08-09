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
"""Configuration for Factory of the Commons."""

from typing import Any, Dict, Generator, Mapping, Sequence

from meltingpot.utils.substrates import colors
from meltingpot.utils.substrates import shapes
from meltingpot.utils.substrates import specs
from ml_collections import config_dict

# Warning: setting `_ENABLE_DEBUG_OBSERVATIONS = True` may cause slowdown.
_ENABLE_DEBUG_OBSERVATIONS = False

_COMPASS = ["N", "E", "S", "W"]
INVISIBLE = (0, 0, 0, 0)

GRASP_SHAPE = """
xxxxxxxx
xxxxxxxx
xxxxxxxx
xxxxxxxx
xxxxxxxx
xxxxxxxx
xoxxxxox
xxooooxx
"""

FLOOR_MARKING = {
    "name":
        "floor_marking",
    "components": [
        {
            "component": "StateManager",
            "kwargs": {
                "initialState":
                    "floor_marking",
                "stateConfigs": [{
                    "state": "floor_marking",
                    "layer": "lowestPhysical",
                    "sprite": "floor_marking",
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
                "spriteNames": ["floor_marking"],
                "spriteShapes": [shapes.FLOOR_MARKING],
                "palettes": [shapes.DISPENSER_BELT_PALETTE],
                "noRotates": [False]
            }
        },
    ]
}

PINK_CUBE_DISPENSING_ANIMATION = {
    "name":
        "pink_cube_dispensing_animation",
    "components": [
        {
            "component": "StateManager",
            "kwargs": {
                "initialState":
                    "waitState",
                "stateConfigs": [
                    {
                        "state": "waitState",
                        "layer": "overlay",
                    },
                    {
                        "state": "pink_cube_dispensing_1",
                        "layer": "overlay",
                        "sprite": "pink_cube_dispensing_1",
                    },
                    {
                        "state": "pink_cube_dispensing_2",
                        "layer": "overlay",
                        "sprite": "pink_cube_dispensing_2",
                    },

                    {
                        "state": "pink_cube_dispensing_3",
                        "layer": "overlay",
                        "sprite": "pink_cube_dispensing_3",
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
                "spriteNames": ["pink_cube_dispensing_1",
                                "pink_cube_dispensing_2",
                                "pink_cube_dispensing_3"],
                "spriteShapes": [shapes.CUBE_DISPENSING_ANIMATION_1,
                                 shapes.CUBE_DISPENSING_ANIMATION_2,
                                 shapes.CUBE_DISPENSING_ANIMATION_3],
                "palettes": [{
                    "a": (255, 174, 182, 255),
                    "A": (240, 161, 169, 255),
                    "&": (237, 140, 151, 255),
                    "x": (0, 0, 0, 0),
                }] * 3,
                "noRotates": [True] * 3,
            }
        },
        {
            "component": "ObjectDispensingAnimation",
            "kwargs": {
                "frameOne": "pink_cube_dispensing_1",
                "frameTwo": "pink_cube_dispensing_2",
                "frameThree": "pink_cube_dispensing_3",
                "waitState": "waitState",
            }
        },
    ]
}

DISPENSER_INDICATOR_PINK_CUBE = {
    "name":
        "dispenser_indicator_pink_cube",
    "components": [
        {
            "component": "StateManager",
            "kwargs": {
                "initialState":
                    "dispenser_pink_cube",
                "stateConfigs": [
                    {
                        "state": "dispenser_pink_cube",
                        "layer": "midPhysical",
                        "sprite": "dispenser_pink_cube",
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
                "spriteNames": ["dispenser_pink_cube"],
                "spriteShapes": [shapes.HOPPER_INDICATOR_SINGLE_BLOCK],
                "palettes": [{
                    "x": (0, 0, 0, 0),
                    "a": (255, 174, 182, 255),
                }],
                "noRotates": [False]
            }
        },
        {
            "component": "DispenserIndicator",
            "kwargs": {
                "objectOne": "PinkCube",
                "objectTwo": "NoneNeeded",
            }
        }
    ]
}


SPAWN_POINT = {
    "name":
        "spawnPoint",
    "components": [
        {
            "component": "StateManager",
            "kwargs": {
                "initialState":
                    "spawnPoint",
                "stateConfigs": [{
                    "state": "spawnPoint",
                    "layer": "alternateLogic",
                    "groups": ["spawnPoints"]
                }],
            }
        },
        {
            "component": "Transform"
        },
    ]
}


def get_blue_cube(initial_state: str):
  """Get a blue cube prefab."""
  prefab = {
      "name":
          "blue_cube_live",
      "components": [
          {
              "component": "StateManager",
              "kwargs": {
                  "initialState": initial_state,
                  "stateConfigs": [
                      {
                          "state": "blue_cube",
                          "layer": "lowerPhysical",
                          "sprite": "blue_cube",
                      },
                      {
                          "state": "blue_jump",
                          "layer": "lowerPhysical",
                          "sprite": "blue_jump",
                      },
                      {
                          "state": "blue_cube_drop_one",
                          "layer": "lowerPhysical",
                          "sprite": "blue_cube_drop_one",
                      },
                      {
                          "state": "blue_cube_drop_two",
                          "layer": "lowerPhysical",
                          "sprite": "blue_cube_drop_two",
                      },
                      {
                          "state": "waitState",
                      }
                  ],
              }
          },
          {
              "component": "Transform"
          },
          {
              "component": "Appearance",
              "kwargs": {
                  "renderMode": "ascii_shape",
                  "spriteNames": ["blue_cube", "blue_cube_drop_one",
                                  "blue_cube_drop_two", "blue_jump"],
                  "spriteShapes": [shapes.BLOCK,
                                   shapes.BLOCK_DROP_1,
                                   shapes.BLOCK_DROP_2,
                                   shapes.CUBE_DISPENSING_ANIMATION_1],
                  "palettes": [shapes.FACTORY_OBJECTS_PALETTE,] * 4,
                  "noRotates": [True] * 4
              }
          },
          {
              "component": "Receivable",
              "kwargs": {
                  "waitState": "waitState",
                  "liveState": "blue_cube",
              }
          },
          {
              "component": "ReceiverDropAnimation",
              "kwargs": {
                  "dropOne": "blue_cube_drop_one",
                  "dropTwo": "blue_cube_drop_two",
              }
          },
          {
              "component": "Token",
              "kwargs": {
                  "type": "BlueCube"
              }
          },
          {
              "component": "ObjectJumpAnimation",
              "kwargs": {
                  "jump": "blue_jump",
                  "drop": "blue_cube",
                  "waitState": "waitState",
              }
          },
          {
              "component": "Graspable",
              "kwargs": {
                  "graspableStates": ("blue_cube",),
                  "disconnectStates": (
                      "blue_jump", "blue_cube_drop_one", "blue_cube_drop_two",
                      "waitState",),
              }
          }
      ]
  }
  return prefab


BANANA = {
    "name":
        "banana",
    "components": [
        {
            "component": "StateManager",
            "kwargs": {
                "initialState": "waitState",
                "stateConfigs": [
                    {
                        "state": "banana",
                        "layer": "lowerPhysical",
                        "sprite": "banana",
                    },
                    {
                        "state": "banana_jump",
                        "layer": "lowerPhysical",
                        "sprite": "banana_jump",
                    },
                    {
                        "state": "banana_drop_one",
                        "layer": "lowerPhysical",
                        "sprite": "banana_drop_one",
                    },
                    {
                        "state": "banana_drop_two",
                        "layer": "lowerPhysical",
                        "sprite": "banana_drop_two",
                    },
                    {
                        "state": "waitState"
                    }
                ],
            }
        },
        {
            "component": "Transform"
        },
        {
            "component": "Appearance",
            "kwargs": {
                "renderMode": "ascii_shape",
                "spriteNames": ["banana", "banana_drop_one", "banana_drop_two",
                                "banana_jump"],
                "spriteShapes": [shapes.BANANA,
                                 shapes.BANANA_DROP_1,
                                 shapes.BANANA_DROP_2,
                                 shapes.BANANA,],
                "palettes": [shapes.FACTORY_OBJECTS_PALETTE,] * 4,
                "noRotates": [True] * 4
            }
        },
        {
            "component": "Receivable",
            "kwargs": {
                "waitState": "waitState",
                "liveState": "banana",
            }
        },
        {
            "component": "ReceiverDropAnimation",
            "kwargs": {
                "dropOne": "banana_drop_one",
                "dropTwo": "banana_drop_two",
            }
        },
        {
            "component": "Token",
            "kwargs": {
                "type": "Banana"
            }
        },
        {
            "component": "SecondObjectJumpAnimation",
            "kwargs": {
                "jump": "banana",
                "drop": "banana",
                "waitState": "waitState",
            }
        },
        {
            "component": "Graspable",
            "kwargs": {
                "graspableStates": ("banana",),
                "disconnectStates": (
                    "banana_jump", "banana_drop_one", "banana_drop_two",
                    "waitState",),
            }
        }
    ]
}

APPLE = {
    "name":
        "apples",
    "components": [
        {
            "component": "StateManager",
            "kwargs": {
                "initialState":
                    "waitState",
                "stateConfigs": [
                    {
                        "state": "waitState",
                    },
                    {
                        "state": "apple",
                        "layer": "appleLayer",
                        "sprite": "apple",
                    },
                    {
                        "state": "apple_jump_state",
                        "layer": "appleLayer",
                        "sprite": "apple_jump_sprite",
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
                "spriteNames": ["apple", "apple_jump_sprite"],
                "spriteShapes": [shapes.APPLE, shapes.APPLE_JUMP],
                "palettes": [shapes.APPLE_RED_PALETTE] * 2,
                "noRotates": [True] * 2,
            }
        },
        {
            "component": "Graspable",
            "kwargs": {
                "graspableStates": ("apple",),
                "disconnectStates": ("apple_jump_state", "waitState",),
            }
        },
        {
            "component": "AppleComponent",
            "kwargs": {
                "liveState": "apple",
                "waitState": "waitState",
                "rewardForEating": 1,
            }
        },
        {
            "component": "Token",
            "kwargs": {
                "type": "Apple"
            }
        },
        {
            "component": "ObjectJumpAnimation",
            "kwargs": {
                "jump": "apple_jump_state",
                "drop": "apple",
                "waitState": "waitState",
            }
        },
        {
            "component": "SecondObjectJumpAnimation",
            "kwargs": {
                "jump": "apple",
                "drop": "apple",
                "waitState": "waitState",
            }
        },

    ]
}

PINK_CUBE = {
    "name":
        "pink_cube",
    "components": [
        {
            "component": "StateManager",
            "kwargs": {
                "initialState": "waitState",
                "stateConfigs": [
                    {
                        "state": "pink_cube",
                        "layer": "lowerPhysical",
                        "sprite": "pink_cube",
                    },
                    {
                        "state": "pink_cube_drop_one",
                        "layer": "lowerPhysical",
                        "sprite": "pink_cube_drop_one",
                    },
                    {
                        "state": "pink_cube_drop_two",
                        "layer": "lowerPhysical",
                        "sprite": "pink_cube_drop_two",
                    },
                    {
                        "state": "pink_jump",
                        "layer": "lowerPhysical",
                        "sprite": "pink_jump",
                    },
                    {
                        "state": "waitState",
                    }
                ],
            }
        },
        {
            "component": "Transform"
        },
        {
            "component": "Appearance",
            "kwargs": {
                "renderMode": "ascii_shape",
                "spriteNames": ["pink_cube", "pink_cube_drop_one",
                                "pink_cube_drop_two", "pink_jump"],
                "spriteShapes": [shapes.BLOCK,
                                 shapes.BLOCK_DROP_1,
                                 shapes.BLOCK_DROP_2,
                                 shapes.CUBE_DISPENSING_ANIMATION_1],
                "palettes": [{
                    "a": (255, 174, 182, 255),
                    "A": (240, 161, 169, 255),
                    "&": (237, 140, 151, 255),
                    "x": (0, 0, 0, 0),
                }] * 4,
                "noRotates": [True] * 4
            }
        },
        {
            "component": "Receivable",
            "kwargs": {
                "waitState": "waitState",
                "liveState": "pink_cube",
            }
        },
        {
            "component": "Token",
            "kwargs": {
                "type": "PinkCube"
            }
        },
        {
            "component": "ReceiverDropAnimation",
            "kwargs": {
                "dropOne": "pink_cube_drop_one",
                "dropTwo": "pink_cube_drop_two",
            }
        },
        {
            "component": "ObjectJumpAnimation",
            "kwargs": {
                "jump": "pink_jump",
                "drop": "pink_cube",
                "waitState": "waitState",
            }
        },
        {
            "component": "Graspable",
            "kwargs": {
                "graspableStates": ("pink_cube",),
                "disconnectStates": (
                    "pink_cube_drop_one", "pink_cube_drop_two", "pink_jump",
                    "waitState",),
            }
        }
    ]
}

APPLE_DISPENSING = {
    "name":
        "apple_dispensing",
    "components": [
        {
            "component": "StateManager",
            "kwargs": {
                "initialState":
                    "waitState",
                "stateConfigs": [
                    {
                        "state": "waitState",
                        "layer": "overlay",
                    },
                    {
                        "state": "apple_dispensing_1",
                        "layer": "overlay",
                        "sprite": "apple_dispensing_1",
                    },
                    {
                        "state": "apple_dispensing_2",
                        "layer": "overlay",
                        "sprite": "apple_dispensing_2",
                    },

                    {
                        "state": "apple_dispensing_3",
                        "layer": "overlay",
                        "sprite": "apple_dispensing_3",
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
                "spriteNames": ["apple_dispensing_1", "apple_dispensing_2",
                                "apple_dispensing_3"],
                "spriteShapes": [shapes.APPLE_DISPENSING_ANIMATION_1,
                                 shapes.APPLE_DISPENSING_ANIMATION_2,
                                 shapes.APPLE_DISPENSING_ANIMATION_3],
                "palettes": [shapes.FACTORY_OBJECTS_PALETTE] * 3,
                "noRotates": [True] * 3,
            }
        },
        {
            "component": "ObjectDispensingAnimation",
            "kwargs": {
                "frameOne": "apple_dispensing_1",
                "frameTwo": "apple_dispensing_2",
                "frameThree": "apple_dispensing_3",
                "waitState": "waitState",
            }
        },
    ]
}

CUBE_APPLE_DISPENSING_ANIMATION = {
    "name":
        "cube_apple_dispensing_animation",
    "components": [
        {
            "component": "StateManager",
            "kwargs": {
                "initialState":
                    "waitState",
                "stateConfigs": [
                    {
                        "state": "waitState",
                        "layer": "overlay",
                    },
                    {
                        "state": "apple_dispensing_1",
                        "layer": "overlay",
                        "sprite": "apple_dispensing_1",
                    },
                    {
                        "state": "apple_dispensing_2",
                        "layer": "overlay",
                        "sprite": "apple_dispensing_2",
                    },
                    {
                        "state": "apple_dispensing_3",
                        "layer": "overlay",
                        "sprite": "apple_dispensing_3",
                    },
                    {
                        "state": "blue_cube_dispensing_1",
                        "layer": "overlay",
                        "sprite": "blue_cube_dispensing_1",
                    },
                    {
                        "state": "blue_cube_dispensing_2",
                        "layer": "overlay",
                        "sprite": "blue_cube_dispensing_2",
                    },
                    {
                        "state": "blue_cube_dispensing_3",
                        "layer": "overlay",
                        "sprite": "blue_cube_dispensing_3",
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
                "spriteNames": ["apple_dispensing_1", "apple_dispensing_2",
                                "apple_dispensing_3", "blue_cube_dispensing_1",
                                "blue_cube_dispensing_2",
                                "blue_cube_dispensing_3"],
                "spriteShapes": [shapes.APPLE_DISPENSING_ANIMATION_1,
                                 shapes.APPLE_DISPENSING_ANIMATION_2,
                                 shapes.APPLE_DISPENSING_ANIMATION_3,
                                 shapes.CUBE_DISPENSING_ANIMATION_1,
                                 shapes.CUBE_DISPENSING_ANIMATION_2,
                                 shapes.CUBE_DISPENSING_ANIMATION_3],
                "palettes": [shapes.FACTORY_OBJECTS_PALETTE] * 6,
                "noRotates": [True] * 6,
            }
        },
        {
            "component": "DoubleObjectDispensingAnimation",
            "kwargs": {
                "frameOne": "blue_cube_dispensing_1",
                "frameTwo": "blue_cube_dispensing_2",
                "frameThree": "blue_cube_dispensing_3",
                "frameFour": "apple_dispensing_1",
                "frameFive": "apple_dispensing_2",
                "frameSix": "apple_dispensing_3",
                "waitState": "waitState",
            }
        },
    ]
}

BANANA_CUBE_DISPENSING_ANIMATION = {
    "name":
        "banana_cube_dispensing_animation",
    "components": [
        {
            "component": "StateManager",
            "kwargs": {
                "initialState":
                    "waitState",
                "stateConfigs": [
                    {
                        "state": "waitState",
                        "layer": "overlay",
                    },
                    {
                        "state": "banana_dispensing_1",
                        "layer": "overlay",
                        "sprite": "banana_dispensing_1",
                    },
                    {
                        "state": "banana_dispensing_2",
                        "layer": "overlay",
                        "sprite": "banana_dispensing_2",
                    },
                    {
                        "state": "banana_dispensing_3",
                        "layer": "overlay",
                        "sprite": "banana_dispensing_3",
                    },
                    {
                        "state": "blue_cube_dispensing_1",
                        "layer": "overlay",
                        "sprite": "blue_cube_dispensing_1",
                    },
                    {
                        "state": "blue_cube_dispensing_2",
                        "layer": "overlay",
                        "sprite": "blue_cube_dispensing_2",
                    },

                    {
                        "state": "blue_cube_dispensing_3",
                        "layer": "overlay",
                        "sprite": "blue_cube_dispensing_3",
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
                "spriteNames": ["banana_dispensing_1", "banana_dispensing_2",
                                "banana_dispensing_3", "blue_cube_dispensing_1",
                                "blue_cube_dispensing_2",
                                "blue_cube_dispensing_3"],
                "spriteShapes": [shapes.BANANA_DISPENSING_ANIMATION_1,
                                 shapes.BANANA,
                                 shapes.BANANA_DISPENSING_ANIMATION_3,
                                 shapes.CUBE_DISPENSING_ANIMATION_1,
                                 shapes.CUBE_DISPENSING_ANIMATION_2,
                                 shapes.CUBE_DISPENSING_ANIMATION_3],
                "palettes": [shapes.FACTORY_OBJECTS_PALETTE] * 6,
                "noRotates": [True] * 6,
            }
        },
        {
            "component": "DoubleObjectDispensingAnimation",
            "kwargs": {
                "frameOne": "blue_cube_dispensing_1",
                "frameTwo": "blue_cube_dispensing_2",
                "frameThree": "blue_cube_dispensing_3",
                "frameFour": "banana_dispensing_1",
                "frameFive": "banana_dispensing_2",
                "frameSix": "banana_dispensing_3",
                "waitState": "waitState",
            }
        },
    ]
}

PINK_CUBE_DISPENSING = {
    "name":
        "pink_cube_dispensing",
    "components": [
        {
            "component": "StateManager",
            "kwargs": {
                "initialState":
                    "waitState",
                "stateConfigs": [
                    {
                        "state": "waitState",
                        "layer": "overlay",
                    },
                    {
                        "state": "pink_cube_dispensing_1",
                        "layer": "overlay",
                        "sprite": "pink_cube_dispensing_1",
                    },
                    {
                        "state": "pink_cube_dispensing_2",
                        "layer": "overlay",
                        "sprite": "pink_cube_dispensing_2",
                    },

                    {
                        "state": "pink_cube_dispensing_3",
                        "layer": "overlay",
                        "sprite": "pink_cube_dispensing_3",
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
                "spriteNames": ["pink_cube_dispensing_1",
                                "pink_cube_dispensing_2",
                                "pink_cube_dispensing_3"],
                "spriteShapes": [shapes.CUBE_DISPENSING_ANIMATION_1,
                                 shapes.CUBE_DISPENSING_ANIMATION_2,
                                 shapes.CUBE_DISPENSING_ANIMATION_3],
                "palettes": [{
                    "a": (255, 174, 182, 255),
                    "A": (240, 161, 169, 255),
                    "&": (237, 140, 151, 255),
                    "x": (0, 0, 0, 0),
                }] * 3,
                "noRotates": [True] * 3,
            }
        },
        {
            "component": "DoubleObjectDispensingAnimation",
            "kwargs": {
                "frameOne": "pink_cube_dispensing_1",
                "frameTwo": "pink_cube_dispensing_2",
                "frameThree": "pink_cube_dispensing_3",
                "frameFour": "waitState",
                "frameFive": "waitState",
                "frameSix": "waitState",
                "waitState": "waitState",
            }
        },
    ]
}


HOPPER_MOUTH = {
    "name":
        "hopper_mouth",
    "components": [
        {
            "component": "StateManager",
            "kwargs": {
                "initialState":
                    "hopper_mouth_open",
                "stateConfigs": [
                    {
                        "state": "hopper_mouth_closed",
                        "layer": "lowestPhysical",
                        "sprite": "hopper_mouth_closed",
                    },
                    {
                        "state": "hopper_mouth_closing",
                        "layer": "lowestPhysical",
                        "sprite": "hopper_mouth_closing",
                    },
                    {
                        "state": "hopper_mouth_open",
                        "layer": "lowestPhysical",
                        "sprite": "hopper_mouth_open",
                    },
                    {
                        "state": "waitState"
                    }
                ],
            }
        },
        {
            "component": "Transform"
        },
        {
            "component": "Appearance",
            "kwargs": {
                "renderMode": "ascii_shape",
                "spriteNames": ["hopper_mouth_closed", "hopper_mouth_closing",
                                "hopper_mouth_open"],
                "spriteShapes": [shapes.HOPPER_CLOSED,
                                 shapes.HOPPER_CLOSING,
                                 shapes.HOPPER_OPEN],
                "palettes": [shapes.FACTORY_MACHINE_BODY_PALETTE] * 3,
                "noRotates": [False] * 3
            }
        },
        {
            "component": "Receiver"
        },
        {
            "component": "HopperMouth",
            "kwargs": {
                "closed": "hopper_mouth_closed",
                "opening": "hopper_mouth_closing",
                "open": "hopper_mouth_open",
            }
        },
    ]
}

HOPPER_BODY = {
    "name":
        "hopper_body",
    "components": [
        {
            "component": "StateManager",
            "kwargs": {
                "initialState":
                    "hopper_body",
                "stateConfigs": [
                    {
                        "state": "hopper_body",
                        "layer": "midPhysical",
                        "sprite": "hopper_body",
                    },
                    {
                        "state": "hopper_body_activated",
                        "layer": "midPhysical",
                        "sprite": "hopper_body_activated",
                    }
                ],
            }
        },
        {
            "component": "Transform"
        },
        {
            "component": "Appearance",
            "kwargs": {
                "renderMode": "ascii_shape",
                "spriteNames": ["hopper_body", "hopper_body_activated"],
                "spriteShapes": [shapes.HOPPER_BODY,
                                 shapes.HOPPER_BODY_ACTIVATED],
                "palettes": [{
                    "a": (140, 129, 129, 255),
                    "b": (84, 77, 77, 255),
                    "f": (92, 98, 120, 255),
                    "g": (92, 98, 120, 255),
                    "c": (92, 98, 120, 255),
                    "x": (0, 0, 0, 0),
                }] * 2,
                "noRotates": [False] * 2
            }
        },
    ]
}


HOPPER_INDICATOR = {
    "name":
        "hopper_indicator",
    "components": [
        {
            "component": "StateManager",
            "kwargs": {
                "initialState":
                    "hopper_indicator_two",
                "stateConfigs": [
                    {
                        "state": "waitState",
                    },
                    {
                        "state": "hopper_indicator_one",
                        "layer": "upperPhysical",
                        "sprite": "hopper_indicator_one",
                        "groups": ["indicator"]
                    },
                    {
                        "state": "hopper_indicator_two",
                        "layer": "upperPhysical",
                        "sprite": "hopper_indicator_two",
                        "groups": ["indicator"]
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
                "spriteNames":
                    [
                        "hopper_indicator_two", "hopper_indicator_one",
                    ],
                "spriteShapes": [
                    shapes.HOPPER_INDICATOR_TWO_BLOCKS,
                    shapes.HOPPER_INDICATOR_ONE_BLOCK,],
                "palettes": [shapes.FACTORY_OBJECTS_PALETTE] * 2,
                "noRotates": [False] * 2
            }
        },
        {
            "component": "ReceiverIndicator",
            "kwargs": {
                "waitState": "waitState",
                "liveState": "hopper_indicator_two",
                "secondLiveState": "hopper_indicator_one",
                "count": "Double",
                "type": "TwoBlocks",
            }
        }
    ]
}

HOPPER_INDICATOR_BLUE_CUBE = {
    "name":
        "hopper_indicator_blue_cube",
    "components": [
        {
            "component": "StateManager",
            "kwargs": {
                "initialState":
                    "waitState",
                "stateConfigs": [
                    {
                        "state": "waitState",
                    },
                    {
                        "state": "blue_cube_indicator",
                        "layer": "upperPhysical",
                        "sprite": "blue_cube_indicator",
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
                "spriteNames": ["blue_cube_indicator"],
                "spriteShapes": [shapes.HOPPER_INDICATOR_SINGLE_BLOCK],
                "palettes": [shapes.FACTORY_OBJECTS_PALETTE],
                "noRotates": [False]
            }
        },
        {
            "component": "ReceiverIndicator",
            "kwargs": {
                "waitState": "waitState",
                "liveState": "blue_cube_indicator",
                "secondLiveState": "waitState",
                "count": "Single",
                "type": "BlueCube"
            }
        }
    ]
}

HOPPER_INDICATOR_BANANA = {
    "name":
        "hopper_indicator_banana",
    "components": [
        {
            "component": "StateManager",
            "kwargs": {
                "initialState":
                    "hopper_banana",
                "stateConfigs": [
                    {
                        "state": "hopper_banana",
                        "layer": "upperPhysical",
                        "sprite": "hopper_banana",
                    },
                    {
                        "state": "waitState"
                    }
                ]
                }
            },
        {
            "component": "ReceiverIndicator",
            "kwargs": {
                "waitState": "waitState",
                "liveState": "hopper_banana",
                "secondLiveState": "waitState",
                "count": "Single",
                "type": "Banana",
            }
        },
        {
            "component": "Transform",
        },
        {
            "component": "Appearance",
            "kwargs": {
                "renderMode": "ascii_shape",
                "spriteNames": ["hopper_banana"],
                "spriteShapes": [shapes.HOPPER_INDICATOR_SINGLE_BANANA],
                "palettes": [shapes.FACTORY_OBJECTS_PALETTE],
                "noRotates": [False]
            }
        },
    ]
}

HOPPER_INDICATOR_PINK_CUBE = {
    "name":
        "hopper_indicator_pink_cube",
    "components": [
        {
            "component": "StateManager",
            "kwargs": {
                "initialState":
                    "waitState",
                "stateConfigs": [
                    {
                        "state": "waitState",
                    },
                    {
                        "state": "hopper_pink_cube",
                        "layer": "upperPhysical",
                        "sprite": "hopper_pink_cube",
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
                "spriteNames": ["hopper_pink_cube"],
                "spriteShapes": [shapes.HOPPER_INDICATOR_SINGLE_BLOCK],
                "palettes":
                    [{
                        "x": (0, 0, 0, 0),
                        "a": (255, 174, 182, 255),
                    }],
                "noRotates": [False]
            }
        },
        {
            "component": "ReceiverIndicator",
            "kwargs": {
                "waitState": "waitState",
                "liveState": "hopper_pink_cube",
                "secondLiveState": "waitState",
                "count": "Single",
                "type": "PinkCube",
            }
        }
    ]
}

DISPENSER_INDICATOR_BANANA_CUBE = {
    "name":
        "dispenser_indicator_banana_cube",
    "components": [
        {
            "component": "StateManager",
            "kwargs": {
                "initialState":
                    "banana_cube",
                "stateConfigs": [
                    {
                        "state": "banana_cube",
                        "layer": "midPhysical",
                        "sprite": "banana_cube",
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
                "spriteNames": ["banana_cube"],
                "spriteShapes": [shapes.HOPPER_INDICATOR_ON],
                "palettes": [shapes.FACTORY_OBJECTS_PALETTE],
                "noRotates": [False]
            }
        },
        {
            "component": "DispenserIndicator",
            "kwargs": {
                "objectOne": "BlueCube",
                "objectTwo": "Banana",
            }
        }
    ]
}

DISPENSER_INDICATOR_CUBE_APPLE = {
    "name":
        "dispenser_indicator_cube_apple",
    "components": [
        {
            "component": "StateManager",
            "kwargs": {
                "initialState":
                    "cube_apple",
                "stateConfigs": [
                    {
                        "state": "cube_apple",
                        "layer": "midPhysical",
                        "sprite": "cube_apple",
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
                "spriteNames": ["cube_apple"],
                "spriteShapes": [shapes.APPLE_CUBE_INDICATOR],
                "palettes": [shapes.FACTORY_OBJECTS_PALETTE],
                "noRotates": [False]
            }
        },
        {
            "component": "DispenserIndicator",
            "kwargs": {
                "objectOne": "Apple",
                "objectTwo": "BlueCube",
            }
        }
    ]
}

DISPENSER_INDICATOR_APPLE = {
    "name":
        "dispenser_indicator_apple",
    "components": [
        {
            "component": "StateManager",
            "kwargs": {
                "initialState":
                    "dispenser_indicator_apple",
                "stateConfigs": [
                    {
                        "state": "dispenser_indicator_apple",
                        "layer": "midPhysical",
                        "sprite": "dispenser_indicator_apple",
                        "groups": ["indicator"]
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
                "spriteNames":
                    [
                        "dispenser_indicator_apple",
                    ],
                "spriteShapes": [
                    shapes.APPLE_INDICATOR],
                "palettes": [shapes.FACTORY_OBJECTS_PALETTE],
                "noRotates": [False]
            }
        },
        {
            "component": "DispenserIndicator",
            "kwargs": {
                "objectOne": "Apple",
                "objectTwo": "NoneNeeded",
            }
        }
    ]
}

DISPENSER_INDICATOR_TWO_APPLES = {
    "name":
        "dispenser_indicator_two_apples",
    "components": [
        {
            "component": "StateManager",
            "kwargs": {
                "initialState":
                    "two_apples",
                "stateConfigs": [
                    {
                        "state": "two_apples",
                        "layer": "midPhysical",
                        "sprite": "two_apples",
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
                "spriteNames": ["two_apples"],
                "spriteShapes": [shapes.DOUBLE_APPLE_INDICATOR],
                "palettes": [shapes.FACTORY_OBJECTS_PALETTE],
                "noRotates": [False]
            }
        },
        {
            "component": "DispenserIndicator",
            "kwargs": {
                "objectOne": "Apple",
                "objectTwo": "Apple",
            }
        }
    ]
}

DISPENSER_BODY = {
    "name":
        "dispenser_body",
    "components": [
        {
            "component": "StateManager",
            "kwargs": {
                "initialState": "dispenser_body",
                "stateConfigs": [
                    {
                        "state": "dispenser_body",
                        "layer": "lowerPhysical",
                        "sprite": "dispenser_body",
                        "groups": ["dispenser"]
                    },
                    {
                        "state": "dispenser_body_activated",
                        "layer": "lowerPhysical",
                        "sprite": "dispenser_body_activated",
                        "groups": ["dispenser"]
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
                "spriteNames": [
                    "dispenser_body",
                    "dispenser_body_activated",
                ],
                "spriteShapes": [
                    shapes.DISPENSER_BODY,
                    shapes.DISPENSER_BODY_ACTIVATED,
                ],
                "palettes": [shapes.FACTORY_MACHINE_BODY_PALETTE] * 2,
                "noRotates": [False] * 2
            }
        },
    ]
}

DISPENSER_BELT = {
    "name":
        "dispenser_belt",
    "components": [
        {
            "component": "StateManager",
            "kwargs": {
                "initialState": "dispenser_belt_deactivated",
                "stateConfigs": [
                    {
                        "state": "dispenser_belt_deactivated",
                        "layer": "lowestPhysical",
                        "sprite": "dispenser_belt_deactivated",
                        "groups": ["dispenser"]
                    },
                    {
                        "state": "dispenser_belt_on_position_1",
                        "layer": "lowestPhysical",
                        "sprite": "dispenser_belt_on_position_1",
                        "groups": ["dispenser"]
                    },
                    {
                        "state": "dispenser_belt_on_position_2",
                        "layer": "lowestPhysical",
                        "sprite": "dispenser_belt_on_position_2",
                        "groups": ["dispenser"]
                    },
                    {
                        "state": "dispenser_belt_on_position_3",
                        "layer": "lowestPhysical",
                        "sprite": "dispenser_belt_on_position_3",
                        "groups": ["dispenser"]
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
                "spriteNames": [
                    "dispenser_belt_deactivated",
                    "dispenser_belt_on_position_1",
                    "dispenser_belt_on_position_2",
                    "dispenser_belt_on_position_3",
                ],
                "spriteShapes": [
                    shapes.DISPENSER_BELT_OFF,
                    shapes.DISPENSER_BELT_ON_POSITION_1,
                    shapes.DISPENSER_BELT_ON_POSITION_2,
                    shapes.DISPENSER_BELT_ON_POSITION_3,
                ],
                "palettes": [shapes.DISPENSER_BELT_PALETTE] * 4,
                "noRotates": [False] * 4
            }
        },
        {
            "component": "ConveyerBeltOnAnimation",
            "kwargs": {
                "waitState": "dispenser_belt_deactivated",
                "stateOne": "dispenser_belt_on_position_1",
                "stateTwo": "dispenser_belt_on_position_2",
                "stateThree": "dispenser_belt_on_position_3",
            }
        }
    ]
}


NW_WALL_CORNER = {
    "name":
        "nw_wall_corner",
    "components": [
        {
            "component": "StateManager",
            "kwargs": {
                "initialState":
                    "nw_wall_corner",
                "stateConfigs": [{
                    "state": "nw_wall_corner",
                    "layer": "lowerPhysical",
                    "sprite": "NwWallCorner",
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
                "spriteNames": ["NwWallCorner"],
                "spriteShapes": [shapes.NW_PERSPECTIVE_WALL],
                "palettes": [shapes.PERSPECTIVE_WALL_PALETTE],
                "noRotates": [False]
            }
        },
        {"component": "BeamBlocker", "kwargs": {"beamType": "hold"}},
        {"component": "BeamBlocker", "kwargs": {"beamType": "shove"}},
    ]
}

NE_WALL_CORNER = {
    "name":
        "ne_wall_corner",
    "components": [
        {
            "component": "StateManager",
            "kwargs": {
                "initialState":
                    "ne_wall_corner",
                "stateConfigs": [{
                    "state": "ne_wall_corner",
                    "layer": "upperPhysical",
                    "sprite": "NeWallCorner",
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
                "spriteNames": ["NeWallCorner"],
                "spriteShapes": [shapes.NE_PERSPECTIVE_WALL],
                "palettes": [shapes.PERSPECTIVE_WALL_PALETTE],
                "noRotates": [False]
            }
        },
        {"component": "BeamBlocker", "kwargs": {"beamType": "hold"}},
        {"component": "BeamBlocker", "kwargs": {"beamType": "shove"}},
    ]
}

WALL_HORIZONTAL = {
    "name":
        "wall_horizontal",
    "components": [
        {
            "component": "StateManager",
            "kwargs": {
                "initialState":
                    "wall_horizontal",
                "stateConfigs": [{
                    "state": "wall_horizontal",
                    "layer": "lowerPhysical",
                    "sprite": "WallHorizontal",
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
                "spriteNames": ["WallHorizontal"],
                "spriteShapes": [shapes.PERSPECTIVE_WALL],
                "palettes": [shapes.PERSPECTIVE_WALL_PALETTE],
                "noRotates": [False]
            }
        },
        {"component": "BeamBlocker", "kwargs": {"beamType": "hold"}},
        {"component": "BeamBlocker", "kwargs": {"beamType": "shove"}},
    ]
}

WALL_T_COUPLING = {
    "name":
        "wall_t_coupling",
    "components": [
        {
            "component": "StateManager",
            "kwargs": {
                "initialState":
                    "wall_t_coupling",
                "stateConfigs": [{
                    "state": "wall_t_coupling",
                    "layer": "upperPhysical",
                    "sprite": "WallTCoupling",
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
                "spriteNames": ["WallTCoupling"],
                "spriteShapes": [shapes.PERSPECTIVE_WALL_T_COUPLING],
                "palettes": [shapes.PERSPECTIVE_WALL_PALETTE],
                "noRotates": [False]
            }
        },
        {"component": "BeamBlocker", "kwargs": {"beamType": "hold"}},
        {"component": "BeamBlocker", "kwargs": {"beamType": "shove"}},
    ]
}

WALL_EAST = {
    "name":
        "wall_east",
    "components": [
        {
            "component": "StateManager",
            "kwargs": {
                "initialState":
                    "wall_east",
                "stateConfigs": [{
                    "state": "wall_east",
                    "layer": "lowerPhysical",
                    "sprite": "WallEast",
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
                "spriteNames": ["WallEast"],
                "spriteShapes": [shapes.E_PERSPECTIVE_WALL],
                "palettes": [shapes.PERSPECTIVE_WALL_PALETTE],
                "noRotates": [False]
            }
        },
        {"component": "BeamBlocker", "kwargs": {"beamType": "hold"}},
        {"component": "BeamBlocker", "kwargs": {"beamType": "shove"}},
    ]
}

WALL_WEST = {
    "name":
        "wall_west",
    "components": [
        {
            "component": "StateManager",
            "kwargs": {
                "initialState":
                    "wall_west",
                "stateConfigs": [{
                    "state": "wall_west",
                    "layer": "lowerPhysical",
                    "sprite": "WallWest",
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
                "spriteNames": ["WallWest"],
                "spriteShapes": [shapes.W_PERSPECTIVE_WALL],
                "palettes": [shapes.PERSPECTIVE_WALL_PALETTE],
                "noRotates": [False]
            }
        },
        {"component": "BeamBlocker", "kwargs": {"beamType": "hold"}},
        {"component": "BeamBlocker", "kwargs": {"beamType": "shove"}},
    ]
}

WALL_MIDDLE = {
    "name":
        "wall_middle",
    "components": [
        {
            "component": "StateManager",
            "kwargs": {
                "initialState":
                    "wall_middle",
                "stateConfigs": [{
                    "state": "wall_middle",
                    "layer": "lowerPhysical",
                    "sprite": "WallMiddle",
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
                "spriteNames": ["WallMiddle"],
                "spriteShapes": [shapes.MID_PERSPECTIVE_WALL],
                "palettes": [shapes.PERSPECTIVE_WALL_PALETTE],
                "noRotates": [False]
            }
        },
        {"component": "BeamBlocker", "kwargs": {"beamType": "hold"}},
        {"component": "BeamBlocker", "kwargs": {"beamType": "shove"}},
    ]
}

THRESHOLD = {
    "name":
        "threshold",
    "components": [
        {
            "component": "StateManager",
            "kwargs": {
                "initialState":
                    "threshold",
                "stateConfigs": [{
                    "state": "threshold",
                    "layer": "lowestPhysical",
                    "sprite": "Threshold",
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
                "spriteNames": ["Threshold"],
                "spriteShapes": [shapes.PERSPECTIVE_THRESHOLD],
                "palettes": [shapes.PERSPECTIVE_WALL_PALETTE],
                "noRotates": [False]
            }
        },
    ]
}

TILED_FLOOR = {
    "name":
        "tiled_floor",
    "components": [
        {
            "component": "StateManager",
            "kwargs": {
                "initialState":
                    "tiled_floor",
                "stateConfigs": [{
                    "state": "tiled_floor",
                    "layer": "background",
                    "sprite": "tiled_floor",
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
                "spriteNames": ["tiled_floor"],
                "spriteShapes": [shapes.METAL_FLOOR_DOUBLE_SPACED],
                "palettes": [shapes.FACTORY_FLOOR_PALETTE],
                "noRotates": [False]
            }
        },
    ]
}

FLOOR_MARKING = {
    "name":
        "floor_marking",
    "components": [
        {
            "component": "StateManager",
            "kwargs": {
                "initialState":
                    "floor_marking",
                "stateConfigs": [{
                    "state": "floor_marking",
                    "layer": "lowestPhysical",
                    "sprite": "floor_marking",
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
                "spriteNames": ["floor_marking"],
                "spriteShapes": [shapes.FLOOR_MARKING],
                "palettes": [shapes.DISPENSER_BELT_PALETTE],
                "noRotates": [False]
            }
        },
    ]
}

FLOOR_MARKING_TOP = {
    "name":
        "floor_marking_top",
    "components": [
        {
            "component": "StateManager",
            "kwargs": {
                "initialState":
                    "floor_marking_top",
                "stateConfigs": [{
                    "state": "floor_marking_top",
                    "layer": "lowestPhysical",
                    "sprite": "floor_marking_top",
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
                "spriteNames": ["floor_marking_top"],
                "spriteShapes": [shapes.FLOOR_MARKING_LONG_TOP],
                "palettes": [shapes.DISPENSER_BELT_PALETTE],
                "noRotates": [False]
            }
        },
    ]
}

FLOOR_MARKING_BOTTOM = {
    "name":
        "floor_marking_bottom",
    "components": [
        {
            "component": "StateManager",
            "kwargs": {
                "initialState":
                    "floor_marking_bottom",
                "stateConfigs": [{
                    "state": "floor_marking_bottom",
                    "layer": "lowestPhysical",
                    "sprite": "floor_marking_bottom",
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
                "spriteNames": ["floor_marking_bottom"],
                "spriteShapes": [shapes.FLOOR_MARKING_LONG_BOTTOM],
                "palettes": [shapes.DISPENSER_BELT_PALETTE],
                "noRotates": [False]
            }
        },
    ]
}

human_readable_colors = list(colors.human_readable)
target_sprite_color = human_readable_colors.pop(0)
grappling_target_color_palette = shapes.get_palette(target_sprite_color)
# Add character mappings to avatar pallete for Magic Beam overlay
grappling_target_color_palette["P"] = (196, 77, 190, 130)
grappling_target_color_palette["p"] = (184, 72, 178, 80)
TARGET_SPRITE_SELF = {
    "default": {
        "name": "Self",
        "shape": shapes.CUTE_AVATAR,
        "palette": shapes.get_palette(target_sprite_color),
        "noRotate": True,
    },
    "grappling": {
        "name": "SelfGrappling",
        "shape": shapes.CUTE_AVATAR_ARMS_UP,
        "palette": grappling_target_color_palette,
        "noRotate": True,
    },
    "grappled": {
        "name": "SelfGrappled",
        "shape": shapes.MAGIC_GRAPPLED_AVATAR,
        "palette": grappling_target_color_palette,
        "noRotate": True,
    },
}

# PREFABS is a dictionary mapping names to template game objects that can
# be cloned and placed in multiple locations accoring to an ascii map.graspable
PREFABS = {
    "spawn_point": SPAWN_POINT,
    # Graspable objects.
    "apple": APPLE,
    "blue_cube_live": get_blue_cube(initial_state="blue_cube"),
    "pink_cube": PINK_CUBE,
    "blue_cube_wait": get_blue_cube(initial_state="waitState"),
    "banana": BANANA,
    # Dynamic components.
    "hopper_body": HOPPER_BODY,
    "hopper_mouth": HOPPER_MOUTH,
    # Hopper indicators.
    "hopper_indicator": HOPPER_INDICATOR,
    "hopper_indicator_pink_cube": HOPPER_INDICATOR_PINK_CUBE,
    "hopper_indicator_blue_cube": HOPPER_INDICATOR_BLUE_CUBE,
    "hopper_indicator_banana": HOPPER_INDICATOR_BANANA,
    # Dispenser indicators.
    "dispenser_indicator_apple": DISPENSER_INDICATOR_APPLE,
    "dispenser_indicator_two_apples": DISPENSER_INDICATOR_TWO_APPLES,
    "dispenser_indicator_pink_cube": DISPENSER_INDICATOR_PINK_CUBE,
    "dispenser_indicator_banana_cube": DISPENSER_INDICATOR_BANANA_CUBE,
    "dispenser_indicator_cube_apple": DISPENSER_INDICATOR_CUBE_APPLE,
    "dispenser_body": DISPENSER_BODY,
    "dispenser_belt": DISPENSER_BELT,
    "apple_dispensing_animation": APPLE_DISPENSING,
    "pink_cube_dispensing_animation": PINK_CUBE_DISPENSING_ANIMATION,
    "banana_cube_dispensing_animation": BANANA_CUBE_DISPENSING_ANIMATION,
    "cube_apple_dispensing_animation": CUBE_APPLE_DISPENSING_ANIMATION,
    # Static components.
    "nw_wall_corner": NW_WALL_CORNER,
    "ne_wall_corner": NE_WALL_CORNER,
    "wall_horizontal": WALL_HORIZONTAL,
    "wall_t_coupling": WALL_T_COUPLING,
    "wall_east": WALL_EAST,
    "wall_west": WALL_WEST,
    "wall_middle": WALL_MIDDLE,
    "threshold": THRESHOLD,
    "tiled_floor": TILED_FLOOR,
    "floor_marking": FLOOR_MARKING,
    "floor_marking_top": FLOOR_MARKING_TOP,
    "floor_marking_bottom": FLOOR_MARKING_BOTTOM,
}

# Primitive action components.
# pylint: disable=bad-whitespace
# pyformat: disable
NOOP       = {"move": 0, "turn":  0, "pickup": 0, "grasp": 0, "hold": 0, "shove":  0}
FORWARD    = {"move": 1, "turn":  0, "pickup": 0, "grasp": 0, "hold": 0, "shove":  0}
STEP_RIGHT = {"move": 2, "turn":  0, "pickup": 0, "grasp": 0, "hold": 0, "shove":  0}
BACKWARD   = {"move": 3, "turn":  0, "pickup": 0, "grasp": 0, "hold": 0, "shove":  0}
STEP_LEFT  = {"move": 4, "turn":  0, "pickup": 0, "grasp": 0, "hold": 0, "shove":  0}
TURN_LEFT  = {"move": 0, "turn": -1, "pickup": 0, "grasp": 0, "hold": 0, "shove":  0}
TURN_RIGHT = {"move": 0, "turn":  1, "pickup": 0, "grasp": 0, "hold": 0, "shove":  0}
PICKUP     = {"move": 0, "turn":  0, "pickup": 1, "grasp": 0, "hold": 0, "shove":  0}
GRASP      = {"move": 0, "turn":  0, "pickup": 0, "grasp": 1, "hold": 0, "shove":  0}
HOLD       = {"move": 0, "turn":  0, "pickup": 0, "grasp": 0, "hold": 1, "shove":  0}
# Notice that SHOVE includes both `hold` and `shove` parts.
SHOVE      = {"move": 0, "turn":  0, "pickup": 0, "grasp": 0, "hold": 1, "shove":  1}
PULL       = {"move": 0, "turn":  0, "pickup": 0, "grasp": 0, "hold": 1, "shove": -1}
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
    PICKUP,
    GRASP,
    HOLD,
    SHOVE,
    PULL,
)


def create_scene():
  """Creates the global scene."""
  scene = {
      "name":
          "scene",
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
              "component": "Transform"
          },
          {
              "component": "StochasticIntervalEpisodeEnding",
              "kwargs": {
                  "minimumFramesPerEpisode": 1000,
                  "intervalLength": 100,  # Set equal to unroll length.
                  "probabilityTerminationPerInterval": 0.1
              }
          }
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


def create_avatar_object(player_idx: int,
                         target_sprite_self: Dict[str, Any],
                         max_stamina_bar_states: int) -> Dict[str, Any]:
  """Create an avatar object."""
  # Lua is 1-indexed.
  lua_index = player_idx + 1
  # Setup the self vs other sprite mapping.
  avatar_sprite_name = "avatarSprite{}".format(lua_index)
  grappling_sprite = "AvatarGrappling" + str(lua_index)
  grappled_sprite = "AvatarGrappled" + str(lua_index)

  custom_sprite_map = {
      avatar_sprite_name: target_sprite_self["default"]["name"],
      grappling_sprite: target_sprite_self["grappling"]["name"],
      grappled_sprite: target_sprite_self["grappled"]["name"],
  }

  live_state_name = "player{}".format(lua_index)
  grappling_state_name = f"player{lua_index}_grappling"
  grappled_state_name = f"player{lua_index}_grappled"

  color_palette = shapes.get_palette(colors.palette[player_idx])
  # Add character mappings to avatar pallete for Magic Beam overlay
  color_palette["P"] = (196, 77, 190, 130)
  color_palette["p"] = (184, 72, 178, 80)
  spawn_group = "spawnPoints"

  avatar_object = {
      "name":
          "avatar",
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
                          "layer": "midPhysical",
                          "sprite": avatar_sprite_name,
                          "contact": "avatar",
                          "groups": ["players"]
                      },
                      {
                          "state": grappling_state_name,
                          "layer": "upperPhysical",
                          "sprite": grappling_sprite,
                          "contact": "avatar",
                          "groups": ["players"]
                      },
                      {
                          "state": grappled_state_name,
                          "layer": "upperPhysical",
                          "sprite": grappled_sprite,
                          "contact": "avatar",
                          "groups": ["players"]},
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
                  "spriteNames": [avatar_sprite_name, grappling_sprite,
                                  grappled_sprite],
                  "spriteShapes": [shapes.CUTE_AVATAR,
                                   shapes.CUTE_AVATAR_ARMS_UP,
                                   shapes.MAGIC_GRAPPLED_AVATAR],
                  "palettes": [color_palette] * 3,
                  "noRotates": [True] * 3
              }
          },
          {
              "component": "AdditionalSprites",
              "kwargs": {
                  "renderMode": "ascii_shape",
                  "customSpriteNames": [
                      target_sprite_self["default"]["name"],
                      target_sprite_self["grappling"]["name"],
                      target_sprite_self["grappled"]["name"],
                  ],
                  "customSpriteShapes": [
                      target_sprite_self["default"]["shape"],
                      target_sprite_self["grappling"]["shape"],
                      target_sprite_self["grappled"]["shape"],
                  ],
                  "customPalettes": [
                      target_sprite_self["default"]["palette"],
                      target_sprite_self["grappling"]["palette"],
                      target_sprite_self["grappled"]["palette"],
                  ],
                  "customNoRotates": [
                      target_sprite_self["default"]["noRotate"],
                      target_sprite_self["grappling"]["noRotate"],
                      target_sprite_self["grappled"]["noRotate"],
                  ],
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
                  "spawnGroup": spawn_group,
                  "actionOrder": [
                      "move",
                      "turn",
                      "pickup",
                      "grasp",
                      # Grappling actions
                      "hold",
                      "shove",
                  ],
                  "actionSpec": {
                      "move": {"default": 0, "min": 0, "max": len(_COMPASS)},
                      "turn": {"default": 0, "min": -1, "max": 1},
                      "pickup": {"default": 0, "min": 0, "max": 1},
                      "grasp": {"default": 0, "min": 0, "max": 1},
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
                  "spriteMap": custom_sprite_map,
              }
          },
          {
              "component": "AvatarGrasp",
              "kwargs": {
                  "shape": GRASP_SHAPE,
                  "palette": color_palette,
                  "graspAction": "grasp",
                  # If multiple objects are at the same position then grasp them
                  # according to their layer in order `precedenceOrder`.
                  "precedenceOrder": ("appleLayer", "lowerPhysical",),
              }
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
                      "decrementRate": 1.0,
                  },
                  "amountInvisible": 6,
                  "amountGreen": 6,
                  "amountYellow": 6,
                  "amountRed": 1,
                  "costlyActions": ["move",],
              }
          },
          {
              "component": "StaminaObservation",
              "kwargs": {
                  "staminaComponent": "Stamina",
              }
          },
      ]
  }
  if _ENABLE_DEBUG_OBSERVATIONS:
    avatar_object["components"].append({
        "component": "LocationObserver",
        "kwargs": {"objectIsAvatar": True, "alsoReportOrientation": True},
    })

  return avatar_object


def create_avatar_objects(num_players: int,
                          max_stamina_bar_states: int = 19):
  """Returns list of avatar objects of length 'num_players'."""
  avatar_objects = []
  for player_idx in range(num_players):
    avatar_object = create_avatar_object(player_idx, TARGET_SPRITE_SELF,
                                         max_stamina_bar_states - 1)
    stamina_bar_objects = _create_stamina_overlay(player_idx,
                                                  max_stamina_bar_states)
    enter_obstacle = _create_enter_obstacle(player_idx)
    avatar_objects.append(avatar_object)
    avatar_objects.append(enter_obstacle)
    avatar_objects.extend(stamina_bar_objects)

  return avatar_objects


def _create_enter_obstacle(player_idx: int) -> Dict[str, Any]:
  # Lua is 1-indexed.
  lua_idx = player_idx + 1
  return {
      "name":
          "enter_obstacle",
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
                          {
                              "state": "obstacleLive",
                              "layer": "lowerPhysical",
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


def get_config():
  """Default configuration for training on the factory2d level."""
  config = config_dict.ConfigDict()

  # Specify the number of players to particate in each episode (optional).
  config.recommended_num_players = 12

  # Action set configuration.
  config.action_set = ACTION_SET
  # Observation format configuration.
  config.individual_observation_names = [
      "RGB",
      "READY_TO_SHOOT",
      "STAMINA",
  ]
  config.global_observation_names = [
      "WORLD.RGB",
  ]

  config.action_spec = specs.action(len(ACTION_SET))
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
      levelName="factory_of_the_commons",
      levelDirectory="meltingpot/lua/levels",
      numPlayers=num_players,
      maxEpisodeLengthFrames=5000,  # The maximum possible number of frames.
      spriteSize=8,
      topology="BOUNDED",  # Choose from ["BOUNDED", "TORUS"],
      simulation={
          "map": config.layout.ascii_map,
          "gameObjects": create_avatar_objects(num_players),
          "scene": create_scene(),
          "prefabs": PREFABS,
          "charPrefabMap": config.layout.char_prefab_map,
      },
  )
  return substrate_definition
