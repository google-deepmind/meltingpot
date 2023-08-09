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
"""Common configuration for all territory_* substrates.

See _Territory: Open_ for the general description of the mechanics at play in
this substrate.
"""

from typing import Any, Mapping, Sequence

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


def get_marking_palette(alpha: float) -> Mapping[str, Sequence[int]]:
  alpha_uint8 = int(alpha * 255)
  assert alpha_uint8 >= 0.0 and alpha_uint8 <= 255, "Color value out of range."
  return {"x": shapes.ALPHA, "o": (0, 0, 0, alpha_uint8)}

FLOOR = {
    "name": "floor",
    "components": [
        {
            "component": "StateManager",
            "kwargs": {
                "initialState": "floor",
                "stateConfigs": [{
                    "state": "floor",
                    "layer": "background",
                    "sprite": "Floor",
                }],
            }
        },
        {
            "component": "Appearance",
            "kwargs": {
                "renderMode": "ascii_shape",
                "spriteNames": ["Floor",],
                "spriteShapes": [shapes.GRAINY_FLOOR],
                "palettes": [{
                    "*": (27, 22, 20, 255),
                    "+": (23, 17, 15, 255),
                }],
                "noRotates": [True]
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
                    "layer": "upperPhysical",
                    "sprite": "Wall",
                }],
            }
        },
        {
            "component": "Appearance",
            "kwargs": {
                "renderMode": "ascii_shape",
                "spriteNames": ["Wall",],
                "spriteShapes": [shapes.FILL],
                "palettes": [{"i": (61, 57, 55, 255)}],
                "noRotates": [False]
            }
        },
        {
            "component": "Transform",
        },
        {
            "component": "AllBeamBlocker",
            "kwargs": {}
        },
    ]
}

WALL_HIGHLIGHT_NW = {
    "name": "nw_highlight",
    "components": [
        {
            "component": "StateManager",
            "kwargs": {
                "initialState": "nw_highlight",
                "stateConfigs": [{
                    "state": "nw_highlight",
                    "layer": "overlay",
                    "sprite": "NWHighlight",
                }],
            }
        },
        {
            "component": "Appearance",
            "kwargs": {
                "renderMode": "ascii_shape",
                "spriteNames": ["NWHighlight",],
                "spriteShapes": [shapes.NW_HIGHLIGHT],
                "palettes": [shapes.HIGHLIGHT_PALETTE],
                "noRotates": [False]
            }
        },
        {
            "component": "Transform",
        },
    ]
}

WALL_HIGHLIGHT_E_W = {
    "name": "e_w_highlight",
    "components": [
        {
            "component": "StateManager",
            "kwargs": {
                "initialState": "e_w_highlight",
                "stateConfigs": [{
                    "state": "e_w_highlight",
                    "layer": "overlay",
                    "sprite": "EWHighlight",
                }],
            }
        },
        {
            "component": "Appearance",
            "kwargs": {
                "renderMode": "ascii_shape",
                "spriteNames": ["EWHighlight",],
                "spriteShapes": [shapes.E_W_HIGHLIGHT],
                "palettes": [shapes.HIGHLIGHT_PALETTE],
                "noRotates": [False]
            }
        },
        {
            "component": "Transform",
        },
    ]
}

WALL_HIGHLIGHT_N_S = {
    "name": "n_s_highlight",
    "components": [
        {
            "component": "StateManager",
            "kwargs": {
                "initialState": "n_s_highlight",
                "stateConfigs": [{
                    "state": "n_s_highlight",
                    "layer": "overlay",
                    "sprite": "NSHighlight",
                }],
            }
        },
        {
            "component": "Appearance",
            "kwargs": {
                "renderMode": "ascii_shape",
                "spriteNames": ["NSHighlight",],
                "spriteShapes": [shapes.N_S_HIGHLIGHT],
                "palettes": [shapes.HIGHLIGHT_PALETTE],
                "noRotates": [False]
            }
        },
        {
            "component": "Transform",
        },
    ]
}

WALL_HIGHLIGHT_NE = {
    "name": "ne_highlight",
    "components": [
        {
            "component": "StateManager",
            "kwargs": {
                "initialState": "ne_highlight",
                "stateConfigs": [{
                    "state": "ne_highlight",
                    "layer": "overlay",
                    "sprite": "NEHighlight",
                }],
            }
        },
        {
            "component": "Appearance",
            "kwargs": {
                "renderMode": "ascii_shape",
                "spriteNames": ["NEHighlight",],
                "spriteShapes": [shapes.NE_HIGHLIGHT],
                "palettes": [shapes.HIGHLIGHT_PALETTE],
                "noRotates": [False]
            }
        },
        {
            "component": "Transform",
        },
    ]
}

WALL_HIGHLIGHT_SE = {
    "name": "se_highlight",
    "components": [
        {
            "component": "StateManager",
            "kwargs": {
                "initialState": "se_highlight",
                "stateConfigs": [{
                    "state": "se_highlight",
                    "layer": "overlay",
                    "sprite": "SEHighlight",
                }],
            }
        },
        {
            "component": "Appearance",
            "kwargs": {
                "renderMode": "ascii_shape",
                "spriteNames": ["SEHighlight",],
                "spriteShapes": [shapes.SE_HIGHLIGHT],
                "palettes": [shapes.HIGHLIGHT_PALETTE],
                "noRotates": [False]
            }
        },
        {
            "component": "Transform",
        },
    ]
}

WALL_HIGHLIGHT_SW = {
    "name": "sw_highlight",
    "components": [
        {
            "component": "StateManager",
            "kwargs": {
                "initialState": "sw_highlight",
                "stateConfigs": [{
                    "state": "sw_highlight",
                    "layer": "overlay",
                    "sprite": "SWHighlight",
                }],
            }
        },
        {
            "component": "Appearance",
            "kwargs": {
                "renderMode": "ascii_shape",
                "spriteNames": ["SWHighlight",],
                "spriteShapes": [shapes.SW_HIGHLIGHT],
                "palettes": [shapes.HIGHLIGHT_PALETTE],
                "noRotates": [False]
            }
        },
        {
            "component": "Transform",
        },
    ]
}

SPAWN_POINT = {
    "name": "spawn_point",
    "components": [
        {
            "component": "StateManager",
            "kwargs": {
                "initialState": "playerSpawnPoint",
                "stateConfigs": [{
                    "state": "playerSpawnPoint",
                    "layer": "logic",
                    "groups": ["spawnPoints"],
                }],
            }
        },
        {
            "component": "Appearance",
            "kwargs": {
                "renderMode": "invisible",
                "spriteNames": [],
                "spriteRGBColors": []
            }
        },
        {
            "component": "Transform",
        },
    ]
}


def get_dry_painted_wall_palette(base_color: shapes.Color
                                 ) -> Mapping[str, shapes.ColorRGBA]:
  return {
      "*": shapes.scale_color(base_color, 0.75, 200),
      "#": shapes.scale_color(base_color, 0.90, 150),
  }


def get_brush_palette(
    base_color: shapes.Color) -> Mapping[str, shapes.ColorRGBA]:
  return {
      "*": base_color + (255,),
      "&": shapes.scale_color(base_color, 0.75, 255),
      "o": shapes.scale_color(base_color, 0.55, 255),
      "O": (70, 70, 70, 255),
      "-": (143, 96, 74, 255),
      "+": (117, 79, 61, 255),
      "k": (199, 176, 135, 255),
      "x": shapes.ALPHA,
  }

PLAYER_COLOR_PALETTES = []
BRUSH_PALETTES = []
for human_readable_color in colors.human_readable:
  PLAYER_COLOR_PALETTES.append(shapes.get_palette(human_readable_color))
  BRUSH_PALETTES.append(get_brush_palette(human_readable_color))


def create_resource(num_players: int) -> PrefabConfig:
  """Configure the prefab to use for all resource objects."""
  # Setup unique states corresponding to each player who can claim the resource.
  claim_state_configs = []
  claim_sprite_names = []
  claim_sprite_rgb_colors = []
  for player_idx in range(num_players):
    lua_player_idx = player_idx + 1
    player_color = colors.human_readable[player_idx]
    wet_sprite_name = "Color" + str(lua_player_idx) + "ResourceSprite"
    claim_state_configs.append({
        "state": "claimed_by_" + str(lua_player_idx),
        "layer": "upperPhysical",
        "sprite": wet_sprite_name,
        "groups": ["claimedResources"]
    })
    claim_sprite_names.append(wet_sprite_name)
    # Use alpha channel to make transparent version of claiming agent's color.
    wet_paint_color = player_color + (75,)
    claim_sprite_rgb_colors.append(wet_paint_color)

  prefab = {
      "name": "resource",
      "components": [
          {
              "component": "StateManager",
              "kwargs": {
                  "initialState": "unclaimed",
                  "stateConfigs": [
                      {"state": "unclaimed",
                       "layer": "upperPhysical",
                       "sprite": "UnclaimedResourceSprite"},
                      {"state": "destroyed"},
                  ] + claim_state_configs,
              }
          },
          {
              "component": "Appearance",
              "kwargs": {
                  "renderMode": "colored_square",
                  "spriteNames": claim_sprite_names,
                  "spriteRGBColors": claim_sprite_rgb_colors
              }
          },
          {
              "component": "Transform",
          },
          {
              "component": "Resource",
              "kwargs": {
                  "initialHealth": 2,
                  "destroyedState": "destroyed",
                  "reward": 1.0,
                  "rewardRate": 0.01,
                  "rewardDelay": 25,
                  "delayTillSelfRepair": 15,
                  "selfRepairProbability": 0.1,
              }
          },
      ]
  }
  return prefab


def create_resource_texture() -> PrefabConfig:
  """Configure the background texture for a resource. It looks like a wall."""
  prefab = {
      "name": "resource_texture",
      "components": [
          {
              "component": "StateManager",
              "kwargs": {
                  "initialState": "unclaimed",
                  "stateConfigs": [
                      {"state": "unclaimed",
                       "layer": "lowerPhysical",
                       "sprite": "UnclaimedResourceSprite"},
                      {"state": "destroyed"},
                  ],
              }
          },
          {
              "component": "Appearance",
              "kwargs": {
                  "renderMode": "ascii_shape",
                  "spriteNames": ["UnclaimedResourceSprite",],
                  "spriteShapes": [shapes.WALL],
                  "palettes": [{"*": (61, 61, 61, 255),
                                "#": (80, 80, 80, 255)}],
                  "noRotates": [True]
              }
          },
          {
              "component": "Transform",
          },
      ]
  }
  return prefab


def create_reward_indicator(num_players) -> PrefabConfig:
  """Configure object indicating if a resource is currently providing reward."""
  # Setup unique states corresponding to each player who can claim the resource.
  claim_state_configs = []
  claim_sprite_names = []
  claim_sprite_shapes = []
  claim_palettes = []
  claim_no_rotates = []
  for player_idx in range(num_players):
    lua_player_idx = player_idx + 1
    player_color = colors.human_readable[player_idx]
    dry_sprite_name = "Color" + str(lua_player_idx) + "DryPaintSprite"
    claim_state_configs.append({
        "state": "dry_claimed_by_" + str(lua_player_idx),
        "layer": "overlay",
        "sprite": dry_sprite_name,
    })
    claim_sprite_names.append(dry_sprite_name)
    claim_sprite_shapes.append(shapes.WALL)
    claim_palettes.append(get_dry_painted_wall_palette(player_color))
    claim_no_rotates.append(True)
  prefab = {
      "name": "reward_indicator",
      "components": [
          {
              "component": "StateManager",
              "kwargs": {
                  "initialState": "inactive",
                  "stateConfigs": [
                      {"state": "inactive"},
                  ] + claim_state_configs,
              }
          },
          {
              "component": "Appearance",
              "kwargs": {
                  "renderMode": "ascii_shape",
                  "spriteNames": claim_sprite_names,
                  "spriteShapes": claim_sprite_shapes,
                  "palettes": claim_palettes,
                  "noRotates": claim_no_rotates
              }
          },
          {
              "component": "Transform",
          },
          {
              "component": "RewardIndicator",
          },
      ]
  }
  return prefab


def create_damage_indicator() -> PrefabConfig:
  """Configure the object indicating whether or not a resource is damaged."""
  damaged_resource_sprite = """
      ,,bb,,,,
      ,,bb,bb,
      ,,,b,,b,
      ,,,b,,,,
      ,,,b,,,b
      ,,,bb,,b
      ,,,bb,,b
      b,,,b,,,
  """
  damaged_resource_palette = {",": shapes.ALPHA, "b": shapes.BLACK}
  prefab = {
      "name": "damage_indicator",
      "components": [
          {
              "component": "StateManager",
              "kwargs": {
                  "initialState": "inactive",
                  "stateConfigs": [
                      {"state": "inactive",
                       "layer": "superDirectionIndicatorLayer"},
                      {"state": "damaged",
                       "layer": "superDirectionIndicatorLayer",
                       "sprite": "DamagedResource"},
                  ],
              }
          },
          {
              "component": "Appearance",
              "kwargs": {
                  "renderMode": "ascii_shape",
                  "spriteNames": ["DamagedResource"],
                  "spriteShapes": [damaged_resource_sprite],
                  "palettes": [damaged_resource_palette],
                  "noRotates": [True]
              }
          },
          {
              "component": "Transform",
          },
      ]
  }
  return prefab


def create_prefabs(num_players: int):
  """Returns the prefabs dictionary."""
  prefabs = {
      "spawn_point": SPAWN_POINT,
      "floor": FLOOR,
      "wall": WALL,
      "wall_highlight_nw": WALL_HIGHLIGHT_NW,
      "wall_highlight_e_w": WALL_HIGHLIGHT_E_W,
      "wall_highlight_n_s": WALL_HIGHLIGHT_N_S,
      "wall_highlight_ne": WALL_HIGHLIGHT_NE,
      "wall_highlight_se": WALL_HIGHLIGHT_SE,
      "wall_highlight_sw": WALL_HIGHLIGHT_SW,
      "resource": create_resource(num_players=num_players),
      "resource_texture": create_resource_texture(),
      "reward_indicator": create_reward_indicator(num_players),
      "damage_indicator": create_damage_indicator(),
  }
  return prefabs


# Primitive action components.
# pylint: disable=bad-whitespace
# pyformat: disable
NOOP       = {"move": 0, "turn":  0, "fireZap": 0, "fireClaim": 0}
FORWARD    = {"move": 1, "turn":  0, "fireZap": 0, "fireClaim": 0}
STEP_RIGHT = {"move": 2, "turn":  0, "fireZap": 0, "fireClaim": 0}
BACKWARD   = {"move": 3, "turn":  0, "fireZap": 0, "fireClaim": 0}
STEP_LEFT  = {"move": 4, "turn":  0, "fireZap": 0, "fireClaim": 0}
TURN_LEFT  = {"move": 0, "turn": -1, "fireZap": 0, "fireClaim": 0}
TURN_RIGHT = {"move": 0, "turn":  1, "fireZap": 0, "fireClaim": 0}
FIRE_ZAP   = {"move": 0, "turn":  0, "fireZap": 1, "fireClaim": 0}
FIRE_CLAIM = {"move": 0, "turn":  0, "fireZap": 0, "fireClaim": 1}
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
    FIRE_CLAIM
)


# The Scene object is a non-physical object, its components implement global
# logic.
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
          {
              "component": "Transform",
          },
          {
              "component": "StochasticIntervalEpisodeEnding",
              "kwargs": {
                  "minimumFramesPerEpisode": 1000,
                  "intervalLength": 100,  # Set equal to unroll length.
                  "probabilityTerminationPerInterval": 0.2
              }
          }
      ]
  }
  return scene


def create_avatar_object(player_idx: int) -> Mapping[str, Any]:
  """Create an avatar object that always sees itself as blue."""
  # Lua is 1-indexed.
  lua_index = player_idx + 1

  color_palette = PLAYER_COLOR_PALETTES[player_idx]
  paintbrush_palette = BRUSH_PALETTES[player_idx]
  live_state_name = "player{}".format(lua_index)
  avatar_sprite_name = "avatarSprite{}".format(lua_index)
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
                       "sprite": avatar_sprite_name,
                       "contact": "avatar",
                       "groups": ["players"]},

                      # Player wait state used when they have been zapped out.
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
                  "spriteNames": [avatar_sprite_name],
                  "spriteShapes": [shapes.CUTE_AVATAR_HOLDING_PAINTBRUSH],
                  "palettes": [{**color_palette, **paintbrush_palette}],
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
                                  "fireZap",
                                  "fireClaim"],
                  "actionSpec": {
                      "move": {"default": 0, "min": 0, "max": len(_COMPASS)},
                      "turn": {"default": 0, "min": -1, "max": 1},
                      "fireZap": {"default": 0, "min": 0, "max": 1},
                      "fireClaim": {"default": 0, "min": 0, "max": 1},
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
              "component": "Paintbrush",
              "kwargs": {
                  "shape": shapes.PAINTBRUSH,
                  "palette": paintbrush_palette,
                  "playerIndex": lua_index,
              }
          },
          {
              "component": "Zapper",
              "kwargs": {
                  "cooldownTime": 4,
                  "beamLength": 2,
                  "beamRadius": 1,
                  "framesTillRespawn": 1e6,  # Effectively never respawn.
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
              "component": "ResourceClaimer",
              "kwargs": {
                  "color": color_palette["*"],
                  "playerIndex": lua_index,
                  "beamLength": 2,
                  "beamRadius": 0,
                  "beamWait": 0,
              }
          },
          {
              "component": "Taste",
              "kwargs": {
                  "role": "none",
                  "rewardAmount": 1.0,
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


def create_avatar_and_associated_objects(num_players):
  """Returns list of avatars and their associated marking objects."""
  avatar_objects = []
  additional_objects = []
  for player_idx in range(0, num_players):
    game_object = create_avatar_object(player_idx)
    avatar_objects.append(game_object)

    marking_object = create_marking_overlay(player_idx)
    additional_objects.append(marking_object)

  return avatar_objects + additional_objects


def get_config():
  """Default configuration."""
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
  config.timestep_spec = specs.timestep({
      "RGB": specs.OBSERVATION["RGB"],
      "READY_TO_SHOOT": specs.OBSERVATION["READY_TO_SHOOT"],
      # Debug only (do not use the following observations in policies).
      "WORLD.RGB": specs.rgb(120, 184),
  })

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
      levelName="territory",
      levelDirectory="meltingpot/lua/levels",
      numPlayers=num_players,
      # Define upper bound of episode length since episodes end stochastically.
      maxEpisodeLengthFrames=5000,
      spriteSize=8,
      topology=config.layout.topology,  # Choose from ["BOUNDED", "TORUS"],
      simulation={
          "map": config.layout.ascii_map,
          "gameObjects": create_avatar_and_associated_objects(num_players),
          "scene": create_scene(),
          "prefabs": create_prefabs(num_players),
          "charPrefabMap": config.layout.char_prefab_map,
      },
  )
  return substrate_definition
