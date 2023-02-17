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
"""Configuration for Stag Hunt in the Matrix (two player, repeated version).

Example video: https://youtu.be/aDp_CArcb1Y

See _Running with Scissors in the Matrix_ for a general description of the
game dynamics. Here the payoff matrix represents the Stag Hunt game. `K = 2`
resources represent "stag" and "hare" pure strategies.

Players have a `5 x 5` observation window.

The episode has a chance of ending stochastically on every 100 step interval
after step 1000. This usually allows time for 8 or more interactions.
"""

from typing import Any, Dict, Mapping, Sequence, Tuple

from ml_collections import config_dict

from meltingpot.python.configs.substrates import the_matrix
from meltingpot.python.utils.substrates import colors
from meltingpot.python.utils.substrates import shapes
from meltingpot.python.utils.substrates import specs

# Warning: setting `_ENABLE_DEBUG_OBSERVATIONS = True` may cause slowdown.
_ENABLE_DEBUG_OBSERVATIONS = False

# The number of resources must match the (square) size of the matrix.
NUM_RESOURCES = 2

# This color is green.
RESOURCE1_COLOR = (30, 225, 185, 255)
RESOURCE1_HIGHLIGHT_COLOR = (98, 234, 206, 255)
RESOURCE1_COLOR_DATA = (RESOURCE1_COLOR, RESOURCE1_HIGHLIGHT_COLOR)
# This color is red.
RESOURCE2_COLOR = (225, 30, 70, 255)
RESOURCE2_HIGHLIGHT_COLOR = (234, 98, 126, 255)
RESOURCE2_COLOR_DATA = (RESOURCE2_COLOR, RESOURCE2_HIGHLIGHT_COLOR)

ASCII_MAP = """
WWWWWWWWWWWWWWWWWWWWWWW
Wn         n         nW
W  2WWW W  W  W WW2   W
W    W 11a W 222 W    W
Wn  WW 11a W a22 WW  nW
W      1aa 2 a22      W
W  2               2  W
Wn WW  WW2 n WW  WWW nW
W   2            2    W
W      22a 2 aa1      W
Wn   W 22a W a11 W   nW
W   2W 222 W a11 WW   W
W  WWWW W  W  W WWW2  W
Wn         n         nW
WWWWWWWWWWWWWWWWWWWWWWW
"""

_resource_names = [
    "resource_class1",
    "resource_class2",
]

# `prefab` determines which prefab game object to use for each `char` in the
# ascii map.
CHAR_PREFAB_MAP = {
    "a": {"type": "choice", "list": _resource_names},
    "1": _resource_names[0],
    "2": _resource_names[1],
    "n": "spawn_point",
    "W": "wall",
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
            "component": "Transform",
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
            "component": "BeamBlocker",
            "kwargs": {
                "beamType": "gameInteraction"
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
        {
            "component": "Transform",
        },
    ]
}

# PLAYER_COLOR_PALETTES is a list with each entry specifying the color to use
# for the player at the corresponding index.
NUM_PLAYERS_UPPER_BOUND = 8
PLAYER_COLOR_PALETTES = []
for idx in range(NUM_PLAYERS_UPPER_BOUND):
  PLAYER_COLOR_PALETTES.append(shapes.get_palette(colors.palette[idx]))

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

TARGET_SPRITE_SELF = {
    "name": "Self",
    "shape": shapes.CUTE_AVATAR,
    "palette": shapes.get_palette((50, 100, 200)),
    "noRotate": True,
}

TARGET_SPRITE_OTHER = {
    "name": "Other",
    "shape": shapes.CUTE_AVATAR,
    "palette": shapes.get_palette((200, 100, 50)),
    "noRotate": True,
}


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
              "component": "TheMatrix",
              "kwargs": {
                  # Prevent interaction before both interactors have collected
                  # at least one resource.
                  "disallowUnreadyInteractions": True,
                  "matrix": [
                      # row player chooses a row of this matrix.
                      # C  D
                      [4, 0],  # C
                      [2, 2],  # D
                  ],
                  "columnPlayerMatrix": [
                      # column player chooses a column of this matrix.
                      # C  D
                      [4, 2],  # C
                      [0, 2],  # D
                  ],
                  "resultIndicatorColorIntervals": [
                      # red       # yellow    # green     # blue      # violet
                      (0.0, 0.5), (0.5, 1.5), (1.5, 2.5), (2.5, 3.5), (3.5, 4.0)
                  ],
              }
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


def create_resource_prefab(
    resource_id: int,
    resource_shape: str,
    resource_palette: Dict[str, Tuple[int, int, int, int]]):
  """Creates resource prefab with provided resource_id, shape, and palette."""
  resource_name = "resource_class{}".format(resource_id)
  resource_prefab = {
      "name": resource_name,
      "components": [
          {
              "component": "StateManager",
              "kwargs": {
                  "initialState": resource_name,
                  "stateConfigs": [
                      {"state": resource_name + "_wait",
                       "groups": ["resourceWaits"]},
                      {"state": resource_name,
                       "layer": "lowerPhysical",
                       "sprite": resource_name + "_sprite"},
                  ]
              },
          },
          {
              "component": "Transform",
          },
          {
              "component": "Appearance",
              "kwargs": {
                  "renderMode": "ascii_shape",
                  "spriteNames": [resource_name + "_sprite"],
                  "spriteShapes": [resource_shape],
                  "palettes": [resource_palette],
                  "noRotates": [True]
              },
          },
          {
              "component": "Resource",
              "kwargs": {
                  "resourceClass": resource_id,
                  "visibleType": resource_name,
                  "waitState": resource_name + "_wait",
                  "regenerationRate": 0.02,
                  "regenerationDelay": 10,
              },
          },
          {
              "component": "Destroyable",
              "kwargs": {
                  "waitState": resource_name + "_wait",
                  # It is possible to destroy resources but takes concerted
                  # effort to do so by zapping them `initialHealth` times.
                  "initialHealth": 3,
              },
          },
      ]
  }
  return resource_prefab


def create_avatar_object(
    player_idx: int,
    all_source_sprite_names: Sequence[str],
    target_sprite_self: Dict[str, Any],
    target_sprite_other: Dict[str, Any],
    turn_off_default_reward: bool = False) -> Dict[str, Any]:
  """Create an avatar object given self vs other sprite data."""
  # Lua is 1-indexed.
  lua_index = player_idx + 1

  # Setup the self vs other sprite mapping.
  source_sprite_self = "Avatar" + str(lua_index)
  custom_sprite_map = {source_sprite_self: target_sprite_self["name"]}
  for name in all_source_sprite_names:
    if name != source_sprite_self:
      custom_sprite_map[name] = target_sprite_other["name"]

  live_state_name = "player{}".format(lua_index)
  avatar_object = {
      "name": "avatar",
      "components": [
          {
              "component": "StateManager",
              "kwargs": {
                  "initialState": live_state_name,
                  "stateConfigs": [
                      {"state": live_state_name,
                       "layer": "upperPhysical",
                       "sprite": source_sprite_self,
                       "contact": "avatar",
                       "groups": ["players"]},

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
                  "renderMode": "colored_square",
                  "spriteNames": [source_sprite_self],
                  # A white square should never be displayed. It will always be
                  # remapped since this is self vs other observation mode.
                  "spriteRGBColors": [(255, 255, 255, 255)],
              }
          },
          {
              "component": "AdditionalSprites",
              "kwargs": {
                  "renderMode": "ascii_shape",
                  "customSpriteNames": [target_sprite_self["name"],
                                        target_sprite_other["name"]],
                  "customSpriteShapes": [target_sprite_self["shape"],
                                         target_sprite_other["shape"]],
                  "customPalettes": [target_sprite_self["palette"],
                                     target_sprite_other["palette"]],
                  "customNoRotates": [target_sprite_self["noRotate"],
                                      target_sprite_other["noRotate"]],
              }
          },
          {
              "component": "Avatar",
              "kwargs": {
                  "index": lua_index,
                  "aliveState": live_state_name,
                  "waitState": "playerWait",
                  "speed": 1.0,
                  "spawnGroup": "spawnPoints",
                  "actionOrder": ["move", "turn", "interact"],
                  "actionSpec": {
                      "move": {"default": 0, "min": 0, "max": len(_COMPASS)},
                      "turn": {"default": 0, "min": -1, "max": 1},
                      "interact": {"default": 0, "min": 0, "max": 1},
                  },
                  "view": {
                      "left": 2,
                      "right": 2,
                      "forward": 3,
                      "backward": 1,
                      "centered": False
                  },
                  "spriteMap": custom_sprite_map,
                  # The following kwarg makes it possible to get rewarded even
                  # on frames when an avatar is "dead". It is needed for in the
                  # matrix games in order to correctly handle the case of two
                  # players getting hit simultaneously by the same beam.
                  "skipWaitStateRewards": False,
              }
          },
          {
              "component": "GameInteractionZapper",
              "kwargs": {
                  "cooldownTime": 2,
                  "beamLength": 3,
                  "beamRadius": 1,
                  "framesTillRespawn": 5,
                  "numResources": NUM_RESOURCES,
                  "endEpisodeOnFirstInteraction": False,
                  # Reset both players' inventories after each interaction.
                  "reset_winner_inventory": True,
                  "reset_loser_inventory": True,
                  # Both players get removed after each interaction.
                  "losingPlayerDies": True,
                  "winningPlayerDies": True,
                  # `freezeOnInteraction` is the number of frames to display the
                  # interaction result indicator, freeze, and delay delivering
                  # all results of interacting.
                  "freezeOnInteraction": 16,
              }
          },
          {
              "component": "ReadyToShootObservation",
              "kwargs": {
                  "zapperComponent": "GameInteractionZapper",
              }
          },
          {
              "component": "InventoryObserver",
              "kwargs": {
              }
          },
          {
              "component": "SpawnResourcesWhenAllPlayersZapped",
          },
          {
              "component": "Taste",
              "kwargs": {
                  "mostTastyResourceClass": -1,  # -1 indicates no preference.
                  # No resource is most tasty when mostTastyResourceClass == -1.
                  "mostTastyReward": 0.1,
              }
          },
          {
              "component": "InteractionTaste",
              "kwargs": {
                  "mostTastyResourceClass": -1,  # -1 indicates no preference.
                  "zeroDefaultInteractionReward": turn_off_default_reward,
                  "extraReward": 1.0,
              }
          },
          {
              "component": "AvatarMetricReporter",
              "kwargs": {
                  "metrics": [
                      {
                          # Report the inventories of both players involved in
                          # an interaction on this frame formatted as
                          # (self inventory, partner inventory).
                          "name": "INTERACTION_INVENTORIES",
                          "type": "tensor.DoubleTensor",
                          "shape": (2, NUM_RESOURCES),
                          "component": "GameInteractionZapper",
                          "variable": "latest_interaction_inventories",
                      },
                      *the_matrix.get_cumulant_metric_configs(NUM_RESOURCES),
                  ]
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


def create_prefabs():
  """Returns a dictionary mapping names to template game objects."""
  prefabs = {
      "wall": WALL,
      "spawn_point": SPAWN_POINT,
  }
  prefabs["resource_class1"] = create_resource_prefab(
      1, shapes.BUTTON, {"*": RESOURCE1_COLOR_DATA[0],
                         "#": RESOURCE1_COLOR_DATA[1],
                         "x": (0, 0, 0, 0)})
  prefabs["resource_class2"] = create_resource_prefab(
      2, shapes.BUTTON, {"*": RESOURCE2_COLOR_DATA[0],
                         "#": RESOURCE2_COLOR_DATA[1],
                         "x": (0, 0, 0, 0)})
  return prefabs


def get_all_source_sprite_names(num_players):
  all_source_sprite_names = []
  for player_idx in range(0, num_players):
    # Lua is 1-indexed.
    lua_index = player_idx + 1
    all_source_sprite_names.append("Avatar" + str(lua_index))

  return all_source_sprite_names


def create_avatar_objects(num_players,
                          turn_off_default_reward: bool = False):
  """Returns list of avatar objects of length 'num_players'."""
  all_source_sprite_names = get_all_source_sprite_names(num_players)
  avatar_objects = []
  for player_idx in range(0, num_players):
    game_object = create_avatar_object(
        player_idx,
        all_source_sprite_names,
        TARGET_SPRITE_SELF,
        TARGET_SPRITE_OTHER,
        turn_off_default_reward=turn_off_default_reward)
    readiness_marker = the_matrix.create_ready_to_interact_marker(player_idx)
    avatar_objects.append(game_object)
    avatar_objects.append(readiness_marker)

  return avatar_objects


def create_world_sprite_map(
    num_players: int, target_sprite_other: Dict[str, Any]) -> Dict[str, str]:
  all_source_sprite_names = get_all_source_sprite_names(num_players)
  world_sprite_map = {}
  for name in all_source_sprite_names:
    world_sprite_map[name] = target_sprite_other["name"]

  return world_sprite_map


def get_config():
  """Default configuration."""
  config = config_dict.ConfigDict()

  # Other parameters that are useful to override in training config files.
  config.turn_off_default_reward = False

  # Action set configuration.
  config.action_set = ACTION_SET
  # Observation format configuration.
  config.individual_observation_names = [
      "RGB",
      "INVENTORY",
      "READY_TO_SHOOT",
      # Debug only (do not use the following observations in policies).
      "INTERACTION_INVENTORIES",
  ]
  config.global_observation_names = [
      "WORLD.RGB",
  ]

  # The specs of the environment (from a single-agent perspective).
  config.action_spec = specs.action(len(ACTION_SET))
  config.timestep_spec = specs.timestep({
      "RGB": specs.rgb(40, 40),
      "INVENTORY": specs.inventory(2),
      "READY_TO_SHOOT": specs.OBSERVATION["READY_TO_SHOOT"],
      # Debug only (do not use the following observations in policies).
      "INTERACTION_INVENTORIES": specs.interaction_inventories(2),
      "WORLD.RGB": specs.rgb(120, 184),
  })

  # The roles assigned to each player.
  config.valid_roles = frozenset({"default"})
  config.default_player_roles = ("default",) * 2

  return config


def build(
    roles: Sequence[str],
    config: config_dict.ConfigDict,
) -> Mapping[str, Any]:
  """Build substrate definition given roles."""
  del config
  num_players = len(roles)
  # Build the rest of the substrate definition.
  substrate_definition = dict(
      levelName="the_matrix",
      levelDirectory="meltingpot/lua/levels",
      numPlayers=num_players,
      maxEpisodeLengthFrames=2500,  # The maximum possible number of frames.
      spriteSize=8,
      topology="BOUNDED",  # Choose from ["BOUNDED", "TORUS"],
      simulation={
          "map": ASCII_MAP,
          "gameObjects": create_avatar_objects(num_players=num_players),
          "scene": create_scene(),
          "prefabs": create_prefabs(),
          "charPrefabMap": CHAR_PREFAB_MAP,
          # worldSpriteMap is needed to make the global view used in videos be
          # be informative in cases where individual avatar views have had
          # sprites remapped to one another (example: self vs other mode).
          "worldSpriteMap": create_world_sprite_map(num_players,
                                                    TARGET_SPRITE_OTHER),
      }
  )
  return substrate_definition
