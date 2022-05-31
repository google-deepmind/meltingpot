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
"""Configuration for Territory: Open.

Example video: https://youtu.be/3hB8lABa6nI

Players can claim a resource in two ways: (1) by touching it, and (2) by using a
"claiming beam", different from the zapping beam, which they also have.
Claimed resources are colored in the unique color of the player that claimed
them. Unclaimed resources are gray. Players cannot walk through resources, they
are like walls.

Once a resource has been claimed a countdown begins. After 100 timesteps, the
claimed resource becomes active. This is visualized by a white and gray plus
sign appearing on top. Active resources provide reward stochastically to the
player that claimed them at a rate of 0.01 per timestep. Thus the more resources
a player claims and can hold until they become active, the more reward they
obtain.

The claiming beam is of length 2. It can color through a resource to
simultaneously color a second resource on the other side. If two players stand
on opposite sides of a wall of resources of width 2 and one player claims all
the way across to the other side (closer to the other player than themselves)
then the player on the other side might reasonably perceive that as a somewhat
aggressive action. Less aggressive of course than the other option both players
have: using their zapping beam. If any resource is zapped twice then it gets
permanently destroyed. It no longer functions as a wall or a resource, allowing
players to pass through.

Like resources, when players are hit by a zapping beam they also get removed
from the game and never regenerate. Once a player has been zapped out it is
gone. All resources it claimed are immediately returned to the unclaimed state.
"""

from typing import Any, Dict

from ml_collections import config_dict
from meltingpot.python.utils.substrates import colors
from meltingpot.python.utils.substrates import game_object_utils
from meltingpot.python.utils.substrates import shapes
from meltingpot.python.utils.substrates import specs

_COMPASS = ["N", "E", "S", "W"]

# This number just needs to be greater than the number of players.
MAX_ALLOWED_NUM_PLAYERS = 10

DEFAULT_ASCII_MAP = """
WWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW
W  RRRRR  RR  RR      RR      RR      W
W     RR      RR      RR              W
W     RR      RR                      W
W RR  RR      RR          R   RR   RR W
W     RR      RR          R   RR      W
W     RR          RRRR    R           W
W  RR RR                  R           W
W     RR       RR         R           W
W     RRRR                     RR     W
W                    RR               W
W                                     W
W  RRRR   RRRRRR           RR    R    W
W                                R    W
W                RR                P  W
W    RR                RR       P     W
W         RR                     P  P W
W                           P  P      W
W                             P   P   W
W  P    P   P  P   P    P P  P  P  P  W
W                                     W
W                                     W
WWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW
"""

# `prefab` determines which prefab game object to use for each `char` in the
# ascii map.
CHAR_PREFAB_MAP = {
    "P": "spawn_point",
    "W": "wall",
    "R": {"type": "all", "list": ["resource", "reward_indicator"]},
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
                "spriteShapes": [shapes.WALL],
                "palettes": [{"*": (95, 95, 95, 255),
                              "&": (100, 100, 100, 255),
                              "@": (109, 109, 109, 255),
                              "#": (152, 152, 152, 255)}],
                "noRotates": [True]
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
            "component": "AllBeamBlocker",
            "kwargs": {}
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
            "kwargs": {
                "position": (0, 0),
                "orientation": "N"
            }
        },
    ]
}

RESOURCE = {
    "name": "resource",
    "components": [
        {
            "component": "StateManager",
            "kwargs": {
                "initialState": "unclaimed",
                "stateConfigs": [
                    {"state": "unclaimed",
                     "layer": "upperPhysical",
                     "sprite": "UnclaimedResourceSprite",
                     "groups": ["unclaimedResources"]},
                    {"state": "destroyed"},
                ],
            }
        },
        {
            "component": "Appearance",
            "kwargs": {
                "spriteNames": ["UnclaimedResourceSprite"],
                # This color is grey.
                "spriteRGBColors": [(64, 64, 64, 255)]
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
            "component": "Resource",
            "kwargs": {
                "initialHealth": 2,
                "destroyedState": "destroyed",
                "reward": 1.0,
                "rewardRate": 0.01,
                "rewardDelay": 100
            }
        },
    ]
}

REWARD_INDICATOR = {
    "name": "reward_indicator",
    "components": [
        {
            "component": "StateManager",
            "kwargs": {
                "initialState": "inactive",
                "stateConfigs": [
                    {"state": "active",
                     "layer": "overlay",
                     "sprite": "ActivelyRewardingResource"},
                    {"state": "inactive"},
                ],
            }
        },
        {
            "component": "Appearance",
            "kwargs": {
                "spriteNames": ["ActivelyRewardingResource",],
                "renderMode": "ascii_shape",
                "spriteShapes": [shapes.PLUS_IN_BOX],
                "palettes": [{"*": (86, 86, 86, 65),
                              "#": (202, 202, 202, 105),
                              "@": (128, 128, 128, 135),
                              "x": (0, 0, 0, 0)}],
                "noRotates": [True]
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
            "component": "RewardIndicator",
            "kwargs": {
            }
        },
    ]
}

# PLAYER_COLOR_PALETTES is a list with each entry specifying the color to use
# for the player at the corresponding index.
PLAYER_COLOR_PALETTES = []
for i in range(MAX_ALLOWED_NUM_PLAYERS):
  PLAYER_COLOR_PALETTES.append(shapes.get_palette(colors.palette[i]))

# Set up player-specific settings for resources.
for j, color in enumerate(colors.palette[:MAX_ALLOWED_NUM_PLAYERS]):
  sprite_name = "Color" + str(j + 1) + "ResourceSprite"
  game_object_utils.get_first_named_component(
      RESOURCE,
      "StateManager")["kwargs"]["stateConfigs"].append({
          "state": "claimed_by_" + str(j + 1),
          "layer": "upperPhysical",
          "sprite": sprite_name,
          "groups": ["claimedResources"]
      })
  game_object_utils.get_first_named_component(
      RESOURCE,
      "Appearance")["kwargs"]["spriteNames"].append(sprite_name)
  game_object_utils.get_first_named_component(
      RESOURCE,
      "Appearance")["kwargs"]["spriteRGBColors"].append(color)

# PREFABS is a dictionary mapping names to template game objects that can
# be cloned and placed in multiple locations accoring to an ascii map.
PREFABS = {
    "wall": WALL,
    "spawn_point": SPAWN_POINT,
    "resource": RESOURCE,
    "reward_indicator": REWARD_INDICATOR,
}

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
              "kwargs": {
                  "position": (0, 0),
                  "orientation": "N"
              },
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


def create_avatar_object(player_idx: int) -> Dict[str, Any]:
  """Create an avatar object that always sees itself as blue."""
  # Lua is 1-indexed.
  lua_index = player_idx + 1

  color_palette = PLAYER_COLOR_PALETTES[player_idx]
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
              "kwargs": {
                  "position": (0, 0),
                  "orientation": "N"
              }
          },
          {
              "component": "Appearance",
              "kwargs": {
                  "renderMode": "ascii_shape",
                  "spriteNames": [avatar_sprite_name],
                  "spriteShapes": [shapes.CUTE_AVATAR],
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
              "component": "AvatarDirectionIndicator",
              # We do not normally use direction indicators for the MAGI suite,
              # but we do use them for territory because they function to claim
              # any resources they contact.
              "kwargs": {"color": (202, 202, 202, 50)}
          },
          {
              "component": "Zapper",
              "kwargs": {
                  "cooldownTime": 2,
                  "beamLength": 3,
                  "beamRadius": 1,
                  "framesTillRespawn": 1e6,  # Effectively never respawn.
                  "penaltyForBeingZapped": 0,
                  "rewardForZapping": 0,
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
              "component": "LocationObserver",
              "kwargs": {
                  "objectIsAvatar": True,
                  "alsoReportOrientation": True
              }
          },
          {
              "component": "Taste",
              "kwargs": {
                  "role": "none",
                  "rewardAmount": 1.0,
                  "firstClaimRewardMultiplier": 10.0,
              }
          },
      ]
  }

  return avatar_object


def create_avatar_objects(num_players):
  """Returns list of avatar objects of length 'num_players'."""
  avatar_objects = []
  for player_idx in range(0, num_players):
    game_object = create_avatar_object(player_idx)
    avatar_objects.append(game_object)

  return avatar_objects


def create_lab2d_settings(num_players: int) -> Dict[str, Any]:
  """Returns the lab2d settings."""
  lab2d_settings = {
      "levelName": "territory",
      "levelDirectory":
          "meltingpot/lua/levels",
      "numPlayers": num_players,
      # Define upper bound of episode length since episodes end stochastically.
      "maxEpisodeLengthFrames": 2000,
      "spriteSize": 8,
      "topology": "BOUNDED",  # Choose from ["BOUNDED", "TORUS"],
      "simulation": {
          "map": DEFAULT_ASCII_MAP,
          "gameObjects": create_avatar_objects(num_players),
          "scene": create_scene(),
          "prefabs": PREFABS,
          "charPrefabMap": CHAR_PREFAB_MAP,
      },
  }
  return lab2d_settings


def get_config(factory=create_lab2d_settings):
  """Default configuration for training on the territory level."""
  config = config_dict.ConfigDict()

  # Lua script configuration.
  config.num_players = 9
  config.lab2d_settings = factory(config.num_players)

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
      "WORLD.RGB": specs.rgb(184, 312),
  })

  return config
