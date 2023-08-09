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
"""Configuration for coop_mining substrate.

Example video: https://youtu.be/KvwGUinjIsk

Two different types of ore appear at random in empty spaces. Players are
equipped with a mining beam that attempts to extract the ore. Iron ore (gray)
can be mined by a single player and confers a reward of 1 when extracted. Gold
ore (yellow) has to be mined by exactly two players within a time window of 3
timesteps and confers a reward of 8 to each of them. When a player mines a gold
ore, it flashes to indicate that it is ready to be mined by another player. If
no other player, or if too many players try to mine within that time, it will
revert back to normal.

This games has some properties in common with Stag-Hunt. Mining iron is akin to
playing Hare, with a reliable payoff, without needing to coordinate with others.
Mining gold is akin to Stag because it has an opportunity cost (not mining
iron). If noone else helps, mining gold gives no reward. However, if two players
stick together (spatially) and go around mining gold, they will both receive
higher reward than if they were mining iron.
"""

from collections.abc import Mapping, Sequence
from typing import Any

from meltingpot.utils.substrates import colors
from meltingpot.utils.substrates import shapes
from meltingpot.utils.substrates import specs
from ml_collections import config_dict

# Warning: setting `_ENABLE_DEBUG_OBSERVATIONS = True` may cause slowdown.
_ENABLE_DEBUG_OBSERVATIONS = False

NUM_ORE_TYPES = 2
MAX_TOKENS_PER_TYPE = 6

ASCII_MAP = """
WWWWWWWWWWWWWWWWWWWWWWWWWWW
WOOOOOOOOOOOOOOOOOOOOOOOOOW
WOPOOOOOOOOOPOOOOOPOOOOOPOW
WOOOOOOOOWOOOOOOOOOOOOOOOOW
WOOOOOOOOWOOOOOOOOOOWOOOOOW
WOOOOOOOOWOOOOOOOOOOWOOOOOW
WOOOOOOOOWWWWWWWOOOOWOOOPOW
WOPOWWOOOOWOOOOOOOOOWOOOOOW
WOOOOOOOOOWOOPOOOOOOOOOOOOW
WOOOOOOOOOWOOOOOWWWOOOOOOOW
WOOOOOOOOOWOOOOOOOOOOOOOOOW
WOOOOOOOOOOOOOOOOOOOOOOOPOW
WOPOOOWWWOOOOOOWWWWWWWWOOOW
WOOWWWWOOOOOOOOOOOOOOOOOOOW
WOOOOOWOOOOWOOOOOPOOOOOOOOW
WOOOOOWOOOOWOOOOOOOOOOOOPOW
WOOOOOWOOOOOWOOOOOOOOWOOOOW
WOOOOOOWOOOOOWWWWOOOOWOOOOW
WOPOOOOOWOOOOOOOOOOOOWOOOOW
WOOOOOOOOWOOOPOOOOOOOOOOPOW
WOOOOOOOOOWOOOOOOOOWOOOOOOW
WOOOOWOOOOOOOOOOOOOWOOOOOOW
WOOOOWOOOOOOOOOWWWWWWWWOOOW
WOOOOWOOOOOOOOOOOOWOOOOOOOW
WOPOOOOOOPOOOOOOOPOOOOOOPOW
WOOOOOOOOOOOOOOOOOOOOOOOOOW
WWWWWWWWWWWWWWWWWWWWWWWWWWW
"""

# `prefab` determines which prefab game object to use for each `char` in the
# ascii map.
CHAR_PREFAB_MAP = {
    "P": "spawn_point",
    "W": "wall",
    "O": "ore",
}

_COMPASS = ["N", "E", "S", "W"]


SCENE = {
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
            "component": "Transform",
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
            "component": "BeamBlocker",
            "kwargs": {
                "beamType": "mine"
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

RAW_ORE = """
xxxxxxxxxxxxxxxx
xxxxxxxxxxxxxxxx
xxxxxxxxxxxxxxxx
xxxxxxxxxxxxxxxx
xxxxxxxxxxxxxxxx
xxxxxxxxxxxxxxxx
xxxxxxxxxxxxxxxx
xxxxxxxxxxxxxxxx
xxxxxx*&&@xxxxxx
xxxxx****&@xxxxx
xxxx**&@*&**xxxx
xxxx*&*&*&@@@xxx
xxx****@&***&@xx
xx****&&*****&&x
******&*****&**&
****************
"""

PARTIAL_ORE = """
xxxxxxxxxxxxxxxx
xxxxxx#xx##xxxxx
xxxxxxx##xxxxxxx
xxxxxx##x#xxxxxx
x##xxxxxxxxxxxxx
xx###xxxxxxxx##x
xxx###xxx####xxx
xxxx#######xxxxx
xxxx######xxxxxx
xx###***###xxxxx
##xx**&@*&###xxx
xxxx*&*&*&@@##xx
xxx****@&***&@xx
xx****&&*****&&x
******&*****&**&
****************
"""

IRON_PALETTE = {
    "*": (70, 60, 70, 255),
    "&": (140, 120, 140, 255),
    "@": (170, 160, 170, 255),
    "#": (255, 240, 255, 255),
    "x": (0, 0, 0, 0)
}

GOLD_PALETTE = {
    "*": (90, 90, 20, 255),
    "&": (180, 180, 40, 255),
    "@": (220, 220, 60, 255),
    "#": (255, 255, 240, 255),
    "x": (0, 0, 0, 0)
}

ORE = {
    "name": "ore",
    "components": [
        {
            "component": "StateManager",
            "kwargs": {
                "initialState": "oreWait",
                "stateConfigs": [
                    {"state": "oreWait",
                     "layer": "lowerPhysical",
                     "sprite": "oreWait",
                     "groups": []},
                    {"state": "ironRaw",
                     "layer": "lowerPhysical",
                     "sprite": "ironRaw",
                     "groups": ["tokens"]},
                    {"state": "goldRaw",
                     "layer": "lowerPhysical",
                     "sprite": "goldRaw",
                     "groups": ["tokens"]},
                    {"state": "goldPartial",
                     "layer": "lowerPhysical",
                     "sprite": "goldPartial",
                     "groups": ["tokens"]},
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
                "spriteNames": ["oreWait", "ironRaw", "goldRaw", "goldPartial"],
                "spriteShapes": [RAW_ORE, RAW_ORE, RAW_ORE, PARTIAL_ORE],
                "palettes": [shapes.INVISIBLE_PALETTE, IRON_PALETTE,
                             GOLD_PALETTE, GOLD_PALETTE],
                "noRotates": [True] * 4,
            }
        },
        {
            "component": "Ore",
            "kwargs": {
                "waitState": "oreWait",
                "rawState": "goldRaw",
                "partialState": "goldPartial",
                "minNumMiners": 2,
                "miningWindow": 3,
            }
        },
        {
            "component": "Ore",
            "kwargs": {
                "waitState": "oreWait",
                "rawState": "ironRaw",
                "partialState": "ironRaw",
                "minNumMiners": 1,
                "miningWindow": 2,
            }
        },
        {
            "component": "FixedRateRegrow",
            "kwargs": {
                "liveStates": ["ironRaw", "goldRaw"],
                "liveRates": [0.0002, 0.00008],
                "waitState": "oreWait",
            }
        },
    ]
}


PLAYER_COLOR_PALETTES = []
for human_readable_color in colors.human_readable:
  PLAYER_COLOR_PALETTES.append(shapes.get_palette(human_readable_color))


def get_avatar_object(num_players: int, player_index: int):
  """Construct an avatar object."""
  lua_index = player_index + 1
  color_palette = PLAYER_COLOR_PALETTES[player_index]
  avatar_sprite_name = "avatarSprite{}".format(lua_index)
  avatar_object = {
      "name": "avatar",
      "components": [
          {
              "component": "StateManager",
              "kwargs": {
                  "initialState": "player",
                  "stateConfigs": [
                      {"state": "player",
                       "layer": "upperPhysical",
                       "sprite": avatar_sprite_name,
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
              "component": "ReadyToShootObservation",
              "kwargs": {
                  "zapperComponent": "MineBeam",
              },
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
                  "aliveState": "player",
                  "waitState": "playerWait",
                  "speed": 1.0,
                  "spawnGroup": "spawnPoints",
                  "actionOrder": ["move", "turn", "mine"],
                  "actionSpec": {
                      "move": {"default": 0, "min": 0, "max": len(_COMPASS)},
                      "turn": {"default": 0, "min": -1, "max": 1},
                      "mine": {"default": 0, "min": 0, "max": 1},
                  },
                  "view": {
                      "left": 5,
                      "right": 5,
                      "forward": 9,
                      "backward": 1,
                      "centered": False
                  }
              }
          },
          {
              "component": "MineBeam",
              "kwargs": {
                  "cooldownTime": 3,
                  "beamLength": 3,
                  "beamRadius": 0,
                  "agentRole": "none",
                  "roleRewardForMining": {
                      "none": [0, 0],
                      "golddigger": [0, 0.2], "irondigger": [0, 0]},
                  "roleRewardForExtracting": {
                      "none": [1, 8],
                      "golddigger": [-1, 8], "irondigger": [8, -1]},
              }
          },
          {
              "component": "MiningTracker",
              "kwargs": {
                  "numPlayers": num_players,
                  "numOreTypes": NUM_ORE_TYPES,
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


def get_avatar_objects(num_players: int):
  return [get_avatar_object(num_players, i) for i in range(num_players)]


# PREFABS is a dictionary mapping names to template game objects that can
# be cloned and placed in multiple locations accoring to an ascii map.
PREFABS = {
    "wall": WALL,
    "spawn_point": SPAWN_POINT,
    "ore": ORE,
}

# Primitive action components.
# pylint: disable=bad-whitespace
# pyformat: disable
NOOP       = {"move": 0, "turn":  0, "mine": 0}
FORWARD    = {"move": 1, "turn":  0, "mine": 0}
STEP_RIGHT = {"move": 2, "turn":  0, "mine": 0}
BACKWARD   = {"move": 3, "turn":  0, "mine": 0}
STEP_LEFT  = {"move": 4, "turn":  0, "mine": 0}
TURN_LEFT  = {"move": 0, "turn": -1, "mine": 0}
TURN_RIGHT = {"move": 0, "turn":  1, "mine": 0}
MINE       = {"move": 0, "turn":  0, "mine": 1}
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
    MINE,
)


def get_config():
  """Default configuration for the coop_mining level."""
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
      "WORLD.RGB": specs.rgb(216, 216),
  })

  # The roles assigned to each player.
  config.valid_roles = frozenset({"default", "target"})
  config.default_player_roles = ("default",) * 6

  return config


def build(
    roles: Sequence[str],
    config: config_dict.ConfigDict,
) -> Mapping[str, Any]:
  """Build substrate given player roles."""
  del config
  num_players = len(roles)

  return dict(
      levelName="coop_mining",
      levelDirectory="meltingpot/lua/levels",
      numPlayers=num_players,
      # Define upper bound of episode length since episodes end stochastically.
      maxEpisodeLengthFrames=5000,
      spriteSize=8,
      topology="BOUNDED",  # Choose from ["BOUNDED", "TORUS"],
      simulation={
          "map": ASCII_MAP,
          "gameObjects": get_avatar_objects(num_players),
          "scene": SCENE,
          "prefabs": PREFABS,
          "charPrefabMap": CHAR_PREFAB_MAP,
      },
  )
