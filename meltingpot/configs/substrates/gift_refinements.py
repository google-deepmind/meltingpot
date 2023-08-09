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
"""Configuration for gift_refinements.

Example video: https://youtu.be/C1C2CJ__mhQ

Tokens randomly spawn in empty spaces. When collected, they are put in the
player's inventory where they can be consumed by for reward. Alternatively, a
token can be refined into three higher refinement tokens and gifted to another
player. This is akin to tokens initially being a raw material, like chunks of
metal, and then being split and shaped into more useful goods for added value.
A token can only be refined a finite number of times, after which it cannot be
split again, nor refined further; although they can still be gifted.

Gift Refinements is inspired by the Trust Game from behavioural economics where
the first player has an endowment and chooses how much to donate to a second
player who receives three times its value. Then the second player chooses how
much to give back to the first.

In Gift Refinements, tokens can only be refined twice (i.e. there are three
types of tokens). The token that spawn are always of the rawest type, and the
only way to create more refined tokens is to gift them to another player.
Players also have a limited inventory capacity of 15 tokens for each token type.
The special gifting is implemented as a beam that the players can fire. If they
hit another player with the beam while their inventory is not full, they lose
one token of the rawest type they currently hold, and the hit player receives
either three token of the next refinement (if the token gifted wasn't already
at maximum refinement), or the token gifted (otherwise).

The players have an action to consume tokens which takes all tokens of all types
currently in their inventory and converts them into reward. All tokens are worth
1 reward regardless of refinement level.

The game is set up in such a way that there are several ways players can form
mutually beneficial interactions, but all of them require trust. For instance,
A pair of players might have one player pick up a token and immediately gift it
to the other one who receives three. Then the second player returns one token
which leaves them with three and two tokens respectively. If they both consume
after this, they both benefitted from the interaction. A more extreme case would
have them take one token and refine it maximally to produce 9 tokens that they
can split five and four with 10 roughly alternating gifting actions.
"""

from collections.abc import Mapping, Sequence
from typing import Any

from meltingpot.utils.substrates import colors
from meltingpot.utils.substrates import shapes
from meltingpot.utils.substrates import specs
from ml_collections import config_dict

# Warning: setting `_ENABLE_DEBUG_OBSERVATIONS = True` may cause slowdown.
_ENABLE_DEBUG_OBSERVATIONS = False

NUM_TOKEN_TYPES = 3
MAX_TOKENS_PER_TYPE = 15

ASCII_MAP = """
WWWWWWWWWWWWWWWWWWWWWWWWWWW
WTTTTTTTTTTTTTTTTTTTTTTTTTW
WTPTTTTTTTTTPTTTTTPTTTTTPTW
WTTTTTTTTWTTTTTTTTTTTTTTTTW
WTTTTTTTTWTTTTTTTTTTWTTTTTW
WTTTTTTTTWTTTTTTTTTTWTTTTTW
WTTTTTTTTWWWWWWWTTTTWTTTPTW
WTPTWWTTTTWTTTTTTTTTWTTTTTW
WTTTTTTTTTWTTPTTTTTTTTTTTTW
WTTTTTTTTTWTTTTTWWWTTTTTTTW
WTTTTTTTTTWTTTTTTTTTTTTTTTW
WTTTTTTTTTTTTTTTTTTTTTTTPTW
WTPTTTWWWTTTTTTWWWWWWWWTTTW
WTTWWWWTTTTTTTTTTTTTTTTTTTW
WTTTTTWTTTTWTTTTTPTTTTTTTTW
WTTTTTWTTTTWTTTTTTTTTTTTPTW
WTTTTTWTTTTTWTTTTTTTTWTTTTW
WTTTTTTWTTTTTWWWWTTTTWTTTTW
WTPTTTTTWTTTTTTTTTTTTWTTTTW
WTTTTTTTTWTTTPTTTTTTTTTTPTW
WTTTTTTTTTWTTTTTTTTWTTTTTTW
WTTTTWTTTTTTTTTTTTTWTTTTTTW
WTTTTWTTTTTTTTTWWWWWWWWTTTW
WTTTTWTTTTTTTTTTTTWTTTTTTTW
WTPTTTTTTPTTTTTTTPTTTTTTPTW
WTTTTTTTTTTTTTTTTTTTTTTTTTW
WWWWWWWWWWWWWWWWWWWWWWWWWWW
"""

# `prefab` determines which prefab game object to use for each `char` in the
# ascii map.
CHAR_PREFAB_MAP = {
    "P": "spawn_point",
    "W": "wall",
    "T": "token",
}

_COMPASS = ["N", "E", "S", "W"]

# The Scene objece is a non-physical object, it components implement global
# logic. In this case, that includes holding the global berry counters to
# implement the regrowth rate, as well as some of the observations.
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
                "beamType": "gift"
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

TOKEN = {
    "name": "token",
    "components": [
        {
            "component": "StateManager",
            "kwargs": {
                "initialState": "tokenWait",
                "stateConfigs": [
                    {"state": "tokenWait",
                     "layer": "lowerPhysical",
                     "sprite": "coinWait",
                     "groups": []},
                    {"state": "token",
                     "layer": "lowerPhysical",
                     "sprite": "coin",
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
                "spriteNames": ["coin", "coinWait"],
                "spriteShapes": [shapes.COIN, shapes.COIN],
                "palettes": [
                    shapes.COIN_PALETTE, shapes.INVISIBLE_PALETTE],
            }
        },
        {
            "component": "Pickable",
            "kwargs": {
                "liveState": "token",
                "waitState": "tokenWait",
                "rewardForPicking": 0.0,
            }
        },
        {
            "component": "FixedRateRegrow",
            "kwargs": {
                "liveState": "token",
                "waitState": "tokenWait",
                "regrowRate": 0.0002,
            }
        },
    ]
}


PLAYER_COLOR_PALETTES = []
for human_readable_color in colors.human_readable:
  PLAYER_COLOR_PALETTES.append(shapes.get_palette(human_readable_color))


def get_avatar_object(num_players: int, player_index: int):
  """Construct an avatar object."""
  # Lua is 1-indexed.
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
                      {
                          "state": "player",
                          "layer": "upperPhysical",
                          "sprite": avatar_sprite_name,
                          "contact": "avatar",
                          "groups": ["players"]
                      },
                      {
                          "state": "playerWait",
                          "groups": ["playerWaits"]
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
                  "spriteNames": [avatar_sprite_name],
                  "spriteShapes": [shapes.CUTE_AVATAR],
                  "palettes": [color_palette],
                  "noRotates": [True],
              }
          },
          {
              "component": "Avatar",
              "kwargs": {
                  "index": lua_index,
                  "aliveState": "player",
                  "waitState": "playerWait",
                  "spawnGroup": "spawnPoints",
                  "actionOrder": [
                      "move", "turn", "refineAndGift", "consumeTokens"
                  ],
                  "actionSpec": {
                      "move": {
                          "default": 0,
                          "min": 0,
                          "max": len(_COMPASS)
                      },
                      "turn": {
                          "default": 0,
                          "min": -1,
                          "max": 1
                      },
                      "refineAndGift": {
                          "default": 0,
                          "min": 0,
                          "max": 1
                      },
                      "consumeTokens": {
                          "default": 0,
                          "min": 0,
                          "max": 1
                      },
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
              "component": "Inventory",
              "kwargs": {
                  "capacityPerType": MAX_TOKENS_PER_TYPE,
                  "numTokenTypes": NUM_TOKEN_TYPES,
              }
          },
          {
              "component": "GiftBeam",
              "kwargs": {
                  "cooldownTime": 3,
                  "beamLength": 5,
                  "beamRadius": 0,
                  "agentRole": "none",
                  "giftMultiplier": 5,
                  "successfulGiftReward": 10,
                  "roleRewardForGifting": {
                      "none": 0.0,
                      "gifter": 0.2,
                      "selfish": -2.0
                  },
              }
          },
          {
              "component": "ReadyToShootObservation",
              "kwargs": {
                  "zapperComponent": "GiftBeam",
              },
          },
          {
              "component": "AvatarMetricReporter",
              "kwargs": {
                  "metrics": [
                      {
                          "name": "INVENTORY",
                          "type": "tensor.DoubleTensor",
                          "shape": [NUM_TOKEN_TYPES],
                          "component": "Inventory",
                          "variable": "inventory"
                      },
                  ]
              }
          },
          {
              "component": "TokenTracker",
              "kwargs": {
                  "numPlayers": num_players,
                  "numTokenTypes": NUM_TOKEN_TYPES,
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


# PREFABS is a dictionary mapping names to template game objects that can
# be cloned and placed in multiple locations accoring to an ascii map.
PREFABS = {
    "wall": WALL,
    "spawn_point": SPAWN_POINT,
    "token": TOKEN,
}


def get_avatar_objects(num_players: int):
  return [get_avatar_object(num_players, i) for i in range(num_players)]


# Primitive action components.
# pylint: disable=bad-whitespace
# pyformat: disable
NOOP            = {
    "move": 0, "turn":  0, "refineAndGift": 0, "consumeTokens": 0}
FORWARD         = {
    "move": 1, "turn":  0, "refineAndGift": 0, "consumeTokens": 0}
STEP_RIGHT      = {
    "move": 2, "turn":  0, "refineAndGift": 0, "consumeTokens": 0}
BACKWARD        = {
    "move": 3, "turn":  0, "refineAndGift": 0, "consumeTokens": 0}
STEP_LEFT       = {
    "move": 4, "turn":  0, "refineAndGift": 0, "consumeTokens": 0}
TURN_LEFT       = {
    "move": 0, "turn": -1, "refineAndGift": 0, "consumeTokens": 0}
TURN_RIGHT      = {
    "move": 0, "turn":  1, "refineAndGift": 0, "consumeTokens": 0}
REFINE_AND_GIFT = {
    "move": 0, "turn":  0, "refineAndGift": 1, "consumeTokens": 0}
CONSUME_TOKENS  = {
    "move": 0, "turn":  0, "refineAndGift": 0, "consumeTokens": 1}
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
    REFINE_AND_GIFT,
    CONSUME_TOKENS,
)


def get_config():
  """Default configuration for the gift_refinements level."""
  config = config_dict.ConfigDict()

  # Action set configuration.
  config.action_set = ACTION_SET
  # Observation format configuration.
  config.individual_observation_names = [
      "RGB",
      "READY_TO_SHOOT",
      "INVENTORY",
  ]
  config.global_observation_names = [
      "WORLD.RGB",
  ]

  # The specs of the environment (from a single-agent perspective).
  config.action_spec = specs.action(len(ACTION_SET))
  config.timestep_spec = specs.timestep({
      "RGB": specs.OBSERVATION["RGB"],
      "READY_TO_SHOOT": specs.OBSERVATION["READY_TO_SHOOT"],
      "INVENTORY": specs.inventory(3),
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
  # Build the rest of the substrate definition.
  substrate_definition = dict(
      levelName="gift_refinements",
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
  return substrate_definition
