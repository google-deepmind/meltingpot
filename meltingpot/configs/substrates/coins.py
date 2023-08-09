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
"""Configuration for running a Coins game in Melting Pot.

Example video: https://youtu.be/a_SYgt4tBsc
"""

from collections.abc import Mapping, Sequence
import random
from typing import Any

from meltingpot.utils.substrates import colors
from meltingpot.utils.substrates import game_object_utils
from meltingpot.utils.substrates import shapes
from meltingpot.utils.substrates import specs
from ml_collections import config_dict as configdict

PrefabConfig = game_object_utils.PrefabConfig

# Warning: setting `_ENABLE_DEBUG_OBSERVATIONS = True` may cause slowdown.
_ENABLE_DEBUG_OBSERVATIONS = False

MANDATED_NUM_PLAYERS = 2

COIN_PALETTES = {
    "coin_red": shapes.get_palette((238, 102, 119)),    # Red.
    "coin_blue": shapes.get_palette((68, 119, 170)),    # Blue.
    "coin_yellow": shapes.get_palette((204, 187, 68)),  # Yellow.
    "coin_green": shapes.get_palette((34, 136, 51)),    # Green.
    "coin_purple": shapes.get_palette((170, 51, 119))   # Purple.
}


def get_ascii_map(
    min_width: int, max_width: int, min_height: int, max_height: int) -> str:
  """Procedurally generate ASCII map."""
  assert min_width <= max_width
  assert min_height <= max_height

  # Sample random map width and height.
  width = random.randint(min_width, max_width)
  height = random.randint(min_height, max_height)

  # Make top row (walls). Pad to max width to ensure all maps are same size.
  ascii_map = ["W"] * (width + 2) + [" "] * (max_width - width)

  # Make middle rows (navigable interior).
  for row in range(height):
    # Add walls and coins.
    ascii_map += ["\nW"] + ["C"] * width + ["W"]

    if row == 1:
      # Add top-right spawn point.
      ascii_map[-3] = "_"
    elif row == height - 2:
      # Add bottom-left spawn point.
      ascii_map[-width] = "_"

    # Pad to max width.
    ascii_map += [" "] * (max_width - width)

  # Make bottom row (walls). Pad to max width.
  ascii_map += ["\n"] + ["W"] * (width + 2) + [" "] * (max_width - width)

  # Pad with extra rows to reach max height.
  for _ in range(max_height - height):
    ascii_map += ["\n"] + [" "] * max_width

  # Join list of strings into single string.
  ascii_map = "".join(ascii_map)

  return ascii_map

# `prefab` determines which prefab game object to use for each `char` in the
# ascii map.
CHAR_PREFAB_MAP = {
    "_": "spawn_point",
    "W": "wall",
    "C": "coin",
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
            "component": "GlobalCoinCollectionTracker",
            "kwargs": {
                "numPlayers": MANDATED_NUM_PLAYERS,
            },
        },
        {
            "component": "StochasticIntervalEpisodeEnding",
            "kwargs": {
                "minimumFramesPerEpisode": 300,
                "intervalLength": 100,  # Set equal to unroll length.
                "probabilityTerminationPerInterval": 0.05
            }
        }
    ]
}
if _ENABLE_DEBUG_OBSERVATIONS:
  SCENE["components"].append({
      "component": "GlobalMetricReporter",
      "kwargs": {
          "metrics": [
              {
                  "name": "COINS_COLLECTED",
                  "type": "tensor.Int32Tensor",
                  "shape": (MANDATED_NUM_PLAYERS, 2),
                  "component": "GlobalCoinCollectionTracker",
                  "variable": "coinsCollected",
              },
          ]
      },
  })


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
        {"component": "Transform",},
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
        {
            "component": "BeamBlocker",
            "kwargs": {
                "beamType": "zap"
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
        {"component": "Transform",},
    ]
}


def get_coin(
    coin_type_a: str,
    coin_type_b: str,
    regrow_rate: float,
    reward_self_for_match: float,
    reward_self_for_mismatch: float,
    reward_other_for_match: float,
    reward_other_for_mismatch: float,
    ) -> PrefabConfig:
  """Create `PrefabConfig` for coin component."""
  return {
      "name": "coin",
      "components": [
          {
              "component": "StateManager",
              "kwargs": {
                  "initialState": "coinWait",
                  "stateConfigs": [
                      {"state": coin_type_a,
                       "layer": "superOverlay",
                       "sprite": coin_type_a,
                      },
                      {"state": coin_type_b,
                       "layer": "superOverlay",
                       "sprite": coin_type_b,
                      },
                      {"state": "coinWait",
                       "layer": "logic",
                      },
                  ]
              }
          },
          {"component": "Transform",},
          {
              "component": "Appearance",
              "kwargs": {
                  "renderMode": "ascii_shape",
                  "spriteNames": [coin_type_a, coin_type_b],
                  "spriteShapes": [shapes.COIN] * 2,
                  "palettes": [COIN_PALETTES[coin_type_a],
                               COIN_PALETTES[coin_type_b]],
                  "noRotates": [False] * 2,
              }
          },
          {
              "component": "Coin",
              "kwargs": {
                  "waitState": "coinWait",
                  "rewardSelfForMatch": reward_self_for_match,
                  "rewardSelfForMismatch": reward_self_for_mismatch,
                  "rewardOtherForMatch": reward_other_for_match,
                  "rewardOtherForMismatch": reward_other_for_mismatch,
              }
          },
          {
              "component": "ChoiceCoinRegrow",
              "kwargs": {
                  "liveStateA": coin_type_a,
                  "liveStateB": coin_type_b,
                  "waitState": "coinWait",
                  "regrowRate": regrow_rate,
              }
          },
      ]
  }


def get_avatar(coin_type: str):
  """Create an avatar object."""
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
                       "sprite": "Avatar",
                       "contact": "avatar",
                       "groups": ["players"]},

                      {"state": "playerWait",
                       "groups": ["playerWaits"]},
                  ]
              }
          },
          {"component": "Transform",},
          {
              "component": "Appearance",
              "kwargs": {
                  "renderMode": "ascii_shape",
                  "spriteNames": ["Avatar"],
                  "spriteShapes": [shapes.CUTE_AVATAR],
                  # Palette to be overwritten.
                  "palettes": [shapes.get_palette(colors.palette[0])],
                  "noRotates": [True]
              }
          },
          {
              "component": "Avatar",
              "kwargs": {
                  "index": -1,  # Player index to be overwritten.
                  "aliveState": "player",
                  "waitState": "playerWait",
                  "spawnGroup": "spawnPoints",
                  "actionOrder": ["move", "turn",],
                  "actionSpec": {
                      "move": {"default": 0, "min": 0, "max": len(_COMPASS)},
                      "turn": {"default": 0, "min": -1, "max": 1},
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
              "component": "PlayerCoinType",
              "kwargs": {
                  "coinType": coin_type,
              },
          },
          {
              "component": "Role",
              "kwargs": {
                  #  Role has no effect if all factors are 1.0.
                  "multiplyRewardSelfForMatch": 1.0,
                  "multiplyRewardSelfForMismatch": 1.0,
                  "multiplyRewardOtherForMatch": 1.0,
                  "multiplyRewardOtherForMismatch": 1.0,
              },
          },
          {
              "component": "PartnerTracker",
              "kwargs": {}
          },
      ]
  }
  # Signals needed for puppeteers.
  metrics = [
      {
          "name": "MISMATCHED_COIN_COLLECTED_BY_PARTNER",
          "type": "Doubles",
          "shape": [],
          "component": "PartnerTracker",
          "variable": "partnerCollectedMismatch",
      },
  ]
  if _ENABLE_DEBUG_OBSERVATIONS:
    avatar_object["components"].append({
        "component": "LocationObserver",
        "kwargs": {"objectIsAvatar": True, "alsoReportOrientation": True},
    })
    # Debug metrics
    metrics.append({
        "name": "MATCHED_COIN_COLLECTED",
        "type": "Doubles",
        "shape": [],
        "component": "Role",
        "variable": "cumulantCollectedMatch",
    })
    metrics.append({
        "name": "MISMATCHED_COIN_COLLECTED",
        "type": "Doubles",
        "shape": [],
        "component": "Role",
        "variable": "cumulantCollectedMismatch",
    })
    metrics.append({
        "name": "MATCHED_COIN_COLLECTED_BY_PARTNER",
        "type": "Doubles",
        "shape": [],
        "component": "PartnerTracker",
        "variable": "partnerCollectedMatch",
    })

  # Add the metrics reporter.
  avatar_object["components"].append({
      "component": "AvatarMetricReporter",
      "kwargs": {"metrics": metrics}
  })

  return avatar_object


# `prefabs` is a dictionary mapping names to template game objects that can
# be cloned and placed in multiple locations accoring to an ascii map.
def get_prefabs(
    coin_type_a: str,
    coin_type_b: str,
    regrow_rate: float = 0.0005,
    reward_self_for_match: float = 1.0,
    reward_self_for_mismatch: float = 1.0,
    reward_other_for_match: float = 0.0,
    reward_other_for_mismatch: float = -2.0,
) -> PrefabConfig:
  """Make `prefabs` (a dictionary mapping names to template game objects)."""
  coin = get_coin(coin_type_a=coin_type_a,
                  coin_type_b=coin_type_b,
                  regrow_rate=regrow_rate,
                  reward_self_for_match=reward_self_for_match,
                  reward_self_for_mismatch=reward_self_for_mismatch,
                  reward_other_for_match=reward_other_for_match,
                  reward_other_for_mismatch=reward_other_for_mismatch)
  return {"wall": WALL, "spawn_point": SPAWN_POINT, "coin": coin}


# `player_color_palettes` is a list with each entry specifying the color to use
# for the player at the corresponding index.
# These correspond to the persistent agent colors, but are meaningless for the
# human player. They will be overridden by the environment_builder.
def get_player_color_palettes(
    coin_type_a: str, coin_type_b: str) -> Sequence[Mapping[str, shapes.Color]]:
  return [COIN_PALETTES[coin_type_a], COIN_PALETTES[coin_type_b]]

# Primitive action components.
# pylint: disable=bad-whitespace
# pyformat: disable
NOOP       = {"move": 0, "turn":  0,}
FORWARD    = {"move": 1, "turn":  0,}
STEP_RIGHT = {"move": 2, "turn":  0,}
BACKWARD   = {"move": 3, "turn":  0,}
STEP_LEFT  = {"move": 4, "turn":  0,}
TURN_LEFT  = {"move": 0, "turn": -1,}
TURN_RIGHT = {"move": 0, "turn":  1,}
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
)


def get_config():
  """Default configuration for the Coins substrate."""
  config = configdict.ConfigDict()

  # Set the size of the map.
  config.min_width = 10
  config.max_width = 15
  config.min_height = 10
  config.max_height = 15

  # Action set configuration.
  config.action_set = ACTION_SET

  # Observation format configuration.
  config.individual_observation_names = [
      "RGB",
      # Global switching signals for puppeteers.
      "MISMATCHED_COIN_COLLECTED_BY_PARTNER",
  ]
  config.global_observation_names = [
      "WORLD.RGB"
  ]

  # The specs of the environment (from a single-agent perspective).
  config.action_spec = specs.action(len(ACTION_SET))
  config.timestep_spec = specs.timestep({
      "RGB": specs.OBSERVATION["RGB"],
      # Switching signals for puppeteers.
      "MISMATCHED_COIN_COLLECTED_BY_PARTNER": specs.float64(),
      # Debug only (do not use the following observations in policies).
      "WORLD.RGB": specs.rgb(136, 136),
  })

  # The roles assigned to each player.
  config.valid_roles = frozenset({"default"})
  config.default_player_roles = ("default",) * MANDATED_NUM_PLAYERS

  return config


def build(
    roles: Sequence[str],
    config: configdict.ConfigDict,
) -> Mapping[str, Any]:
  """Build the coins substrate given player roles."""
  assert len(roles) == MANDATED_NUM_PLAYERS, "Wrong number of players"
  # Randomly choose colors.
  coin_type_a, coin_type_b = random.sample(tuple(COIN_PALETTES), k=2)

  # Manually build avatar config.
  num_players = len(roles)
  player_color_palettes = get_player_color_palettes(
      coin_type_a=coin_type_a, coin_type_b=coin_type_b)
  avatar_objects = game_object_utils.build_avatar_objects(
      num_players, {"avatar": get_avatar(coin_type_a)}, player_color_palettes)  # pytype: disable=wrong-arg-types  # allow-recursive-types
  game_object_utils.get_first_named_component(
      avatar_objects[1], "PlayerCoinType")["kwargs"]["coinType"] = coin_type_b

  # Build the substrate definition.
  substrate_definition = dict(
      levelName="coins",
      levelDirectory="meltingpot/lua/levels",
      numPlayers=num_players,
      # Define upper bound of episode length since episodes end stochastically.
      maxEpisodeLengthFrames=5000,
      spriteSize=8,
      topology="BOUNDED",  # Choose from ["BOUNDED", "TORUS"],
      simulation={
          "map": get_ascii_map(min_width=config.min_width,
                               max_width=config.max_width,
                               min_height=config.min_height,
                               max_height=config.max_height),
          "scene": SCENE,
          "prefabs": get_prefabs(coin_type_a=coin_type_a,
                                 coin_type_b=coin_type_b),
          "charPrefabMap": CHAR_PREFAB_MAP,
          "gameObjects": avatar_objects,
      }
  )
  return substrate_definition
