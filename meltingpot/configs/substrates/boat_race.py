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
"""Configuration for boat_race.

Example video: https://youtu.be/sEh1hRJVuFw

Six players engage in a back and forth series of boat races across a river to
reach a patch of apples, which confer reward when eaten. Boats, however, cannot
be rowed by a single player, and thus, players need to find a partner before
each race and coordinate their rowing during the race to cross the river. When
the players are on the boat, they can choose from two different rowing actions
at each timestamp: (a) paddle, which is efficient, but costly if not
coordinated with its partner; and (b) flail, an inefficient action which isn't
affected by the partner's action. When both players paddle simultaneously, the
boat moves one cell every few timesteps. When either player flails, the boat has
a probability of moving one cell, and a reward penalty is given to its partner
if that partner is currently paddling, i.e. if they have executed the paddle
action within the last few timesteps.
"""

from collections.abc import Mapping, Sequence
from typing import Any

from meltingpot.utils.substrates import colors
from meltingpot.utils.substrates import shapes
from meltingpot.utils.substrates import specs
from ml_collections import config_dict as configdict

# Warning: setting `_ENABLE_DEBUG_OBSERVATIONS = True` may cause slowdown.
_ENABLE_DEBUG_OBSERVATIONS = False

# This substrate only makes sense with exactly six players.
MANDATED_NUM_PLAYERS = 6
NUM_RACES = 8
PARTNER_DURATION = 75
RACE_DURATION = 225
UNROLL_LENGTH = 100

ASCII_MAP = r"""
WWWWWWWWWWWWWWWWWWWWWWWWWW
W                        W
W                        W
W                        W
W      RRRRRRRRRRRR      W
W      RRRRRRRRRRRR      W
W      RRRRRRRRRRRR      W
W      RRRRRRRRRRRR      W
W                        W
W      S  SS  SS  S      W
W      S%%SS%%SS%%S      W
W      S  SS  SS  S      W
~~~~~~~~gg~~gg~~gg~~~~~~~~
~~~~~~~~{{~~{{~~{{~~~~~~~~
~~~~~~~~AA~~AA~~AA~~~~~~~~
~~~~~~~~{{~~{{~~{{~~~~~~~~
~~~~~~~~{{~~{{~~{{~~~~~~~~
~~~~~~~~AA~~AA~~AA~~~~~~~~
~~~~~~~~{{~~{{~~{{~~~~~~~~
~~~~~~~~{{~~{{~~{{~~~~~~~~
~~~~~~~~AA~~AA~~AA~~~~~~~~
~~~~~~~~{{~~{{~~{{~~~~~~~~
~~~~~~~~{{~~{{~~{{~~~~~~~~
~~~~~~~~AA~~AA~~AA~~~~~~~~
~~~~~~~~/\~~/\~~/\~~~~~~~~
~~~~~~~p;:qp;:qp;:q~~~~~~~
W      SLJSSLJSSLJS      W
W      S--SS--SS--S      W
W      S  SS  SS  S      W
W                        W
W      OOOOOOOOOOOO      W
W      OOOOOOOOOOOO      W
W      OOOOOOOOOOOO      W
W      OOOOOOOOOOOO      W
W                        W
W    ________________    W
W    ________________    W
WWWWWWWWWWWWWWWWWWWWWWWWWW
"""

# `prefab` determines which prefab game object to use for each `char` in the
# ascii map.
CHAR_PREFAB_MAP = {
    "_": {"type": "all", "list": ["floor", "spawn_point"]},
    " ": "floor",
    "W": "wall",
    "S": {"type": "all", "list": ["floor", "semaphore"]},
    "A": {"type": "all", "list": ["water_background", "single_apple"]},
    "R": {"type": "all", "list": ["floor", "respawning_apple_north"]},
    "O": {"type": "all", "list": ["floor", "respawning_apple_south"]},
    "%": {"type": "all", "list": ["floor", "barrier_north"]},
    "-": {"type": "all", "list": ["floor", "barrier_south"]},
    "~": "water_blocking",
    "{": "water_background",
    "g": {"type": "all", "list": ["goal_north", "water_background"]},
    "/": {"type": "all", "list": ["boat_FL", "water_background"]},
    "\\": {"type": "all", "list": ["boat_FR", "water_background"]},
    "L": {"type": "all", "list": ["floor", "boat_RL"]},
    "J": {"type": "all", "list": ["floor", "boat_RR"]},
    "p": {"type": "all", "list": ["oar_L", "water_blocking"]},
    "q": {"type": "all", "list": ["oar_R", "water_blocking"]},
    ";": {"type": "all", "list": ["seat_L", "goal_south", "water_background"]},
    ":": {"type": "all", "list": ["seat_R", "goal_south", "water_background"]},
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
                "initialState": "partnerChoice",
                "stateConfigs": [{
                    "state": "ForceEmbark",
                }, {
                    "state": "partnerChoice",
                }, {
                    "state": "semaphore_yellow",
                }, {
                    "state": "semaphore_green",
                }, {
                    "state": "boatRace",
                }, {
                    "state": "semaphore_red",  # A temporary state at end game.
                }],
            }
        },
        {"component": "Transform",},
        {
            "component": "RaceManager",
            "kwargs": {
                "raceStartTime": PARTNER_DURATION,
                "raceDuration": RACE_DURATION,
            },
        },
        {
            "component": "GlobalRaceTracker",
            "kwargs": {
                "numPlayers": MANDATED_NUM_PLAYERS,
            },
        },
        {
            "component": "EpisodeManager",
            "kwargs": {
                "checkInterval": UNROLL_LENGTH,
            },
        },
    ]
}
if _ENABLE_DEBUG_OBSERVATIONS:
  SCENE["components"].append({
      "component": "GlobalMetricReporter",
      "kwargs": {
          "metrics": [
              {
                  "name": "RACE_START",
                  "type": "tensor.Int32Tensor",
                  "shape": (MANDATED_NUM_PLAYERS // 2, 2),
                  "component": "GlobalRaceTracker",
                  "variable": "raceStart",
              },
              {
                  "name": "STROKES",
                  "type": "tensor.Int32Tensor",
                  "shape": (MANDATED_NUM_PLAYERS,),
                  "component": "GlobalRaceTracker",
                  "variable": "strokes",
              },
          ]
      },
  })

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
                    "+": (157, 142, 120, 255),
                    "*": (154, 139, 115, 255),
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


SINGLE_APPLE = {
    "name": "single_apple",
    "components": [
        {
            "component": "StateManager",
            "kwargs": {
                "initialState": "apple",
                "stateConfigs": [
                    {"state": "apple",
                     "layer": "singleAppleLayer",
                     "sprite": "apple",
                    },
                    {"state": "appleWait",
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
                "spriteNames": ["apple"],
                "spriteShapes": [shapes.HD_APPLE],
                "palettes": [shapes.get_palette((40, 180, 40, 255))],
                "noRotates": [False],
            }
        },
        {
            "component": "Edible",
            "kwargs": {
                "liveState": "apple",
                "waitState": "appleWait",
                "rewardForEating": 1.0,
            }
        },
    ]
}


def get_respawning_apple(bank_side: str):
  initial_state = "apple" if bank_side == "N" else "applePause"
  return {
      "name": "apple",
      "components": [
          {
              "component": "StateManager",
              "kwargs": {
                  "initialState": initial_state,
                  "stateConfigs": [
                      {"state": "apple",
                       "layer": "superOverlay",
                       "sprite": "apple",
                      },
                      {"state": "appleWait",
                       "layer": "logic",
                      },
                      {"state": "applePause",
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
                  "spriteNames": ["apple"],
                  "spriteShapes": [shapes.HD_APPLE],
                  "palettes": [shapes.get_palette((40, 180, 40, 255))],
                  "noRotates": [False],
              }
          },
          {
              "component": "Edible",
              "kwargs": {
                  "liveState": "apple",
                  "waitState": "appleWait",
                  "rewardForEating": 1.0,
              }
          },
          {
              "component": "FixedRateRegrow",
              "kwargs": {
                  "liveState": "apple",
                  "waitState": "appleWait",
                  "regrowRate": 0.1,
              }
          },
      ]
  }


SEMAPHORE = {
    "name": "semaphore",
    "components": [
        {
            "component": "StateManager",
            "kwargs": {
                "initialState": "red",
                "stateConfigs": [
                    {"state": "red",
                     "layer": "upperPhysical",
                     "sprite": "red",
                     "groups": ["semaphore"]},
                    {"state": "yellow",
                     "layer": "upperPhysical",
                     "sprite": "yellow",
                     "groups": ["semaphore"]},
                    {"state": "green",
                     "layer": "upperPhysical",
                     "sprite": "green",
                     "groups": ["semaphore"]},
                ]
            }
        },
        {"component": "Transform",},
        {
            "component": "Appearance",
            "kwargs": {
                "renderMode": "ascii_shape",
                "spriteNames": ["red", "yellow", "green"],
                "spriteShapes": [shapes.COIN] * 3,
                "palettes": [shapes.RED_COIN_PALETTE, shapes.COIN_PALETTE,
                             shapes.GREEN_COIN_PALETTE],
                "noRotates": [False] * 3,
            }
        },
    ]
}


def get_barrier(bank_side: str = "N"):
  initial_state = "off" if bank_side == "N" else "on"
  return {
      "name": "barrier",
      "components": [
          {
              "component": "StateManager",
              "kwargs": {
                  "initialState": initial_state,
                  "stateConfigs": [
                      {"state": "on",
                       "layer": "upperPhysical",
                       "sprite": "barrierOn",
                       "groups": ["barrier"]},
                      {"state": "off",
                       "layer": "superOverlay",
                       "sprite": "barrierOff",
                       "groups": ["barrier"]},
                  ]
              }
          },
          {"component": "Transform",},
          {
              "component": "Appearance",
              "kwargs": {
                  "renderMode": "ascii_shape",
                  "spriteNames": ["barrierOn", "barrierOff"],
                  "spriteShapes": [shapes.BARRIER_ON, shapes.BARRIER_OFF],
                  "palettes": [shapes.GRAY_PALETTE] * 2,
                  "noRotates": [False] * 2,
              }
          },
      ]
  }


def get_water(layer: str):
  """Get a water game object at the specified layer, possibly with a goal."""
  return {
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
                  "palettes": [shapes.WATER_PALETTE] * 4,
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
      ]
  }


def get_goal(bank_side: str = "N"):
  return {
      "name": "water_goal",
      "components": [
          {
              "component": "StateManager",
              "kwargs": {
                  "initialState": "goalNonBlocking",
                  "stateConfigs": [{
                      "state": "goalNonBlocking",
                      "layer": "logic",
                  }, {
                      "state": "goalBlocking",
                      "layer": "upperPhysical",
                  }],
              }
          },
          {"component": "Transform",},
          {
              "component": "WaterGoal",
              "kwargs": {
                  "bank_side": bank_side
              },
          }
      ]
  }


def get_boat(front: bool, left: bool):
  suffix = "{}{}".format("F" if front else "R", "L" if left else "R")
  shape = {
      "FL": shapes.BOAT_FRONT_L,
      "FR": shapes.BOAT_FRONT_R,
      "RL": shapes.BOAT_REAR_L,
      "RR": shapes.BOAT_REAR_R,
  }
  return {
      "name": f"boat_{suffix}",
      "components": [
          {
              "component": "StateManager",
              "kwargs": {
                  "initialState": "boat",
                  "stateConfigs": [
                      {"state": "boat",
                       "layer": "lowerPhysical",
                       "sprite": f"Boat{suffix}",
                       "groups": ["boat"]},
                      {"state": "boatFull",
                       "layer": "overlay",
                       "sprite": f"Boat{suffix}",
                       "groups": ["boat"]},
                  ]
              }
          },
          {"component": "Transform",},
          {
              "component": "Appearance",
              "kwargs": {
                  "renderMode": "ascii_shape",
                  "spriteNames": [f"Boat{suffix}"],
                  "spriteShapes": [shape[suffix]],
                  "palettes": [shapes.BOAT_PALETTE],
                  "noRotates": [False]
              }
          },
      ]
  }


def get_seat(left: bool):
  """Get a seat prefab. Left seats contain the BoatManager component."""
  suffix = "L" if left else "R"
  shape = {
      "L": shapes.BOAT_SEAT_L,
      "R": shapes.BOAT_SEAT_R,
  }
  seat = {
      "name": f"seat_{suffix}",
      "components": [
          {
              "component": "StateManager",
              "kwargs": {
                  "initialState": "seat",
                  "stateConfigs": [
                      {"state": "seat",
                       "layer": "lowerPhysical",
                       "sprite": f"Seat{suffix}",
                       "groups": ["seat", "boat"]},
                      {"state": "seatTaken",
                       "layer": "overlay",
                       "sprite": f"Seat{suffix}",
                       "contact": "boat"},
                      {"state": "seatUsed",
                       "layer": "lowerPhysical",
                       "sprite": f"Seat{suffix}"},
                  ]
              }
          },
          {"component": "Transform",},
          {
              "component": "Appearance",
              "kwargs": {
                  "renderMode": "ascii_shape",
                  "spriteNames": [f"Seat{suffix}"],
                  "spriteShapes": [shape[suffix]],
                  "palettes": [shapes.BOAT_PALETTE],
                  "noRotates": [False]
              }
          },
          {
              "component": "Seat",
              "kwargs": {
              },
          },
      ]
  }
  if left:
    seat["components"] += [
        {
            "component": "BoatManager",
            "kwargs": {
                "flailEffectiveness": 0.1,
            }
        }
    ]
  return seat


def get_oar(left: bool):
  suffix = "L" if left else "R"
  shape = {
      "L": [shapes.OAR_DOWN_L, shapes.OAR_UP_L, shapes.OAR_UP_L],
      "R": [shapes.OAR_DOWN_R, shapes.OAR_UP_R, shapes.OAR_UP_R],
  }
  return {
      "name": f"oar_{suffix}",
      "components": [
          {
              "component": "StateManager",
              "kwargs": {
                  "initialState": "oarDown",
                  "stateConfigs": [
                      {"state": "oarDown",
                       "layer": "overlay",
                       "sprite": f"OarDown{suffix}",
                       "groups": ["oar", "boat"]},

                      {"state": "oarUp_row",
                       "layer": "overlay",
                       "sprite": f"OarUp{suffix}Row",
                       "groups": ["oar", "boat"]},

                      {"state": "oarUp_flail",
                       "layer": "overlay",
                       "sprite": f"OarUp{suffix}Flail",
                       "groups": ["oar", "boat"]},
                  ]
              }
          },
          {"component": "Transform",},
          {
              "component": "Appearance",
              "kwargs": {
                  "renderMode": "ascii_shape",
                  "spriteNames": [
                      f"OarDown{suffix}",
                      f"OarUp{suffix}Row",
                      f"OarUp{suffix}Flail",
                  ],
                  "spriteShapes": shape[suffix],
                  "palettes": [shapes.GRAY_PALETTE] * 3,
                  "noRotates": [False] * 3
              }
          },
      ]
  }


AVATAR = {
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

                    {"state": "rowing",
                     "layer": "superOverlay",
                     "sprite": "Avatar",
                     "contact": "avatar",
                     "groups": ["players"]},

                    {"state": "landed",
                     "layer": "upperPhysical",
                     "sprite": "Avatar",
                     "contact": "avatar",
                     "groups": ["players"]},
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
                "palettes": [shapes.get_palette(colors.human_readable[0])],
                "noRotates": [True]
            }
        },
        {
            "component": "Avatar",
            "kwargs": {
                "index": -1,  # player index to be overwritten.
                "aliveState": "player",
                "waitState": "playerWait",
                "spawnGroup": "spawnPoints",
                "actionOrder": [
                    "move", "turn", "row", "flail"],
                "actionSpec": {
                    "move": {"default": 0, "min": 0, "max": len(_COMPASS)},
                    "turn": {"default": 0, "min": -1, "max": 1},
                    "row": {"default": 0, "min": 0, "max": 1},
                    "flail": {"default": 0, "min": 0, "max": 1},
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
            "component": "Rowing",
            "kwargs": {
                "cooldownTime": 5,
                "playerRole": "none",
            },
        },
        {
            "component": "StrokesTracker",
            "kwargs": {}
        },
    ]
}
if _ENABLE_DEBUG_OBSERVATIONS:
  AVATAR["components"].append({
      "component": "LocationObserver",
      "kwargs": {"objectIsAvatar": True, "alsoReportOrientation": True},
  })


# PREFABS is a dictionary mapping names to template game objects that can
# be cloned and placed in multiple locations accoring to an ascii map.
PREFABS = {
    "floor": FLOOR,
    "wall": WALL,
    "spawn_point": SPAWN_POINT,
    "water_blocking": get_water("upperPhysical"),
    "water_background": get_water("background"),
    "goal_north": get_goal(bank_side="N"),
    "goal_south": get_goal(bank_side="S"),
    "barrier_north": get_barrier(bank_side="N"),
    "barrier_south": get_barrier(bank_side="S"),
    "single_apple": SINGLE_APPLE,
    "respawning_apple_north": get_respawning_apple(bank_side="N"),
    "respawning_apple_south": get_respawning_apple(bank_side="S"),
    "semaphore": SEMAPHORE,
    "boat_FL": get_boat(front=True, left=True),
    "boat_FR": get_boat(front=True, left=False),
    "boat_RL": get_boat(front=False, left=True),
    "boat_RR": get_boat(front=False, left=False),
    "seat_L": get_seat(left=True),
    "seat_R": get_seat(left=False),
    "oar_L": get_oar(left=True),
    "oar_R": get_oar(left=False),
    "avatar": AVATAR,
}

# PLAYER_COLOR_PALETTES is a list with each entry specifying the color to use
# for the player at the corresponding index.
# These correspond to the persistent agent colors, but are meaningless for the
# human player. They will be overridden by the environment_builder.
PLAYER_COLOR_PALETTES = [
    shapes.get_palette(colors.human_readable[0]),
    shapes.get_palette(colors.human_readable[1]),
    shapes.get_palette(colors.human_readable[2]),
    shapes.get_palette(colors.human_readable[3]),
    shapes.get_palette(colors.human_readable[4]),
    shapes.get_palette(colors.human_readable[5]),
]

# Primitive action components.
# pylint: disable=bad-whitespace
# pyformat: disable
NOOP       = {"move": 0, "turn":  0, "row": 0, "flail": 0}
FORWARD    = {"move": 1, "turn":  0, "row": 0, "flail": 0}
STEP_RIGHT = {"move": 2, "turn":  0, "row": 0, "flail": 0}
BACKWARD   = {"move": 3, "turn":  0, "row": 0, "flail": 0}
STEP_LEFT  = {"move": 4, "turn":  0, "row": 0, "flail": 0}
TURN_LEFT  = {"move": 0, "turn": -1, "row": 0, "flail": 0}
TURN_RIGHT = {"move": 0, "turn":  1, "row": 0, "flail": 0}
ROW        = {"move": 0, "turn":  0, "row": 1, "flail": 0}
FLAIL      = {"move": 0, "turn":  0, "row": 0, "flail": 1}
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
    ROW,
    FLAIL,
)


def get_config():
  """Configuration for the boat_race substrate."""
  config = configdict.ConfigDict()

  # Specify the number of players to particate in each episode (optional).
  config.recommended_num_players = MANDATED_NUM_PLAYERS

  # Action set configuration.
  config.action_set = ACTION_SET
  # Observation format configuration.
  config.individual_observation_names = [
      "RGB",
  ]
  config.global_observation_names = [
      "WORLD.RGB",
  ]

  # The specs of the environment (from a single-agent perspective).
  config.action_spec = specs.action(len(ACTION_SET))
  config.timestep_spec = specs.timestep({
      "RGB": specs.OBSERVATION["RGB"],
      # Debug only (do not use the following observations in policies).
      "WORLD.RGB": specs.rgb(304, 208),
  })

  # The roles assigned to each player.
  config.valid_roles = frozenset({"default", "target"})

  return config


def build(
    roles: Sequence[str],
    config: configdict.ConfigDict,
) -> Mapping[str, Any]:
  """Build boat_race substrate given player roles."""
  assert len(roles) == MANDATED_NUM_PLAYERS, "Wrong number of players"
  assert "num_races" in config, (
      "Cannot build substrate without specifying the number of races. Try "
      "using the specific config (e.g. `boat_race__eight_races`) instead.")

  # Build the rest of the substrate definition.
  substrate_definition = dict(
      levelName="boat_race",
      levelDirectory="meltingpot/lua/levels",
      numPlayers=MANDATED_NUM_PLAYERS,
      maxEpisodeLengthFrames=config.num_races * (PARTNER_DURATION +
                                                 RACE_DURATION),
      spriteSize=8,
      topology="BOUNDED",  # Choose from ["BOUNDED", "TORUS"],
      simulation={
          "map": ASCII_MAP,
          "scene": SCENE,
          "prefabs": PREFABS,
          "charPrefabMap": CHAR_PREFAB_MAP,
          "playerPalettes": PLAYER_COLOR_PALETTES,
      },
  )
  return substrate_definition
