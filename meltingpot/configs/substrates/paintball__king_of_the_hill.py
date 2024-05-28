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
"""Configuration for King of the Hill.

Example video: https://youtu.be/VVAfeObAZzI

See _Capture the Flag_ for the description of the painting, zapping, and
movement mechanics, which also operate in this substrate.

In the King of the Hill substrate the goal is to control the hill region in
the center of the map. The hill is considered to be controlled by a team if at
least 80% of it has been colored in one team's color. The status of the hill is
indicated by indicator tiles around the map an in the center. Red indicator
tiles mean the red team is in control. Blue indicator tiles mean the blue team
is in control. Purple indicator tiles mean no team is in control.
"""

from typing import Any, Dict, Mapping, Optional, Sequence

from meltingpot.utils.substrates import shapes
from meltingpot.utils.substrates import specs
from ml_collections import config_dict
import numpy as np

# Warning: setting `_ENABLE_DEBUG_OBSERVATIONS = True` may cause slowdown.
_ENABLE_DEBUG_OBSERVATIONS = False

_COMPASS = ["N", "E", "S", "W"]

ASCII_MAP = """
IIIIIIIIIIIIIIIIIIIIIII
IWWWWWWWWWWWWWWWWWWWWWI
IWPPP,PPPP,P,PPPP,PPPWI
IWPPP,,PP,,,,,PP,,PPPWI
IWPPP,,,,,,,,,,,,,PPPWI
IWP,,WW,,,,,,,,,WW,,PWI
IW,,,WWDWWWDWWW,WW,,,WI
IW,,,,,,uuuuuuu,D,,,,WI
IW,,,,WlGGGGGGGrW,,,,WI
IWHWWHWlGGGGGGGrWHWWHWI
IWHWWHWlGGGGGGGrWHWWHWI
IW,,,,DlGGGIGGGrD,,,,WI
IWHWWHWlGGGGGGGrWHWWHWI
IWHWWHWlGGGGGGGrWHWWHWI
IW,,,,WlGGGGGGGrW,,,,WI
IW,,,,D,ddddddd,,,,,,WI
IW,,,WW,WWWDWWWDWW,,,WI
IWQ,,WW,,,,,,,,,WW,,QWI
IWQQQ,,,,,,,,,,,,,QQQWI
IWQQQ,,QQ,,,,,QQ,,QQQWI
IWQQQ,QQQQ,Q,QQQQ,QQQWI
IWWWWWWWWWWWWWWWWWWWWWI
IIIIIIIIIIIIIIIIIIIIIII
"""

# `prefab` determines which prefab game object to use for each `char` in the
# ascii map.
CHAR_PREFAB_MAP = {
    "P": {"type": "all", "list": ["spawn_point_red", "ground"]},
    "Q": {"type": "all", "list": ["spawn_point_blue", "ground"]},
    "W": "wall",
    "D": {"type": "choice",
          "list": ["destroyable_wall"] * 9 + ["destroyed_wall"]},
    "H": {"type": "choice",
          "list": ["destroyable_wall"] * 3 + ["destroyed_wall"]},
    "G": "hill",
    ",": "ground",
    "I": {"type": "all", "list": ["indicator", "indicator_frame"]},

    # Lines marking the edge of the hill.
    "u": {"type": "all", "list": ["ground", "line_north"]},
    "r": {"type": "all", "list": ["ground", "line_west"]},
    "d": {"type": "all", "list": ["ground", "line_south"]},
    "l": {"type": "all", "list": ["ground", "line_east"]},
}

RED_COLOR = (225, 55, 85, 255)
DARKER_RED_COLOR = (200, 35, 55, 255)
DARKEST_RED_COLOR = (160, 5, 25, 255)

BLUE_COLOR = (85, 55, 225, 255)
DARKER_BLUE_COLOR = (55, 35, 200, 255)
DARKEST_BLUE_COLOR = (25, 5, 160, 255)

PURPLE_COLOR = (107, 63, 160, 255)

LINE_NORTH = """
xxxxxxxx
xxxxxxxx
xxxxxxxx
xxxxxxxx
xxxxxxxx
xxxxxxxx
xxxxxxxx
oooooooo
"""
LINE_SOUTH = shapes.flip_vertical(LINE_NORTH)
LINE_EAST = """
xxxxxxxo
xxxxxxxo
xxxxxxxo
xxxxxxxo
xxxxxxxo
xxxxxxxo
xxxxxxxo
xxxxxxxo
"""
LINE_WEST = shapes.flip_horizontal(LINE_EAST)


def multiply_tuple(color_tuple, factor):
  alpha = color_tuple[3]
  return tuple([int(np.min([x * factor, alpha])) for x in color_tuple[0: 3]])

TEAMS_DATA = {
    "red": {"color": RED_COLOR,
            "spawn_group": "{}SpawnPoints".format("red")},
    "blue": {"color": BLUE_COLOR,
             "spawn_group": "{}SpawnPoints".format("blue")},
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
            "component": "AllBeamBlocker",
            "kwargs": {}
        },
    ]
}


def get_marking_line(orientation: str):
  """Return a line prefab to trace out the area of the hill."""
  if orientation == "N":
    shape = LINE_NORTH
  elif orientation == "E":
    shape = LINE_EAST
  elif orientation == "S":
    shape = LINE_SOUTH
  elif orientation == "W":
    shape = LINE_WEST
  else:
    raise ValueError(f"Unrecognized orientation: {orientation}")

  line_name = f"line_{orientation}"
  prefab = {
      "name": line_name,
      "components": [
          {
              "component": "StateManager",
              "kwargs": {
                  "initialState": line_name,
                  "stateConfigs": [{
                      "state": line_name,
                      "layer": "lowerPhysical",
                      "sprite": line_name,
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
                  "spriteNames": [line_name,],
                  "spriteShapes": [shape],
                  "palettes": [{"x": (0, 0, 0, 0),
                                "o": (75, 75, 75, 120)}],
                  "noRotates": [False]
              }
          },
      ]
  }
  return prefab


INDICATOR_FRAME = {
    "name": "indicator_frame",
    "components": [
        {
            "component": "StateManager",
            "kwargs": {
                "initialState": "inert",
                "stateConfigs": [
                    {"state": "inert",
                     "layer": "superOverlay",
                     "sprite": "InertFrame"}
                ]
            }
        },
        {"component": "Transform",},
        {
            "component": "Appearance",
            "kwargs": {
                "renderMode": "ascii_shape",
                "spriteNames": ["InertFrame"],
                "spriteShapes": [shapes.BUTTON],
                "palettes": [{"*": (0, 0, 0, 0),
                              "x": (55, 55, 55, 255),
                              "#": (0, 0, 0, 0)}],
                "noRotates": [True]
            }
        },
    ]
}


INDICATOR = {
    "name": "control_indicator",
    "components": [
        {
            "component": "StateManager",
            "kwargs": {
                "initialState": "uncontrolled",
                "stateConfigs": [
                    {
                        "state": "uncontrolled",
                        "layer": "background",
                        "sprite": "UncontrolledIndicator",
                    },
                    {
                        "state": "red",
                        "layer": "background",
                        "sprite": "RedIndicator",
                    },
                    {
                        "state": "blue",
                        "layer": "background",
                        "sprite": "BlueIndicator",
                    }
                ]
            }
        },
        {"component": "Transform",},
        {
            "component": "Appearance",
            "kwargs": {
                "spriteNames": ["UncontrolledIndicator",
                                "RedIndicator",
                                "BlueIndicator"],
                "spriteRGBColors": [PURPLE_COLOR,
                                    DARKER_RED_COLOR,
                                    DARKER_BLUE_COLOR]
            }
        },
        {"component": "ControlIndicator",},
    ]
}


def create_ground_prefab(is_hill=False):
  """Return a prefab for a normal ground or a hill prefab."""
  if is_hill:
    sprite_names = ["RedHill", "BlueHill"]
    sprite_colors = [DARKER_RED_COLOR, DARKER_BLUE_COLOR]
    groups = ["grounds", "hills"]
    clean_groups = ["hill_clean",]
    red_groups = ["hill_red",]
    blue_groups = ["hill_blue",]
  else:
    sprite_names = ["RedGround", "BlueGround"]
    sprite_colors = [DARKEST_RED_COLOR, DARKEST_BLUE_COLOR]
    groups = ["grounds",]
    clean_groups = []
    red_groups = []
    blue_groups = []

  prefab = {
      "name": "ground",
      "components": [
          {
              "component": "StateManager",
              "kwargs": {
                  "initialState": "clean",
                  "stateConfigs": [
                      {
                          "state": "clean",
                          "layer": "alternateLogic",
                          "groups": groups + clean_groups,
                      },
                      {
                          "state": "red",
                          "layer": "alternateLogic",
                          "sprite": sprite_names[0],
                          "groups": groups + red_groups,
                      },
                      {
                          "state": "blue",
                          "layer": "alternateLogic",
                          "sprite": sprite_names[1],
                          "groups": groups + blue_groups,
                      },
                  ]
              }
          },
          {"component": "Transform",},
          {
              "component": "Appearance",
              "kwargs": {
                  "spriteNames": sprite_names,
                  "spriteRGBColors": sprite_colors
              }
          },
          {
              "component": "GroundOrHill",
              "kwargs": {
                  # Must set the name to "Ground" so the color zapper component
                  # can find the location underneath the avatar to color it.
                  "name": "Ground",
                  "teamNames": ["red", "blue"],
                  "isHill": is_hill,
              }
          },
      ]
  }
  return prefab


def create_destroyable_wall_prefab(initial_state):
  """Return destroyable wall prefab, potentially starting in destroyed state."""
  if initial_state == "destroyed":
    initial_health = 0
  else:
    initial_health = 5
  prefab = {
      "name": "destroyableWall",
      "components": [
          {
              "component": "StateManager",
              "kwargs": {
                  "initialState": initial_state,
                  "stateConfigs": [
                      {
                          "state": "destroyable",
                          "layer": "upperPhysical",
                          "sprite": "DestroyableWall",
                      },
                      {
                          "state": "damaged",
                          "layer": "upperPhysical",
                          "sprite": "DamagedWall",
                      },
                      {
                          "state": "destroyed",
                          "layer": "alternateLogic",
                          "sprite": "Rubble",
                      },
                  ],
              }
          },
          {"component": "Transform",},
          {
              "component": "Appearance",
              "kwargs": {
                  "renderMode": "ascii_shape",
                  "spriteNames": ["DestroyableWall",
                                  "DamagedWall",
                                  "Rubble"],
                  "spriteShapes": [shapes.WALL,
                                   shapes.WALL,
                                   shapes.WALL],
                  "palettes": [{"*": (55, 55, 55, 255),
                                "&": (100, 100, 100, 255),
                                "@": (109, 109, 109, 255),
                                "#": (152, 152, 152, 255)},
                               {"*": (55, 55, 55, 255),
                                "&": (100, 100, 100, 255),
                                "@": (79, 79, 79, 255),
                                "#": (152, 152, 152, 255)},
                               {"*": (0, 0, 0, 255),
                                "&": (0, 0, 0, 255),
                                "@": (29, 29, 29, 255),
                                "#": (0, 0, 0, 255)}],
                  "noRotates": [True] * 3
              }
          },
          {
              "component": "Destroyable",
              "kwargs": {"hitNames": ["red", "blue"],
                         "initialHealth": initial_health,
                         "damagedHealthLevel": 2}
          }
      ]
  }
  return prefab


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
                      "layer": "logic",
                      "groups": [TEAMS_DATA[team]["spawn_group"]],
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


# PREFABS is a dictionary mapping names to template game objects that can
# be cloned and placed in multiple locations accoring to an ascii map.
PREFABS = {
    "wall": WALL,
    "spawn_point_red": create_spawn_point_prefab("red"),
    "spawn_point_blue": create_spawn_point_prefab("blue"),
    "destroyable_wall": create_destroyable_wall_prefab("destroyable"),
    "destroyed_wall": create_destroyable_wall_prefab("destroyed"),
    "hill": create_ground_prefab(is_hill=True),
    "ground": create_ground_prefab(is_hill=False),
    "indicator": INDICATOR,
    "indicator_frame": INDICATOR_FRAME,
    "line_north": get_marking_line("N"),
    "line_east": get_marking_line("E"),
    "line_south": get_marking_line("S"),
    "line_west": get_marking_line("W"),
}

# Primitive action components.
# pylint: disable=bad-whitespace
# pyformat: disable
NOOP       = {"move": 0, "turn":  0, "fireZap": 0}
FORWARD    = {"move": 1, "turn":  0, "fireZap": 0}
STEP_RIGHT = {"move": 2, "turn":  0, "fireZap": 0}
BACKWARD   = {"move": 3, "turn":  0, "fireZap": 0}
STEP_LEFT  = {"move": 4, "turn":  0, "fireZap": 0}
TURN_LEFT  = {"move": 0, "turn": -1, "fireZap": 0}
TURN_RIGHT = {"move": 0, "turn":  1, "fireZap": 0}
FIRE_ZAP_A = {"move": 0, "turn":  0, "fireZap": 1}
FIRE_ZAP_B = {"move": 0, "turn":  0, "fireZap": 2}
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
    FIRE_ZAP_A,  # a short-range beam with a wide area of effect
    FIRE_ZAP_B,  # a longer range beam with a thin area of effect
)


# The Scene is a non-physical object, its components implement global logic.
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
          {"component": "Transform",},
          {
              "component": "HillManager",
              "kwargs": {
                  "percentToCapture": 80,
                  "rewardPerStepInControl": 1.0,
              }
          }
      ]
  }
  return scene


def create_avatar_object(
    player_idx: int,
    team: str,
    override_taste_kwargs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
  """Create an avatar object."""
  # Lua is 1-indexed.
  lua_index = player_idx + 1

  team_color = TEAMS_DATA[team]["color"]

  health1_avatar_sprite_name = "avatarSprite{}Health1".format(lua_index)
  health2_avatar_sprite_name = "avatarSprite{}Health2".format(lua_index)
  health3_avatar_sprite_name = "avatarSprite{}Health3".format(lua_index)

  health1_color_palette = shapes.get_palette(multiply_tuple(team_color, 0.35))
  health2_color_palette = shapes.get_palette(team_color)
  health3_color_palette = shapes.get_palette(multiply_tuple(team_color, 1.75))

  taste_kwargs = {
      # select `mode` from:
      #   ("none", "control_hill", "paint_hill", "zap_while_in_control")
      "mode": "none",
      "rewardAmount": 0.0,
      "zeroMainReward": False,
      "minFramesBetweenHillRewards": 0,
  }
  if override_taste_kwargs:
    taste_kwargs.update(override_taste_kwargs)

  avatar_object = {
      "name": "avatar",
      "components": [
          {
              "component": "StateManager",
              "kwargs": {
                  "initialState": "health2",
                  "stateConfigs": [
                      {"state": "health1",
                       "layer": "upperPhysical",
                       "sprite": health1_avatar_sprite_name,
                       "contact": "avatar",
                       "groups": ["players"]},
                      {"state": "health2",
                       "layer": "upperPhysical",
                       "sprite": health2_avatar_sprite_name,
                       "contact": "avatar",
                       "groups": ["players"]},
                      {"state": "health3",
                       "layer": "upperPhysical",
                       "sprite": health3_avatar_sprite_name,
                       "contact": "avatar",
                       "groups": ["players"]},

                      # Player wait state used when they have been zapped out.
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
                  "spriteNames": [health1_avatar_sprite_name,
                                  health2_avatar_sprite_name,
                                  health3_avatar_sprite_name],
                  "spriteShapes": [shapes.CUTE_AVATAR,
                                   shapes.CUTE_AVATAR,
                                   shapes.CUTE_AVATAR],
                  "palettes": [health1_color_palette,
                               health2_color_palette,
                               health3_color_palette],
                  "noRotates": [True] * 3
              }
          },
          {
              "component": "Avatar",
              "kwargs": {
                  "index": lua_index,
                  "aliveState": "health2",
                  "additionalLiveStates": ["health1", "health3"],
                  "waitState": "playerWait",
                  "spawnGroup": TEAMS_DATA[team]["spawn_group"],
                  "actionOrder": ["move",
                                  "turn",
                                  "fireZap"],
                  "actionSpec": {
                      "move": {"default": 0, "min": 0, "max": len(_COMPASS)},
                      "turn": {"default": 0, "min": -1, "max": 1},
                      "fireZap": {"default": 0, "min": 0, "max": 2},
                  },
                  "view": {
                      "left": 5,
                      "right": 5,
                      "forward": 9,
                      "backward": 1,
                      "centered": False
                  },
                  # The following kwarg makes it possible to get rewarded for
                  # team rewards even when an avatar is "dead".
                  "skipWaitStateRewards": False,
              }
          },
          {
              "component": "ColorZapper",
              "kwargs": {
                  "team": team,
                  # The color zapper beam is somewhat transparent.
                  "color": (team_color[0], team_color[1], team_color[2], 150),
                  "cooldownTime": 2,
                  "beamLength": 3,
                  "beamRadius": 1,
                  "secondaryBeamCooldownTime": 4,
                  "secondaryBeamLength": 6,
                  "secondaryBeamRadius": 0,
                  "aliveStates": ["health1", "health2", "health3"],
              }
          },
          {
              "component": "ReadyToShootObservation",
              "kwargs": {
                  "zapperComponent": "ColorZapper",
              }
          },
          {
              "component": "ZappedByColor",
              "kwargs": {
                  "team": team,
                  "allTeamNames": ["red", "blue"],
                  "framesTillRespawn": 80,
                  "penaltyForBeingZapped": 0,
                  "rewardForZapping": 0,
                  "healthRegenerationRate": 0.05,
                  "maxHealthOnGround": 2,
                  "maxHealthOnOwnColor": 3,
                  "maxHealthOnEnemyColor": 1,
                  "groundLayer": "alternateLogic",
              }
          },
          {
              "component": "TeamMember",
              "kwargs": {"team": team}
          },
          {
              "component": "Taste",
              "kwargs": taste_kwargs
          },
      ]
  }
  if _ENABLE_DEBUG_OBSERVATIONS:
    avatar_object["components"].append({
        "component": "LocationObserver",
        "kwargs": {"objectIsAvatar": True, "alsoReportOrientation": True},
    })

  return avatar_object


def _even_vs_odd_team_assignment(num_players,
                                 taste_kwargs: Optional[Any] = None):
  """Assign players with even ids to red team and odd ids to blue team."""
  avatar_objects = []
  for player_idx in range(0, num_players):
    if player_idx % 2 == 0:
      team = "red"
    else:
      team = "blue"
    game_object = create_avatar_object(player_idx, team,
                                       override_taste_kwargs=taste_kwargs)
    avatar_objects.append(game_object)

  return avatar_objects


def _low_vs_high_team_assignment(num_players,
                                 taste_kwargs: Optional[Any] = None):
  """Assign players with id below the median id to blue and above it to red."""
  median = np.median(range(num_players))
  avatar_objects = []
  for player_idx in range(0, num_players):
    if player_idx < median:
      team = "blue"
    elif player_idx > median:
      team = "red"
    else:
      raise ValueError("num_players must be even")
    game_object = create_avatar_object(player_idx, team,
                                       override_taste_kwargs=taste_kwargs)
    avatar_objects.append(game_object)

  return avatar_objects


def create_avatar_objects(num_players,
                          taste_kwargs: Optional[Any] = None,
                          fixed_teams: Optional[bool] = False):
  """Returns list of avatar objects of length 'num_players'."""
  assert num_players % 2 == 0, "num players must be divisible by 2"
  if fixed_teams:
    avatar_objects = _low_vs_high_team_assignment(num_players,
                                                  taste_kwargs=taste_kwargs)
  else:
    avatar_objects = _even_vs_odd_team_assignment(num_players,
                                                  taste_kwargs=taste_kwargs)
  return avatar_objects


def get_config():
  """Default configuration."""
  config = config_dict.ConfigDict()

  # If shaping_kwargs are None then use the default reward structure in which
  # all positive rewards come from your team being in control of the hill and
  # all negative rewwards come from the opposing team being in control of the
  # hill. The default reward structure is zero sum.
  config.shaping_kwargs = None

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
      "WORLD.RGB": specs.rgb(184, 184),
  })

  # The roles assigned to each player.
  config.valid_roles = frozenset({"default"})
  config.default_player_roles = ("default",) * 8

  return config


def build(
    roles: Sequence[str],
    config: config_dict.ConfigDict,
) -> Mapping[str, Any]:
  """Build substrate definition given player roles."""
  num_players = len(roles)
  substrate_definition = dict(
      levelName="paintball__king_of_the_hill",
      levelDirectory="meltingpot/lua/levels",
      numPlayers=num_players,
      maxEpisodeLengthFrames=1000,
      spriteSize=8,
      topology="BOUNDED",  # Choose from ["BOUNDED", "TORUS"],
      simulation={
          "map": ASCII_MAP,
          "gameObjects": create_avatar_objects(
              num_players, taste_kwargs=config.shaping_kwargs),
          "scene": create_scene(),
          "prefabs": PREFABS,
          "charPrefabMap": CHAR_PREFAB_MAP,
      },
  )
  return substrate_definition
