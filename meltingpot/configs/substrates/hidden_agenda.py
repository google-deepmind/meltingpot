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
"""Configuration for hidden_agenda.

Example video: https://youtu.be/voJWckiOh5k

A social deduction (sometimes also called 'hidden role') game where players can
have one of two roles: crewmates and impostors. All players with the same role
are in the same team. The game is a zero-sum competitive game across teams, that
is, all players in the team get the same reward, and their reward is the
negative of the reward for the other team. The roles are hidden (hence the name)
and must be inferred from observations. The crewmates have a numeric advantage
(4 players), while the impostor (1 player) have an information advantage (they
know the roles of every player).

Players can move around a 2D world which contains gems that can be picked up by
walking over them, a deposit in the center of the map (a grate) where collected
gems can be dropped, and a voting room where rounds of deliberation occur (see
below).

Crewmates can carry up to two gems in their inventory at the same time. The gems
must be deposited in the grate before more gems can be collected. Impostors have
a freezing beam that they can use to freeze crewmates. Frozen crewmates are
unable to move or take any action for the rest of the episode.

After a predefined time (200 steps) or whenever an impostor fires its beam
within the field of view of another player (and is not frozen by it) a
deliberation phase starts. During deliberation, players are teleported to the
deliberation room, where they cannot take any movement or firing actions. Their
actions are limited to voting actions. Votes can be for any of the player
indices (0 to 4), 'abstain', or 'no-vote' (if the player is frozen or voted
out). The deliberation phase lasts 25 steps and players are able to change their
vote at any point. If there is a simple majority in the last step of the
deliberation, the player is voted out and removed from the game. Players can
observe the voting of every other player as a special observation called
`VOTING`. Episodes last up to 3000 steps.

The game has several win conditions:

1.  The crewmembers deposit enough gems (32). Crewmates win.
2.  The impostor is voted out during the deliberation phase. Crewmates win.
3.  There is only one crewmate active (i.e. not voted out nor frozen). Impostor
    wins.

If neither of the above conditions are met before the episode ends, the game is
considered a tie, and players get zero reward.
"""
import copy
from typing import Any, Mapping, Sequence

from meltingpot.utils.substrates import shapes
from meltingpot.utils.substrates import specs
from ml_collections import config_dict

# Warning: setting `_ENABLE_DEBUG_OBSERVATIONS = True` may cause slowdown.
_ENABLE_DEBUG_OBSERVATIONS = False

# This substrate only makes sense with exactly five players.
MANDATED_NUM_PLAYERS = 5

# Hidden Agenda palette for player slots
HIDDEN_AGENDA_COLORS = [
    (37, 133, 190),  # light blue
    (133, 37, 190),  # indigo
    (255, 95, 10),  # orange
    (37, 190, 133),  # sea green
    (220, 40, 110),  # salmon pink
    (180, 180, 0),  # golden yellow
    (133, 190, 37),  # lime green
    (135, 73, 124),  # dark pink-purple
    (140, 115, 105),  # light brown
]

# Dictionary mapping the characters in the map with prefabs.
CHAR_PREFAB_MAP = {
    "*": {"type": "all", "list": ["checkered_flooring", "spawn_point"]},
    "V": {"type": "all", "list": ["tiled_flooring1", "voting_spawn_point"]},
    "D": {"type": "all", "list": ["tiled_floor", "teleport_spawn_point"]},
    "F": "nw_wall_corner",
    "7": "ne_wall_corner",
    "J": "se_wall_corner",
    "L": "sw_wall_corner",
    "[": "w_ship_solid_wall",
    "]": "e_ship_solid_wall",
    "^": "n_ship_solid_wall",
    "v": "s_ship_solid_wall",
    "-": "wall_north",
    "T": "tcoupling_n",
    "Z": "tcoupling_e",
    "i": "tcoupling_s",
    "t": "tcoupling_w",
    "|": "wall_west",
    "f": "fill",
    ",": {"type": "all", "list": ["nw_grate", "gem_deposit"]},
    "_": {"type": "all", "list": ["n_grate", "gem_deposit"]},
    ";": {"type": "all", "list": ["ne_grate", "gem_deposit"]},
    "!": {"type": "all", "list": ["w_grate", "gem_deposit"]},
    "=": {"type": "all", "list": ["inner_grate", "gem_deposit"]},
    "1": {"type": "all", "list": ["e_grate", "gem_deposit"]},
    "+": {"type": "all", "list": ["se_grate", "gem_deposit"]},
    "'": {"type": "all", "list": ["s_grate", "gem_deposit"]},
    "`": {"type": "all", "list": ["sw_grate", "gem_deposit"]},
    "/": {"type": "all", "list": ["tiled_floor", "glass_wall"]},
    "n": "tiled_floor",
    "U": "tiled_flooring1",
    "u": "tiled_flooring2",
    "m": "metal_flooring",
    "e": "metal_panel_flooring",
    "x": "checkered_flooring",
    "w": "wood_flooring",
    "~": "threshold",
    "%": {"type": "all", "list": ["metal_panel_flooring", "gem"]},
    "@": {"type": "all", "list": ["metal_flooring", "gem"]},
    "&": {"type": "all", "list": ["wood_flooring", "gem"]},
    "#": {"type": "all", "list": ["tiled_floor", "gem"]},
}

ASCII_MAP = """
F----------^^-------^^----------7
|@mmmmmmmmm[]DDDDDDD[]mmmmmmmmmm|
|mmmmmm@mmm[]///////[]mm@mmm@mmm|
|m@mmmm@mmm|UuVuVuVuU|mmmmm@mmm@|
|mmmm@mm@mm|uVuUuUuVu|mmmmm@mmmm|
|m@mmm@mmmm|UuVuUuVuU|mm@mmmmm@m|
|mm@m@mm@mm|uUuVuVuUu|mm@mm@mmmm|
t-~~~~~~~~-i---------i-~~~~~~~~-Z
|xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx|
|xxxxxxxxxx*xx,___;xx*xxxxxxxxxx|
|xxxxxxxxxx**x!===1x**xxxxxxxxxx|
|xxxxxxxxxx**x!===1x**xxxxxxxxxx|
|xxxxxxxxxx*xx`'''+xx*xxxxxxxxxx|
|xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx|
t-~~~~~~~~-^^^^^^^^^^^-~~~~~~~~-Z
|mmmm@mm@mm[fffffffff]mm@mmmm@mm|
|mmmmmm@mmm[fffffffff]mm@m@mmmmm|
|m@mmmmmm@m[fffffffff]@mmmmm@mmm|
|mmmmm@mmmm[fffffffff]mm@mmmmmm@|
|m@mmmm@mm@[fffffffff]mm@mmmm@mm|
|mmm@mm@mmm[fffffffff]@mmmmmmmmm|
L----------vvvvvvvvvvv----------J
"""


COMPASS = ["N", "E", "S", "W"]

# Aesthetic components for Hidden Agenda

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
                "spriteShapes": [shapes.NW_SHIP_WALL],
                "palettes": [shapes.SHIP_PALETTE],
                "noRotates": [False]
            }
        },
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
                "spriteShapes": [shapes.NE_SHIP_WALL],
                "palettes": [shapes.SHIP_PALETTE],
                "noRotates": [False]
            }
        },
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
                "spriteShapes": [shapes.SE_SHIP_WALL],
                "palettes": [shapes.SHIP_PALETTE],
                "noRotates": [False]
            }
        },
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
                "spriteShapes": [shapes.SW_SHIP_WALL],
                "palettes": [shapes.SHIP_PALETTE],
                "noRotates": [False]
            }
        },
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
                "spriteShapes": [shapes.NS_SHIP_WALL],
                "palettes": [shapes.SHIP_PALETTE],
                "noRotates": [False]
            }
        },
    ]
}

TCOUPLING_E = {
    "name": "tcoupling_e",
    "components": [
        {
            "component": "StateManager",
            "kwargs": {
                "initialState": "tcoupling_e",
                "stateConfigs": [{
                    "state": "tcoupling_e",
                    "layer": "upperPhysical",
                    "sprite": "TcouplingE",
                }],
            }
        },
        {"component": "Transform"},
        {
            "component": "Appearance",
            "kwargs": {
                "renderMode": "ascii_shape",
                "spriteNames": ["TcouplingE"],
                "spriteShapes": [shapes.SHIP_WALL_TCOUPLING_E],
                "palettes": [shapes.SHIP_PALETTE],
                "noRotates": [False]
            }
        },
    ]
}

TCOUPLING_W = {
    "name": "tcoupling_w",
    "components": [
        {
            "component": "StateManager",
            "kwargs": {
                "initialState": "tcoupling_w",
                "stateConfigs": [{
                    "state": "tcoupling_w",
                    "layer": "upperPhysical",
                    "sprite": "TcouplingW",
                }],
            }
        },
        {"component": "Transform"},
        {
            "component": "Appearance",
            "kwargs": {
                "renderMode": "ascii_shape",
                "spriteNames": ["TcouplingW"],
                "spriteShapes": [shapes.SHIP_WALL_TCOUPLING_W],
                "palettes": [shapes.SHIP_PALETTE],
                "noRotates": [False]
            }
        },
    ]
}

TCOUPLING_N = {
    "name": "tcoupling_n",
    "components": [
        {
            "component": "StateManager",
            "kwargs": {
                "initialState": "tcoupling_n",
                "stateConfigs": [{
                    "state": "tcoupling_n",
                    "layer": "upperPhysical",
                    "sprite": "TcouplingN",
                }],
            }
        },
        {"component": "Transform"},
        {
            "component": "Appearance",
            "kwargs": {
                "renderMode": "ascii_shape",
                "spriteNames": ["TcouplingN"],
                "spriteShapes": [shapes.SHIP_WALL_TCOUPLING_N],
                "palettes": [shapes.SHIP_PALETTE],
                "noRotates": [False]
            }
        },
    ]
}

TCOUPLING_S = {
    "name": "tcoupling_s",
    "components": [
        {
            "component": "StateManager",
            "kwargs": {
                "initialState": "tcoupling_s",
                "stateConfigs": [{
                    "state": "tcoupling_s",
                    "layer": "upperPhysical",
                    "sprite": "TcouplingS",
                }],
            }
        },
        {"component": "Transform"},
        {
            "component": "Appearance",
            "kwargs": {
                "renderMode": "ascii_shape",
                "spriteNames": ["TcouplingS"],
                "spriteShapes": [shapes.SHIP_WALL_TCOUPLING_S],
                "palettes": [shapes.SHIP_PALETTE],
                "noRotates": [False]
            }
        },
    ]
}

W_SHIP_SOLID_WALL = {
    "name": "w_ship_solid_wall",
    "components": [
        {
            "component": "StateManager",
            "kwargs": {
                "initialState": "w_ship_solid_wall",
                "stateConfigs": [{
                    "state": "w_ship_solid_wall",
                    "layer": "upperPhysical",
                    "sprite": "WShipSolidWall",
                }],
            }
        },
        {"component": "Transform"},
        {
            "component": "Appearance",
            "kwargs": {
                "renderMode": "ascii_shape",
                "spriteNames": ["WShipSolidWall"],
                "spriteShapes": [shapes.W_SHIP_SOLID_WALL],
                "palettes": [shapes.SHIP_PALETTE],
                "noRotates": [False]
            }
        },
    ]
}

N_SHIP_SOLID_WALL = {
    "name": "n_ship_solid_wall",
    "components": [
        {
            "component": "StateManager",
            "kwargs": {
                "initialState": "n_ship_solid_wall",
                "stateConfigs": [{
                    "state": "n_ship_solid_wall",
                    "layer": "upperPhysical",
                    "sprite": "NShipSolidWall",
                }],
            }
        },
        {"component": "Transform"},
        {
            "component": "Appearance",
            "kwargs": {
                "renderMode": "ascii_shape",
                "spriteNames": ["NShipSolidWall"],
                "spriteShapes": [shapes.N_SHIP_SOLID_WALL],
                "palettes": [shapes.SHIP_PALETTE],
                "noRotates": [False]
            }
        },
    ]
}

E_SHIP_SOLID_WALL = {
    "name": "e_ship_solid_wall",
    "components": [
        {
            "component": "StateManager",
            "kwargs": {
                "initialState": "e_ship_solid_wall",
                "stateConfigs": [{
                    "state": "e_ship_solid_wall",
                    "layer": "upperPhysical",
                    "sprite": "EShipSolidWall",
                }],
            }
        },
        {"component": "Transform"},
        {
            "component": "Appearance",
            "kwargs": {
                "renderMode": "ascii_shape",
                "spriteNames": ["EShipSolidWall"],
                "spriteShapes": [shapes.E_SHIP_SOLID_WALL],
                "palettes": [shapes.SHIP_PALETTE],
                "noRotates": [False]
            }
        },
    ]
}

S_SHIP_SOLID_WALL = {
    "name": "s_ship_solid_wall",
    "components": [
        {
            "component": "StateManager",
            "kwargs": {
                "initialState": "s_ship_solid_wall",
                "stateConfigs": [{
                    "state": "s_ship_solid_wall",
                    "layer": "upperPhysical",
                    "sprite": "SShipSolidWall",
                }],
            }
        },
        {"component": "Transform"},
        {
            "component": "Appearance",
            "kwargs": {
                "renderMode": "ascii_shape",
                "spriteNames": ["SShipSolidWall"],
                "spriteShapes": [shapes.S_SHIP_SOLID_WALL],
                "palettes": [shapes.SHIP_PALETTE],
                "noRotates": [False]
            }
        },
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
                "spriteShapes": [shapes.EW_SHIP_WALL],
                "palettes": [shapes.SHIP_PALETTE],
                "noRotates": [False]
            }
        },
    ]
}

NW_GRATE = {
    "name": "nw_grate",
    "components": [
        {
            "component": "StateManager",
            "kwargs": {
                "initialState": "nw_grate",
                "stateConfigs": [{
                    "state": "nw_grate",
                    "layer": "background",
                    "sprite": "nw_grate",
                }],
            }
        },
        {"component": "Transform"},
        {
            "component": "Appearance",
            "kwargs": {
                "renderMode": "ascii_shape",
                "spriteNames": ["nw_grate"],
                "spriteShapes": [shapes.NW_GRATE],
                "palettes": [shapes.GRATE_PALETTE],
                "noRotates": [False]
            }
        },
    ]
}

N_GRATE = {
    "name": "n_grate",
    "components": [
        {
            "component": "StateManager",
            "kwargs": {
                "initialState": "n_grate",
                "stateConfigs": [{
                    "state": "n_grate",
                    "layer": "background",
                    "sprite": "n_grate",
                }],
            }
        },
        {"component": "Transform"},
        {
            "component": "Appearance",
            "kwargs": {
                "renderMode": "ascii_shape",
                "spriteNames": ["n_grate"],
                "spriteShapes": [shapes.N_GRATE],
                "palettes": [shapes.GRATE_PALETTE],
                "noRotates": [False]
            }
        },
    ]
}

NE_GRATE = {
    "name": "ne_grate",
    "components": [
        {
            "component": "StateManager",
            "kwargs": {
                "initialState": "ne_grate",
                "stateConfigs": [{
                    "state": "ne_grate",
                    "layer": "background",
                    "sprite": "ne_grate",
                }],
            }
        },
        {"component": "Transform"},
        {
            "component": "Appearance",
            "kwargs": {
                "renderMode": "ascii_shape",
                "spriteNames": ["ne_grate"],
                "spriteShapes": [shapes.NE_GRATE],
                "palettes": [shapes.GRATE_PALETTE],
                "noRotates": [False]
            }
        },
    ]
}

W_GRATE = {
    "name": "w_grate",
    "components": [
        {
            "component": "StateManager",
            "kwargs": {
                "initialState": "w_grate",
                "stateConfigs": [{
                    "state": "w_grate",
                    "layer": "background",
                    "sprite": "w_grate",
                }],
            }
        },
        {"component": "Transform"},
        {
            "component": "Appearance",
            "kwargs": {
                "renderMode": "ascii_shape",
                "spriteNames": ["w_grate"],
                "spriteShapes": [shapes.W_GRATE],
                "palettes": [shapes.GRATE_PALETTE],
                "noRotates": [False]
            }
        },
    ]
}

INNER_GRATE = {
    "name": "inner_grate",
    "components": [
        {
            "component": "StateManager",
            "kwargs": {
                "initialState": "inner_grate",
                "stateConfigs": [{
                    "state": "inner_grate",
                    "layer": "background",
                    "sprite": "inner_grate",
                }],
            }
        },
        {"component": "Transform"},
        {
            "component": "Appearance",
            "kwargs": {
                "renderMode": "ascii_shape",
                "spriteNames": ["inner_grate"],
                "spriteShapes": [shapes.INNER_GRATE],
                "palettes": [shapes.GRATE_PALETTE],
                "noRotates": [False]
            }
        },
    ]
}

E_GRATE = {
    "name": "e_grate",
    "components": [
        {
            "component": "StateManager",
            "kwargs": {
                "initialState": "e_grate",
                "stateConfigs": [{
                    "state": "e_grate",
                    "layer": "background",
                    "sprite": "e_grate",
                }],
            }
        },
        {"component": "Transform"},
        {
            "component": "Appearance",
            "kwargs": {
                "renderMode": "ascii_shape",
                "spriteNames": ["e_grate"],
                "spriteShapes": [shapes.E_GRATE],
                "palettes": [shapes.GRATE_PALETTE],
                "noRotates": [False]
            }
        },
    ]
}

SE_GRATE = {
    "name": "se_grate",
    "components": [
        {
            "component": "StateManager",
            "kwargs": {
                "initialState": "se_grate",
                "stateConfigs": [{
                    "state": "se_grate",
                    "layer": "background",
                    "sprite": "se_grate",
                }],
            }
        },
        {"component": "Transform"},
        {
            "component": "Appearance",
            "kwargs": {
                "renderMode": "ascii_shape",
                "spriteNames": ["se_grate"],
                "spriteShapes": [shapes.SE_GRATE],
                "palettes": [shapes.GRATE_PALETTE],
                "noRotates": [False]
            }
        },
    ]
}

S_GRATE = {
    "name": "s_grate",
    "components": [
        {
            "component": "StateManager",
            "kwargs": {
                "initialState": "s_grate",
                "stateConfigs": [{
                    "state": "s_grate",
                    "layer": "background",
                    "sprite": "s_grate",
                }],
            }
        },
        {"component": "Transform"},
        {
            "component": "Appearance",
            "kwargs": {
                "renderMode": "ascii_shape",
                "spriteNames": ["s_grate"],
                "spriteShapes": [shapes.S_GRATE],
                "palettes": [shapes.GRATE_PALETTE],
                "noRotates": [False]
            }
        },
    ]
}

SW_GRATE = {
    "name": "sw_grate",
    "components": [
        {
            "component": "StateManager",
            "kwargs": {
                "initialState": "sw_grate",
                "stateConfigs": [{
                    "state": "sw_grate",
                    "layer": "background",
                    "sprite": "sw_grate",
                }],
            }
        },
        {"component": "Transform"},
        {
            "component": "Appearance",
            "kwargs": {
                "renderMode": "ascii_shape",
                "spriteNames": ["sw_grate"],
                "spriteShapes": [shapes.SW_GRATE],
                "palettes": [shapes.GRATE_PALETTE],
                "noRotates": [False]
            }
        },
    ]
}

GLASS_WALL = {
    "name": "glass_wall",
    "components": [
        {
            "component": "StateManager",
            "kwargs": {
                "initialState": "glass_wall",
                "stateConfigs": [{
                    "state": "glass_wall",
                    "layer": "upperPhysical",
                    "sprite": "glass_wall",
                }],
            }
        },
        {"component": "Transform"},
        {
            "component": "Appearance",
            "kwargs": {
                "renderMode": "ascii_shape",
                "spriteNames": ["glass_wall"],
                "spriteShapes": [shapes.GLASS_WALL],
                "palettes": [shapes.GLASS_PALETTE],
                "noRotates": [False]
            }
        },
    ]
}

FILL = {
    "name": "fill",
    "components": [
        {
            "component": "StateManager",
            "kwargs": {
                "initialState": "fill",
                "stateConfigs": [{
                    "state": "fill",
                    "layer": "upperPhysical",
                    "sprite": "fill",
                }],
            }
        },
        {"component": "Transform"},
        {
            "component": "Appearance",
            "kwargs": {
                "renderMode": "ascii_shape",
                "spriteNames": ["fill"],
                "spriteShapes": [shapes.FILL],
                "palettes": [{"i": (58, 68, 102, 255),}],
                "noRotates": [False]
            }
        },
    ]
}

TILED_FLOOR = {
    "name": "tiled_floor",
    "components": [
        {
            "component": "StateManager",
            "kwargs": {
                "initialState": "tiled_floor",
                "stateConfigs": [{
                    "state": "tiled_floor",
                    "layer": "background",
                    "sprite": "tiled_floor",
                }],
            }
        },
        {"component": "Transform"},
        {
            "component": "Appearance",
            "kwargs": {
                "renderMode": "ascii_shape",
                "spriteNames": ["tiled_floor"],
                "spriteShapes": [shapes.TILED_FLOOR_GREY],
                "palettes": [{"o": (204, 199, 192, 255),
                              "-": (194, 189, 182, 255),}],
                "noRotates": [False]
            }
        },
    ]
}

WOOD_FLOOR = {
    "name": "wood_floor",
    "components": [
        {
            "component": "StateManager",
            "kwargs": {
                "initialState": "wood_floor",
                "stateConfigs": [{
                    "state": "wood_floor",
                    "layer": "background",
                    "sprite": "wood_floor",
                }],
            }
        },
        {"component": "Transform"},
        {
            "component": "Appearance",
            "kwargs": {
                "renderMode": "ascii_shape",
                "spriteNames": ["wood_floor"],
                "spriteShapes": [shapes.WOOD_FLOOR],
                "palettes": [shapes.WOOD_FLOOR_PALETTE],
                "noRotates": [False]
            }
        },
    ]
}

METAL_FLOORING = {
    "name": "metal_flooring",
    "components": [
        {
            "component": "StateManager",
            "kwargs": {
                "initialState": "metal_flooring",
                "stateConfigs": [{
                    "state": "metal_flooring",
                    "layer": "background",
                    "sprite": "metal_flooring",
                }],
            }
        },
        {"component": "Transform"},
        {
            "component": "Appearance",
            "kwargs": {
                "renderMode": "ascii_shape",
                "spriteNames": ["metal_flooring"],
                "spriteShapes": [shapes.METAL_TILE],
                "palettes": [shapes.METAL_FLOOR_PALETTE],
                "noRotates": [False]
            }
        },
    ]
}

METAL_PANEL_FLOORING = {
    "name": "metal_panel_flooring",
    "components": [
        {
            "component": "StateManager",
            "kwargs": {
                "initialState": "metal_panel_flooring",
                "stateConfigs": [{
                    "state": "metal_panel_flooring",
                    "layer": "background",
                    "sprite": "metal_panel_flooring",
                }],
            }
        },
        {"component": "Transform"},
        {
            "component": "Appearance",
            "kwargs": {
                "renderMode": "ascii_shape",
                "spriteNames": ["metal_panel_flooring"],
                "spriteShapes": [shapes.METAL_PANEL],
                "palettes": [shapes.METAL_PANEL_FLOOR_PALETTE],
                "noRotates": [False]
            }
        },
    ]
}

CHECKERED_FLOORING = {
    "name": "checkered_flooring",
    "components": [
        {
            "component": "StateManager",
            "kwargs": {
                "initialState": "checkered_flooring",
                "stateConfigs": [{
                    "state": "checkered_flooring",
                    "layer": "background",
                    "sprite": "checkered_flooring",
                }],
            }
        },
        {"component": "Transform"},
        {
            "component": "Appearance",
            "kwargs": {
                "renderMode": "ascii_shape",
                "spriteNames": ["checkered_flooring"],
                "spriteShapes": [shapes.CHECKERED_TILE],
                "palettes": [{"X": (120, 108, 108, 255),
                              "x": (115, 103, 103, 255),}],
                "noRotates": [False]
            }
        },
    ]
}

TILED_FLOORING1 = {
    "name": "tiled_flooring1",
    "components": [
        {
            "component": "StateManager",
            "kwargs": {
                "initialState": "tiled_flooring1",
                "stateConfigs": [{
                    "state": "tiled_flooring1",
                    "layer": "background",
                    "sprite": "tiled_flooring1",
                }],
            }
        },
        {"component": "Transform"},
        {
            "component": "Appearance",
            "kwargs": {
                "renderMode": "ascii_shape",
                "spriteNames": ["tiled_flooring1"],
                "spriteShapes": [shapes.TILE1],
                "palettes": [shapes.TILE_FLOOR_PALETTE],
                "noRotates": [False]
            }
        },
    ]
}

TILED_FLOORING2 = {
    "name": "tiled_flooring2",
    "components": [
        {
            "component": "StateManager",
            "kwargs": {
                "initialState": "tiled_flooring2",
                "stateConfigs": [{
                    "state": "tiled_flooring2",
                    "layer": "background",
                    "sprite": "tiled_flooring2",
                }],
            }
        },
        {"component": "Transform"},
        {
            "component": "Appearance",
            "kwargs": {
                "renderMode": "ascii_shape",
                "spriteNames": ["tiled_flooring2"],
                "spriteShapes": [shapes.TILE2],
                "palettes": [shapes.TILE_FLOOR_PALETTE],
                "noRotates": [False]
            }
        },
    ]
}

THRESHOLD = {
    "name": "threshold",
    "components": [
        {
            "component": "StateManager",
            "kwargs": {
                "initialState": "threshold",
                "stateConfigs": [{
                    "state": "threshold",
                    "layer": "background",
                    "sprite": "threshold",
                }],
            }
        },
        {"component": "Transform"},
        {
            "component": "Appearance",
            "kwargs": {
                "renderMode": "ascii_shape",
                "spriteNames": ["threshold"],
                "spriteShapes": [shapes.THRESHOLD],
                "palettes": [{"X": (92, 95, 92, 255),
                              "x": (106, 108, 106, 255),}],
                "noRotates": [False]
            }
        },
    ]
}

# Functional components for Hidden Agenda

SPAWN_POINT = {
    "name": "spawn_point",
    "components": [
        {
            "component": "StateManager",
            "kwargs": {
                "initialState": "spawnPoint",
                "stateConfigs": [{
                    "state": "spawnPoint",
                    "groups": ["spawnPoints"],
                }],
            }
        },
        {
            "component": "Transform",
        },
    ]
}

VOTING_SPAWN_POINT = {
    "name": "voting_spawn_point",
    "components": [
        {
            "component": "StateManager",
            "kwargs": {
                "initialState": "votingSpawnPoint",
                "stateConfigs": [{
                    "state": "votingSpawnPoint",
                    "groups": ["votingSpawnPoints"],
                }],
            }
        },
        {
            "component": "Transform",
        },
    ]
}

TELEPORT_SPAWN_POINT = {
    "name": "teleport_spawn_point",
    "components": [
        {
            "component": "StateManager",
            "kwargs": {
                "initialState": "teleportSpawnPoint",
                "stateConfigs": [{
                    "state": "teleportSpawnPoint",
                    "groups": ["teleportSpawnPoints"],
                }],
            }
        },
        {
            "component": "Transform",
        },
    ]
}


def get_gem_prefab(crewmate_pseudoreward: float):
  return {
      "name": "gem",
      "components": [
          {
              "component": "StateManager",
              "kwargs": {
                  "initialState": "gem",
                  "stateConfigs": [{
                      "state": "gem",
                      "layer": "lowerPhysical",
                      "sprite": "Gem",
                  }, {
                      "state": "gemWait",
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
                  "spriteNames": ["Gem",],
                  "spriteShapes": [shapes.SMALL_SPHERE],
                  "palettes": [shapes.MOULD_PALETTE],
                  "noRotates": [True],
              }
          },
          {
              "component": "Collectable",
              "kwargs": {
                  "liveState": "gem",
                  "waitState": "gemWait",
                  "rewardForCollecting_crewmate": crewmate_pseudoreward,
                  "rewardForCollecting_impostor": 0.0,
                  "regrowRate": 0.001,
              }
          },
      ]
  }


def get_gem_deposit_prefab(crewmate_pseudoreward: float):
  return {
      "name": "gem_deposit",
      "components": [
          {
              "component": "StateManager",
              "kwargs": {
                  "initialState": "gemDeposit",
                  "stateConfigs": [{
                      "state": "gemDeposit",
                      "layer": "lowerPhysical",
                      "sprite": "GemDeposit",
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
                  "spriteNames": ["GemDeposit"],
                  "spriteShapes": ["x"],
                  "palettes": [shapes.GRATE_PALETTE],
                  "noRotates": [False],
              }
          },
          {
              "component": "Deposit",
              "kwargs": {
                  "crewmateReward": crewmate_pseudoreward,
                  "impostorReward": 0.0,
              }
          }
      ]
  }


def create_player(player_idx: int, role: str, num_players: int,
                  pseudoreward_for_freezing: float,
                  pseudoreward_for_being_frozen: float):
  """Create a prefab for a Player (Impostor or Crewmate).

  Args:
    player_idx: The index of this player.
    role: Whether this player will be a `crewmate` or an `impostor`.
    num_players: The number of players in the environment.
    pseudoreward_for_freezing: Peudoreward given when the impostor successfully
      freezes a crewmate.
    pseudoreward_for_being_frozen: Pseudoreward (usually negative) given to a
      crewmate that was frozen by the impostor.

  Returns:
    A prefab (dictionary) for a Player.
  """
  # Lua is 1-indexed.
  lua_index = player_idx + 1
  live_state_name = f"player{lua_index}"
  avatar_sprite_name = f"avatarSprite{lua_index}"

  if role == "impostor":
    sprite_map = {avatar_sprite_name: f"Player_impostor{lua_index}"}
  else:
    sprite_map = {}

  player = {
      "name": "player",
      "components": [
          {
              "component": "StateManager",
              "kwargs": {
                  "initialState": live_state_name,
                  "stateConfigs": [
                      {"state": live_state_name,
                       "layer": "upperPhysical",
                       "sprite": avatar_sprite_name,
                       "contact": "avatar"},
                      {"state": "playerWait"},
                      {"state": "playerBody",
                       "layer": "upperPhysical",
                       "sprite": "Player_tagged"},
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
                  "spriteNames": [avatar_sprite_name, "Player_tagged"],
                  "spriteShapes": [shapes.CUTE_AVATAR,
                                   shapes.CUTE_AVATAR_FROZEN],
                  "palettes": [
                      shapes.get_palette(HIDDEN_AGENDA_COLORS[player_idx]),
                      shapes.get_palette(HIDDEN_AGENDA_COLORS[player_idx])
                  ],
                  "noRotates": [True]
              }
          },
          {
              "component": "AdditionalPlayerSprites",
              "kwargs": {
                  "renderMode":
                      "ascii_shape",
                  "customSpriteNames": [
                      "Player_impostor" + str(i + 1) for i in range(num_players)
                  ],
                  "customSpriteShapes": [shapes.CUTE_AVATAR_W_BUBBLE] *
                                        num_players,
                  "customPalettes": [
                      shapes.get_palette(HIDDEN_AGENDA_COLORS[i])
                      for i in range(num_players)
                  ],
                  "customNoRotates": [True]
              }
          },
          {
              "component": "Avatar",
              "kwargs": {
                  "index": lua_index,
                  "spawnGroup": "spawnPoints",
                  "aliveState": live_state_name,
                  "waitState": "playerWait",
                  "actionOrder": ["move", "turn", "tag", "vote"],
                  "actionSpec": {
                      "move": {"default": 0, "min": 0, "max": len(COMPASS)},
                      "turn": {"default": 0, "min": -1, "max": 1},
                      "tag": {"default": 0, "min": 0, "max": 1},
                      "vote": {"default": 0, "min": 0, "max": num_players + 1},
                  },
                  "view": {
                      "left": 5,
                      "right": 5,
                      "forward": 9,
                      "backward": 1,
                      "centered": False
                  },
                  "spriteMap": sprite_map,
              },
          },
          {
              "component": "Role",
              "kwargs": {
                  "frozenState": "playerBody",
                  "role": role,  # `crewmate` or `impostor`.
              }
          },
          {
              "component": "Inventory",
              "kwargs": {
                  "max_gems": 1,
              }
          },
          {
              "component": "AdditionalObserver",
              "kwargs": {
                  "num_players": num_players,
              }
          },
          {
              "component": "Tagger",
              "kwargs": {
                  "cooldownTime": 50,
                  "beamLength": 2,
                  "beamRadius": 2,
                  "penaltyForBeingTagged": pseudoreward_for_being_frozen,
                  "rewardForTagging": pseudoreward_for_freezing,
                  "removeHitPlayer": "freeze",
              }
          },
          {
              "component": "ReadyToShootObservation",
              "kwargs": {
                  "zapperComponent": "Tagger",
              },
          },
          {
              "component": "Voting",
              "kwargs": {
                  "spawnGroup": "votingSpawnPoints",
                  "votingActive": False,
                  "votingMethod": "deliberation",
                  "votingValues": {},
              }
          },
      ]
  }
  if _ENABLE_DEBUG_OBSERVATIONS:
    player["components"].append({
        "component": "LocationObserver",
        "kwargs": {"objectIsAvatar": True, "alsoReportOrientation": True},
    })

  return player


def create_prefabs(
    crewmate_collect_pseudoreward: float,
    crewmate_deposit_pseudoreward: float,
):
  """Create prefabs dictionary from individual prefabs.

  Args:
    crewmate_collect_pseudoreward: Pseudoreward given to crewmates when they
      collect a gem.
    crewmate_deposit_pseudoreward: Pseudoreward given to crewmates when they
      deposit a gem.

  Returns:
    A dictionary of prefabs for components in the environment.
  """

  # PREFABS is a dictionary mapping names to template game objects that can
  # be cloned and placed in multiple locations accoring to an ascii map.
  prefabs = {
      "nw_wall_corner": NW_WALL_CORNER,
      "ne_wall_corner": NE_WALL_CORNER,
      "sw_wall_corner": SW_WALL_CORNER,
      "n_ship_solid_wall": N_SHIP_SOLID_WALL,
      "e_ship_solid_wall": E_SHIP_SOLID_WALL,
      "s_ship_solid_wall": S_SHIP_SOLID_WALL,
      "w_ship_solid_wall": W_SHIP_SOLID_WALL,
      "wall_north": WALL_NORTH,
      "se_wall_corner": SE_WALL_CORNER,
      "tcoupling_e": TCOUPLING_E,
      "tcoupling_w": TCOUPLING_W,
      "tcoupling_n": TCOUPLING_N,
      "tcoupling_s": TCOUPLING_S,
      "wall_west": WALL_WEST,
      "nw_grate": NW_GRATE,
      "n_grate": N_GRATE,
      "ne_grate": NE_GRATE,
      "w_grate": W_GRATE,
      "inner_grate": INNER_GRATE,
      "e_grate": E_GRATE,
      "se_grate": SE_GRATE,
      "s_grate": S_GRATE,
      "sw_grate": SW_GRATE,
      "glass_wall": GLASS_WALL,
      "fill": FILL,
      "tiled_floor": TILED_FLOOR,
      "tiled_flooring1": TILED_FLOORING1,
      "tiled_flooring2": TILED_FLOORING2,
      "wood_flooring": WOOD_FLOOR,
      "metal_flooring": METAL_FLOORING,
      "metal_panel_flooring": METAL_PANEL_FLOORING,
      "checkered_flooring": CHECKERED_FLOORING,
      "threshold": THRESHOLD,
      "gem": get_gem_prefab(crewmate_collect_pseudoreward),
      "spawn_point": SPAWN_POINT,
      "voting_spawn_point": VOTING_SPAWN_POINT,
      "teleport_spawn_point": TELEPORT_SPAWN_POINT,
      "gem_deposit": get_gem_deposit_prefab(crewmate_deposit_pseudoreward),
  }
  return prefabs

# Primitive action components.
# pylint: disable=bad-whitespace
# pyformat: disable
NOOP       = {"move": 0, "turn": 0,  "tag": 0, "vote": 0}
FORWARD    = {"move": 1, "turn": 0,  "tag": 0, "vote": 0}
STEP_RIGHT = {"move": 2, "turn": 0,  "tag": 0, "vote": 0}
BACKWARD   = {"move": 3, "turn": 0,  "tag": 0, "vote": 0}
STEP_LEFT  = {"move": 4, "turn": 0,  "tag": 0, "vote": 0}
TURN_LEFT  = {"move": 0, "turn": -1, "tag": 0, "vote": 0}
TURN_RIGHT = {"move": 0, "turn": 1,  "tag": 0, "vote": 0}
TAG        = {"move": 0, "turn": 0,  "tag": 1, "vote": 0}
# pylint: enable=bad-whitespace
# pyformat: enable


def create_action_set(num_players):
  """Create the action set for the agents."""
  action_set = [
      NOOP,
      FORWARD, BACKWARD,
      STEP_LEFT, STEP_RIGHT,
      TURN_LEFT, TURN_RIGHT,
      TAG,
  ]
  # vote for each player and no-vote.
  for player in range(1, num_players+2):
    vote = copy.deepcopy(NOOP)
    vote["vote"] = player
    action_set.append(vote)
  return action_set


def get_config():
  """Default configuration for the Hidden Agenda level."""
  config = config_dict.ConfigDict()

  # Specify the number of players to particate in each episode (optional).
  config.recommended_num_players = MANDATED_NUM_PLAYERS

  # Configurable pseudorewards.
  # The canonical substrate requires these pseudorewards to be 0. However,
  # you can set them to something else to improve training. For example, you'd
  # use `crewmate_collect_pseudoreward=0.25` and
  # `crewmate_deposit_pseudoreward=0.25` to aid the crewmates in learning to
  # collect and deposit gems. Also, you can set `pseudoreward_for_freezing=1`
  # and `pseudoreward_for_being_frozen=-1` to help players learn about freezing
  # mechanics.
  config.pseudorewards = dict(
      crewmate_collect_pseudoreward=0.0,
      crewmate_deposit_pseudoreward=0.0,
      pseudoreward_for_freezing=0.0,
      pseudoreward_for_being_frozen=0.0,
  )

  config.scene_prefab = {
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
              "component":
                  "Progress",
              "kwargs":
                  config_dict.ConfigDict({
                      "num_players":
                          MANDATED_NUM_PLAYERS,
                      "goal": 32,
                      "potential_pseudorewards": False,
                      "crewmate_task_reward": 1,
                      "impostor_task_reward": -1,
                      "crewmate_tag_reward": -1,
                      "impostor_tag_reward": 1,
                      "crewmate_vote_reward": 1,
                      "impostor_vote_reward": -1,
                      "incorrect_vote_reward": 0,
                      "correct_vote_reward": 0,
                      "step_reward": 0,
                      "teleport_spawn_group":
                          "teleportSpawnPoints",
                      "voting_params": {
                          "type": "deliberation",
                          "votingPhaseCooldown": 25,
                          "votingFrameFrequency": 200,
                          "taggingTriggerVoting": True,
                      }
                  })
          },
      ]
  }
  # The voting matrix metric is always used.
  metrics = [{
      "name": "VOTING",
      "type": "tensor.DoubleTensor",
      "shape": (MANDATED_NUM_PLAYERS, MANDATED_NUM_PLAYERS + 2),
      "component": "Progress",
      "variable": "votingMatrix",
  }]
  if _ENABLE_DEBUG_OBSERVATIONS:
    metrics.append({
        "name": "GLOBAL_PROGRESS",
        "type": "tensor.DoubleTensor",
        "shape": (1,),
        "component": "Progress",
        "variable": "progress_bar",
    })
    metrics.append({
        "name": "IDENTITIES",
        "type": "tensor.DoubleTensor",
        "shape": (MANDATED_NUM_PLAYERS,),
        "component": "Progress",
        "variable": "identity_tensor"
    })
    metrics.append({
        "name": "VOTING",
        "type": "tensor.DoubleTensor",
        "shape": (MANDATED_NUM_PLAYERS, MANDATED_NUM_PLAYERS + 2),
        "component": "Progress",
        "variable": "votingMatrix"
    })

  # Add the global metrics reporter
  config.scene_prefab["components"].append({
      "component": "GlobalMetricReporter",
      "kwargs": {
          "metrics": metrics
      }
  })

  # Action set configuration.
  config.action_set = create_action_set(MANDATED_NUM_PLAYERS)

  # Observation format configuration.
  config.individual_observation_names = [
      "RGB",
      "INVENTORY",
      "READY_TO_SHOOT",
      "VOTING",
  ]
  config.global_observation_names = [
      "WORLD.RGB",
  ]

  # The specs of the environment (from a single-agent perspective).
  config.action_spec = specs.action(len(config.action_set))
  config.timestep_spec = specs.timestep({
      "RGB": specs.OBSERVATION["RGB"],
      "INVENTORY": specs.inventory(1),
      "READY_TO_SHOOT": specs.OBSERVATION["READY_TO_SHOOT"],
      "VOTING": specs.float64(MANDATED_NUM_PLAYERS, MANDATED_NUM_PLAYERS + 2),
      # Debug only (do not use the following observations in policies).
      "WORLD.RGB": specs.rgb(176, 264),
  })

  # The roles assigned to each player.
  config.valid_roles = frozenset({"crewmate",
                                  "impostor",})
  config.default_player_roles = ("crewmate",) * 4 + ("impostor",)

  return config


def build(
    roles: Sequence[str],
    config: config_dict.ConfigDict,
) -> Mapping[str, Any]:
  """Build the hidden_agenda substrate given player preferences."""
  # Build avatars.
  num_players = len(roles)
  avatar_objects = []
  for player_idx, role in enumerate(roles):
    # Create an avatar with the correct role.
    avatar_objects.append(create_player(
        player_idx=player_idx,
        role=role,
        num_players=num_players,
        pseudoreward_for_freezing=
        config.pseudorewards.pseudoreward_for_freezing,
        pseudoreward_for_being_frozen=
        config.pseudorewards.pseudoreward_for_being_frozen))
  substrate_definition = dict(
      levelName="hidden_agenda",
      levelDirectory="meltingpot/lua/levels",
      maxEpisodeLengthFrames=3000,
      spriteSize=8,
      numPlayers=num_players,
      topology="BOUNDED",  # Choose from ["BOUNDED", "TORUS"],
      simulation={
          "map": ASCII_MAP,
          "gameObjects": avatar_objects,
          "prefabs": create_prefabs(
              config.pseudorewards.crewmate_collect_pseudoreward,
              config.pseudorewards.crewmate_deposit_pseudoreward,
          ),
          "charPrefabMap": CHAR_PREFAB_MAP,
          "scene": config.scene_prefab,
      },
  )
  return substrate_definition
