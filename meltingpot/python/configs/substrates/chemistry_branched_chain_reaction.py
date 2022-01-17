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
"""Configuration for Chemistry: Branched Chain Reaction.

Example video: https://youtu.be/ZhRB-_ruoH8

Individuals are rewarded by driving chemical reactions involving specific
molecules. They need to suitably coordinate the alternation of branches while
keeping certain elements apart that would otherwise react unfavourably, so as
not to run out of the molecules required for continuing the chain. Combining
molecules efficiently requires coordination but can also lead to exclusion of
players.

Reactions are defined by a directed graph. Reactant nodes project into reaction
nodes, which project out to product nodes. Reactions occur stochastically when
all reactants are brought near one another. Agents can carry a single molecule
around the map with them at a time. Agents are rewarded when a specific reaction
occurs that involves the molecule they are currently carrying (as either a
reactant or a product).
"""

import copy

from ml_collections import config_dict
import networkx as nx
from meltingpot.python.utils.substrates import colors
from meltingpot.python.utils.substrates import game_object_utils
from meltingpot.python.utils.substrates import reaction_graph_utils as graph_utils
from meltingpot.python.utils.substrates import shapes


# Map reaction to rewards.
DEFAULT_REWARDING_REACTIONS = {"LowerX": 1,
                               "HigherX": 1,
                               "LowerY": 1,
                               "HigherY": 1,
                               "DestroyB": -1,
                               "DestroyD": -1}

# Define the default reaction query configuration. It can be overridden on a per
# compount basis.
DEFAULT_REACTION_CONFIG = {"radius": 1, "query_type": "disc"}

REACTIVITY_LEVELS = {
    "ground": {"background": 0.00001,
               "low": 0.88,
               "medium": 0.89,
               "high": 0.9},
    "stomach": {"background": 0.0,
                "low": 0.88,
                "medium": 0.89,
                "high": 0.9},
}


def make_graph():
  """User defined graph construction function using networkx."""
  # Note: You can copy-paste this function into colab to visualize the graph.
  g = nx.MultiDiGraph()
  # First add the "empty" and "activated" nodes, which are always present.
  graph_utils.add_system_nodes(g)

  g.add_node("LowerX", reaction=True)
  # Reactants:
  g.add_edge("ax", "LowerX")
  g.add_edge("bx", "LowerX")
  # Products:
  g.add_edge("LowerX", "c")
  g.add_edge("LowerX", "by")

  g.add_node("LowerY", reaction=True)
  # Reactants:
  g.add_edge("ay", "LowerY")
  g.add_edge("by", "LowerY")
  # Products:
  g.add_edge("LowerY", "c")
  g.add_edge("LowerY", "bx")

  # Inhibit y with a product of the x-producing cycle.
  g.add_node("HigherX", reaction=True)
  # Reactants:
  g.add_edge("c", "HigherX")
  g.add_edge("dx", "HigherX")
  # Products:
  g.add_edge("HigherX", "ay")
  g.add_edge("HigherX", "dy")

  g.add_node("HigherY", reaction=True)
  g.add_edge("c", "HigherY")
  g.add_edge("dy", "HigherY")
  # Products:
  g.add_edge("HigherY", "ax")
  g.add_edge("HigherY", "dx")

  g.add_node("DestroyB", reaction=True)
  g.add_edge("bx", "DestroyB")
  g.add_edge("by", "DestroyB")
  # Products:
  g.add_edge("DestroyB", "empty")
  g.add_edge("DestroyB", "empty")

  g.add_node("DestroyD", reaction=True)
  g.add_edge("dx", "DestroyD")
  g.add_edge("dy", "DestroyD")
  # Products:
  g.add_edge("DestroyD", "empty")
  g.add_edge("DestroyD", "empty")

  # Properties of compounds
  # Color:
  g.nodes["ax"]["color"] = (153, 204, 255)  # blue 1
  g.nodes["bx"]["color"] = (102, 154, 255)  # blue 2
  g.nodes["dx"]["color"] = (201, 15, 255)  # blue 3 = purple

  g.nodes["ay"]["color"] = (102, 255, 153)  # green 1
  g.nodes["by"]["color"] = (52, 255, 102)  # green 2
  g.nodes["dy"]["color"] = (0, 255, 0)  # green 3

  g.nodes["c"]["color"] = (255, 0, 0)  # red

  # Reactivity:
  g.nodes["ax"]["reactivity"] = "low"
  g.nodes["bx"]["reactivity"] = "high"
  g.nodes["dx"]["reactivity"] = "high"

  g.nodes["ay"]["reactivity"] = "low"
  g.nodes["by"]["reactivity"] = "high"
  g.nodes["dy"]["reactivity"] = "high"

  g.nodes["c"]["reactivity"] = "medium"

  g.nodes["empty"]["reactivity"] = "background"

  # The following commented line documents how to set the query config for a
  # specific compound, overriding the default query configuration.
  # g.nodes["food1"]["query_config"] = {"radius": 3, "queryType": "diamond"}

  return g

ASCII_MAP = """
~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~b~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~
~~d~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~a~~~~~~1~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~3~~~
~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~2~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~
"""

# `prefab` determines which compound to use for each `char` in the ascii map.
CHAR_PREFAB_MAP = {
    "~": "empty",
    "a": "ax",
    "b": "bx",
    "d": "dx",
    "1": "ay",
    "2": "by",
    "3": "dy",
    "c": "c",
}

# PLAYER_COLOR_PALETTES is a list with each entry specifying the color to use
# for the player at the corresponding index.
NUM_PLAYERS_UPPER_BOUND = 60
PLAYER_COLOR_PALETTES = []
for i in range(NUM_PLAYERS_UPPER_BOUND):
  PLAYER_COLOR_PALETTES.append(shapes.get_palette(colors.palette[i]))

# Primitive action components.
# pylint: disable=bad-whitespace
# pyformat: disable
NOOP       = {"move": 0, "turn":  0, "ioAction": 0}
FORWARD    = {"move": 1, "turn":  0, "ioAction": 0}
STEP_RIGHT = {"move": 2, "turn":  0, "ioAction": 0}
BACKWARD   = {"move": 3, "turn":  0, "ioAction": 0}
STEP_LEFT  = {"move": 4, "turn":  0, "ioAction": 0}
TURN_LEFT  = {"move": 0, "turn": -1, "ioAction": 0}
TURN_RIGHT = {"move": 0, "turn":  1, "ioAction": 0}
IO_ACTION  = {"move": 0, "turn":  0, "ioAction": 1}
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
    IO_ACTION,
)

TARGET_SPRITE_SELF = {
    "name": "Self",
    "shape": shapes.AVATAR_DEFAULT,
    "palette": shapes.get_palette((50, 100, 200)),
    "noRotate": False,
}


def create_avatar_objects(prefabs, num_players):
  """Returns list of avatar objects of length 'num_players'."""
  avatar_objects = []
  additional_game_objects = []
  for player_idx in range(0, num_players):
    game_object = graph_utils.create_avatar_constant_self_view(
        DEFAULT_REWARDING_REACTIONS, player_idx, TARGET_SPRITE_SELF)
    avatar_objects.append(game_object)

    # Add the overlaid avatar stomach on top of each avatar.
    avatar_stomach = copy.deepcopy(prefabs["avatar_stomach"])
    game_object_utils.get_first_named_component(
        avatar_stomach,
        "AvatarStomach")["kwargs"]["playerIndex"] = player_idx + 1
    additional_game_objects.append(avatar_stomach)

  return avatar_objects, additional_game_objects


def get_lab2d_settings(num_players):
  """Returns the lab2d settings.

  Args:
    num_players: the number of players in the game.
  """

  # Must create compounds and reactions.
  compounds, reactions = graph_utils.graph_semantics(make_graph())

  avatar_object_templates = {
      "avatar": graph_utils.create_avatar(DEFAULT_REWARDING_REACTIONS),
      "avatar_stomach": graph_utils.create_stomach(
          compounds,
          REACTIVITY_LEVELS["stomach"],
          default_reaction_radius=DEFAULT_REACTION_CONFIG["radius"],
          default_reaction_query_type=DEFAULT_REACTION_CONFIG["query_type"],
          priority_mode=True),
  }
  cell_prefabs = {}
  cell_prefabs = graph_utils.add_compounds_to_prefabs_dictionary(
      cell_prefabs, compounds, REACTIVITY_LEVELS["ground"], sprites=True,
      default_reaction_radius=DEFAULT_REACTION_CONFIG["radius"],
      default_reaction_query_type=DEFAULT_REACTION_CONFIG["query_type"],
      priority_mode=True)

  avatar_objects, additional_objects = create_avatar_objects(
      avatar_object_templates, num_players)

  # Lua script configuration.
  lab2d_settings = {
      "levelName": "grid_land",
      "levelDirectory":
          "meltingpot/lua/levels",
      "numPlayers": num_players,
      "maxEpisodeLengthFrames": 1000,
      "spriteSize": 8,
      "topology": "BOUNDED",
      "simulation": {
          "map": ASCII_MAP,
          "gameObjects": avatar_objects + additional_objects,
          "scene": graph_utils.create_scene(reactions,
                                            stochastic_episode_ending=True),
          "prefabs": cell_prefabs,
          "charPrefabMap": CHAR_PREFAB_MAP,
      },
  }

  return lab2d_settings


def get_config():
  """Default configuration for training on the grid_land level."""
  config = config_dict.ConfigDict()

  # Basic configuration.
  config.num_players = 8

  config.lab2d_settings = get_lab2d_settings(config.num_players)

  # Action set configuration.
  config.action_set = ACTION_SET
  # Observation format configuration.
  config.individual_observation_names = [
      "RGB",
      "POSITION",
      "ORIENTATION",
  ]
  config.global_observation_names = [
      "WORLD.RGB",
  ]

  return config
