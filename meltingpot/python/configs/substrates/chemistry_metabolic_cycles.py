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
"""Configuration for Chemistry: Metabolic Cycles.

Example video: https://youtu.be/oFK9VujhpeI

Individuals benefit from two different food generating reaction cycles. Both
will run on their own (autocatalytically), but require energy to continue.
Bringing together side products from both cycles generates new energy such that
the cycles can continue. The population needs to keep both cycles running to get
high rewards.

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
DEFAULT_REWARDING_REACTIONS = {"MetabolizeFood1": 1,
                               "MetabolizeFood2": 1,
                               "MetabolizeXY": 10}

# Define the default reaction query configuration. It can be overridden on a per
# compount basis.
DEFAULT_REACTION_CONFIG = {"radius": 1, "query_type": "disc"}

REACTIVITY_LEVELS = {
    "ground": {"background": 0.00001,
               "low": 0.005,
               "medium": 0.001,
               "high": 0.9},
    "stomach": {"background": 0.0,
                "low": 0.0025,
                "medium": 0.25,
                "high": 0.9},
}


def dissipate_when_paired(g, reaction_name, compound):
  g.add_node(reaction_name, reaction=True)
  # Reactants:
  g.add_edge(compound, reaction_name)
  g.add_edge(compound, reaction_name)
  # Products:
  g.add_edge(reaction_name, "empty")
  g.add_edge(reaction_name, "empty")


def cycle(g, reaction_prefix, intermediates, product, secondary_product=None,
          food="food"):
  """Add a reaction cycle."""
  # Reaction cycle x, reaction 1
  reaction_1 = "{}1{}".format(reaction_prefix, product)
  g.add_node(reaction_1, reaction=True)
  # Reactants:
  g.add_edge(intermediates[0], reaction_1)
  g.add_edge(intermediates[1], reaction_1)
  g.add_edge("empty", reaction_1)
  # Products:
  g.add_edge(reaction_1, intermediates[1])
  g.add_edge(reaction_1, intermediates[2])
  g.add_edge(reaction_1, food)

  # Reaction cycle x, reaction 2
  reaction_2 = "{}2{}".format(reaction_prefix, product)
  g.add_node(reaction_2, reaction=True)
  # Reactants:
  g.add_edge(intermediates[1], reaction_2)
  g.add_edge(intermediates[2], reaction_2)
  g.add_edge("energy", reaction_2)
  # Products:
  g.add_edge(reaction_2, intermediates[2])
  g.add_edge(reaction_2, intermediates[0])
  g.add_edge(reaction_2, "energy")

  # Reaction cycle x, reaction 3
  reaction_3 = "{}3{}".format(reaction_prefix, product)
  g.add_node(reaction_3, reaction=True)
  # Reactants:
  g.add_edge(intermediates[2], reaction_3)
  g.add_edge(intermediates[0], reaction_3)
  g.add_edge("empty", reaction_3)
  if secondary_product is not None:
    g.add_edge("empty", reaction_3)
  # Products:
  g.add_edge(reaction_3, intermediates[0])
  g.add_edge(reaction_3, intermediates[1])
  g.add_edge(reaction_3, product)
  if secondary_product is not None:
    g.add_edge(reaction_3, secondary_product)


def make_graph():
  """User defined graph construction function using networkx."""
  # Note: You can copy-paste this function into colab to visualize the graph.
  g = nx.MultiDiGraph()
  # First add the "empty" and "activated" nodes, which are always present.
  graph_utils.add_system_nodes(g)

  cycle(g, "R",
        intermediates=["ax", "bx", "cx"],
        product="x",
        secondary_product="iy",
        food="food1")
  cycle(g, "R",
        intermediates=["ay", "by", "cy"],
        product="y",
        secondary_product="ix",
        food="food2")

  # Inhibit x with a product of the y-producing cycle.
  g.add_node("InhibitX", reaction=True)
  # Reactants:
  g.add_edge("x", "InhibitX")
  g.add_edge("ix", "InhibitX")
  # Products:
  g.add_edge("InhibitX", "empty")
  g.add_edge("InhibitX", "empty")

  # Inhibit y with a product of the x-producing cycle.
  g.add_node("InhibitY", reaction=True)
  # Reactants:
  g.add_edge("y", "InhibitY")
  g.add_edge("iy", "InhibitY")
  # Products:
  g.add_edge("InhibitY", "empty")
  g.add_edge("InhibitY", "empty")

  # Food can be metabolized in the stomach.
  g.add_node("MetabolizeFood1", reaction=True)
  # Reactants:
  g.add_edge("food1", "MetabolizeFood1")
  # Products:
  g.add_edge("MetabolizeFood1", "empty")

  # Food can be metabolized in the stomach.
  g.add_node("MetabolizeFood2", reaction=True)
  # Reactants:
  g.add_edge("food2", "MetabolizeFood2")
  # Products:
  g.add_edge("MetabolizeFood2", "empty")

  # Food spontaneously appears from time to time.
  g.add_node("SpawnFood1", reaction=True)
  # Reactants:
  g.add_edge("empty", "SpawnFood1")
  # Products:
  g.add_edge("SpawnFood1", "food1")

  # Food spontaneously appears from time to time.
  g.add_node("SpawnFood2", reaction=True)
  # Reactants:
  g.add_edge("empty", "SpawnFood2")
  # Products:
  g.add_edge("SpawnFood2", "food2")

  # x and y can be combined to produce energy.
  g.add_node("MetabolizeXY", reaction=True)
  # Reactants:
  g.add_edge("x", "MetabolizeXY")
  g.add_edge("y", "MetabolizeXY")
  # Products:
  g.add_edge("MetabolizeXY", "energy")
  g.add_edge("MetabolizeXY", "energy")

  # Energy spontaneously dissipates.
  g.add_node("DissipateEnergy", reaction=True)
  # Reactants:
  g.add_edge("energy", "DissipateEnergy")
  # Products:
  g.add_edge("DissipateEnergy", "empty")

  # Prevent inhibitors from accumulating by dissipating them whenever they pair.
  dissipate_when_paired(g, "DissipateIX", "ix")
  dissipate_when_paired(g, "DissipateIY", "iy")

  # Properties of compounds
  # Color:
  g.nodes["ax"]["color"] = (153, 204, 255)  # blue 1
  g.nodes["bx"]["color"] = (102, 204, 255)  # blue 2
  g.nodes["cx"]["color"] = (51, 153, 255)  # blue 3

  g.nodes["ay"]["color"] = (102, 255, 153)  # green 1
  g.nodes["by"]["color"] = (102, 255, 102)  # green 2
  g.nodes["cy"]["color"] = (0, 255, 0)  # green 3

  g.nodes["x"]["color"] = (0, 51, 204)  # dark blue
  g.nodes["y"]["color"] = (0, 51, 0)  # dark green
  g.nodes["food1"]["color"] = (255, 255, 0)  # yellow
  g.nodes["food2"]["color"] = (255, 215, 0)  # gold
  g.nodes["energy"]["color"] = (255, 0, 0)  # red

  g.nodes["ix"]["color"] = (102, 153, 153)  # greyish green
  g.nodes["iy"]["color"] = (51, 102, 153)  # greyish blue

  # Reactivity:
  g.nodes["ax"]["reactivity"] = "high"
  g.nodes["bx"]["reactivity"] = "high"
  g.nodes["cx"]["reactivity"] = "high"

  g.nodes["ay"]["reactivity"] = "high"
  g.nodes["by"]["reactivity"] = "high"
  g.nodes["cy"]["reactivity"] = "high"

  g.nodes["x"]["reactivity"] = "medium"
  g.nodes["y"]["reactivity"] = "medium"

  g.nodes["ix"]["reactivity"] = "high"
  g.nodes["iy"]["reactivity"] = "high"

  g.nodes["food1"]["reactivity"] = "medium"
  g.nodes["food2"]["reactivity"] = "medium"
  g.nodes["energy"]["reactivity"] = "low"
  g.nodes["empty"]["reactivity"] = "background"

  # The following commented line documents how to set the query config for a
  # specific compound, overriding the default query configuration.
  # g.nodes["food1"]["query_config"] = {"radius": 3, "queryType": "diamond"}

  return g

ASCII_MAP = """
~~~~~~~~~~~a~~~~~~~~~~~~~
~~~~~~~~c~~~~~~~~~~~~~~~~
~~~~~~~~~~~b~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~1~~~
~~~~~~~~~~~~~~~~~~~~~~~~~
1~~3~~~~hhhhhhh~~~~~3~~2~
~~~~~~~~~~~~~~~~~~~~~~~~~
~2~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~c~~~~~~~~~~~~~~
~~~~~~~~~~~~a~~~~~~~~~~~~
~~~~~~~~~~b~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~
"""

# `prefab` determines which compound to use for each `char` in the ascii map.
CHAR_PREFAB_MAP = {
    "~": "empty",
    "a": "ax",
    "b": "bx",
    "c": "cx",
    "1": "ay",
    "2": "by",
    "3": "cy",
    "h": "energy",
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
      "episodeLengthFrames": 1000,
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
