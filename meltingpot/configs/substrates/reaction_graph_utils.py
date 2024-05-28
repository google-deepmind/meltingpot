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
"""Library of functions for defining chemical motifs."""

from typing import Any, Dict

from meltingpot.utils.substrates import shapes
import networkx as nx
import numpy as np

EMPTY_COLOR = shapes.PETRI_DISH_PALETTE["@"]
WHITE_COLOR = (255, 255, 255, 255)  # A white color.

DIAMOND_SHAPE = """
xxxabxxx
xxaabbxx
xaaabbbx
aaaabbbb
ddddcccc
xdddcccx
xxddccxx
xxxdcxxx
"""

SQUARE_SHAPE = """
bbbbbbbb
bbbbbbbb
bbbbbbbb
bbbbbbbb
bbbbbbbb
bbbbbbbb
bbbbbbbb
bbbbbbbb
"""

ENERGY_SHAPE = """
xxxxxxxx
xxxxxxxx
xxxabxxx
xxaabbxx
xxddccxx
xxxdcxxx
xxxxxxxx
xxxxxxxx
"""

FOOD_SHAPE = """
xxxxxxxx
xxxxxxxx
xdddbbxx
ddbbbxxx
xxbddbbx
xdddbbxx
xxbbddbb
xxxxxxxx
"""


def graph_semantics(g):
  """Convert a networkx.DiGraph to compounds and reactions for grid_land."""
  compounds = {}
  reactions = {}
  for node, attributes in g.nodes.items():
    if attributes.get("reaction"):
      reactants = [e[0] for e in g.in_edges(node)]
      products = [e[1] for e in g.out_edges(node)]
      reactions[node] = create_reaction(reactants, products, attributes)
    if not attributes.get("reaction"):
      compounds[node] = create_compound(attributes)

  return compounds, reactions


def create_reaction(reactants, products, attributes):
  # TODO(b/192926758): support fixedSwapOrder = False, in that case, pass
  # reactants# and products as a dictionary mapping to the number required (not
  # a list with possibly repeated entries like the current version).
  return {
      "reactants": reactants,
      "products": products,
      "fixedSwapOrder": attributes.get("fixedSwapOrder", True),
      "priority": attributes.get("priority", 1),
  }


def create_compound(attributes):
  """Convert node attributes to dictionary structure needed for a compound."""
  data = {
      # Use black color if none provided.
      "color": attributes.get("color", (0, 0, 0, 0)),
      "properties": {
          # Use (0, 0) for structure if none provided,
          "structure": attributes.get("structure", (0, 0)),
      },
  }
  for k, v in attributes.items():
    data[k] = v
  return data


def add_system_nodes(g: nx.DiGraph):
  """Add several nodes that must always be present for the system to function.

  Args:
    g: (nx.DiGraph): directed graph representing the reaction system.
  """
  g.add_nodes_from([
      # Add a node for the "empty" compound.
      ("empty", {"color": EMPTY_COLOR,
                 "reactivity": "low"}),
      # Add a node for the "activated" compound.
      ("activated", {"color": WHITE_COLOR,
                     "immovable": True}),
      # Add unused nodes that serve only to make all standard groups valid so
      # their corresponding updater can be created.
      ("_unused_a", {"reactivity": "low"}),
      ("_unused_b", {"reactivity": "medium"}),
      ("_unused_c", {"reactivity": "high"})
  ])


def add_compounds_to_prefabs_dictionary(prefabs,
                                        compounds,
                                        reactivity_levels,
                                        sprites=False,
                                        default_reaction_radius=None,
                                        default_reaction_query_type=None,
                                        priority_mode=False):
  """Add compounds."""
  for compound_name in compounds.keys():
    prefabs[compound_name] = create_cell_prefab(
        compound_name,
        compounds,
        reactivity_levels,
        sprites=sprites,
        default_reaction_radius=default_reaction_radius,
        default_reaction_query_type=default_reaction_query_type,
        priority_mode=priority_mode)
  return prefabs


def multiply_tuple(color_tuple, factor):
  if len(color_tuple) == 3:
    return tuple([int(np.min([x * factor, 255])) for x in color_tuple])
  elif len(color_tuple) == 4:
    return tuple([int(np.min([x * factor])) for x in color_tuple])


def adjust_color_opacity(color_tuple, factor):
  apply_opacity = tuple([color_tuple[0], color_tuple[1], color_tuple[2],
                         color_tuple[3] * factor])
  return tuple([int(np.min([x])) for x in apply_opacity])


def get_matter_palette(sprite_color):
  return {
      "*": sprite_color,
      "b": shapes.WHITE,
      "x": shapes.ALPHA,
      # Shades for liquid matter.
      "L": shapes.adjust_color_brightness(sprite_color, 0.85),
      "l": shapes.adjust_color_brightness(sprite_color, 0.90),
      "w": shapes.adjust_color_brightness(sprite_color, 0.95),
  }


def get_cytoavatar_palette(sprite_color):
  return {
      "*": (184, 61, 187, 255),
      "&": (161, 53, 146, 255),
      "o": sprite_color,
      ",": shapes.BLACK,
      "x": shapes.ALPHA,
      "#": shapes.WHITE,
  }


def create_cell_prefab(compound_name, compounds, reactivity_levels,
                       sprites=False, default_reaction_radius=None,
                       default_reaction_query_type=None, priority_mode=False):
  """Create prefab for a cell object initially set to state=`compound_name`."""
  state_configs = []
  states_to_properties = {}
  sprite_colors = []
  query_configs = {}
  special_sprites = {}
  for compound, attributes in compounds.items():
    groups = []
    if "reactivity" in attributes:
      reactivity_group = attributes["reactivity"]
      groups.append(reactivity_group)
    if "immovable" in attributes and attributes["immovable"]:
      groups.append("immovables")
    if "query_config" in attributes:
      query_configs[compound] = attributes["query_config"]
    if "sprite" in attributes:
      special_sprites[compound] = attributes["sprite"]

    state_config = {
        "state": compound,
        "sprite": compound,
        "layer": "lowerPhysical",
        "groups": groups  + ["spawnPoints"],
    }
    state_configs.append(state_config)
    states_to_properties[compound] = attributes["properties"]
    sprite_colors.append(attributes["color"])

  # Configure the Reactant component.
  reactivities = {}
  for key, value in reactivity_levels.items():
    reactivities[key] = value

  if sprites:
    def get_palette(sprite_color):
      return {
          "x": EMPTY_COLOR[0:len(sprite_color)],
          "a": (252,) * len(sprite_color),
          "b": sprite_color,
          "c": multiply_tuple(sprite_color, 0.2),
          "d": sprite_color
      }
    appearance_kwargs = {
        "renderMode": "ascii_shape",
        "spriteNames": list(compounds.keys()),
        "spriteShapes": [DIAMOND_SHAPE] * len(sprite_colors),
        "palettes": [get_palette(color) for color in sprite_colors],
        "noRotates": [True] * len(sprite_colors),
    }
    # Must ensure "empty" and "activated" are not given the diamond sprite.
    for i, compound in enumerate(appearance_kwargs["spriteNames"]):
      if compound in ["empty", "activated"]:
        appearance_kwargs["spriteShapes"][i] = SQUARE_SHAPE
      if compound in special_sprites:
        appearance_kwargs["spriteShapes"][i] = special_sprites[compound]
  else:
    appearance_kwargs = {
        "spriteNames": list(compounds.keys()),
        "spriteRGBColors": sprite_colors,
    }

  prefab = {
      "name": "cell",
      "components": [
          {
              "component": "StateManager",
              "kwargs": {
                  "initialState": compound_name,
                  "stateConfigs": state_configs,
              }
          },
          {
              "component": "Transform",
          },
          {
              "component": "Appearance",
              "kwargs": appearance_kwargs
          },
          {
              "component": "Cell",
              "kwargs": {
                  "numCellStates": len(state_configs),
                  "statesToProperties": states_to_properties,
                  # The radius over which to search for neighbors on every step.
                  "radius": default_reaction_radius,
                  # Query according to L1 (diamond) or L2 (disc) norm.
                  "queryType": default_reaction_query_type,
                  # Layers on which to search for neighbors on every step.
                  "interactionLayers": ["lowerPhysical", "overlay"],
                  # You can override query properties on a per state basis.
                  "stateSpecificQueryConfig": query_configs,
              },
          },
          {
              "component": "Reactant",
              "kwargs": {
                  "name": "Reactant",
                  "reactivities": reactivities,
                  "priorityMode": priority_mode,
              }
          },
          {
              "component": "Product",
              "kwargs": {
                  "name": "Product",
              }
          },
      ]
  }
  return prefab


def create_vesicle(player_idx: int,
                   compounds,
                   reactivity_levels,
                   default_reaction_radius=None,
                   default_reaction_query_type=None,
                   priority_mode=False):
  """Construct prefab for an avatar's vesicle object."""
  # Lua is 1-indexed.
  lua_index = player_idx + 1

  vesicle_prefix = "vesicle_"
  state_configs = []
  states_to_properties = {}
  sprite_colors = []
  sprite_shapes = []
  query_configs = {}
  for compound, attributes in compounds.items():
    groups = []
    sprite_shape = shapes.SINGLE_HOLDING_LIQUID
    if "reactivity" in attributes:
      reactivity_group = (vesicle_prefix +
                          attributes["reactivity"])
      groups.append(reactivity_group)
    if "immovable" in attributes and attributes["immovable"]:
      groups.append("immovables")
    if "query_config" in attributes:
      query_configs[compound] = attributes["query_config"]

    sprite_color = attributes["color"]
    if compound == "empty":
      sprite_shape = shapes.SQUARE
      sprite_color = shapes.ALPHA
    state_config = {
        "state": compound,
        "sprite": compound + "_vesicle",
        "layer": "overlay",
        "groups": groups,
    }
    state_configs.append(state_config)
    states_to_properties[compound] = attributes["properties"]
    sprite_colors.append(sprite_color)
    sprite_shapes.append(sprite_shape)

  # Configure the Reactant component.
  reactivities = {}
  for key, value in reactivity_levels.items():
    reactivities[vesicle_prefix + key] = value

  prefab = {
      "name": f"avatar_vesicle_{lua_index}",
      "components": [
          {
              "component": "StateManager",
              "kwargs": {
                  "initialState": "preInit",
                  "stateConfigs": state_configs +
                                  [{"state": "preInit"}],
              }
          },
          {
              "component": "Transform",
          },
          {
              "component": "Appearance",
              "kwargs": {
                  "renderMode": "ascii_shape",
                  "spriteNames": [key + "_vesicle" for key in compounds.keys()],
                  "spriteShapes": sprite_shapes,
                  "palettes": [get_matter_palette(sprite_colors[i])
                               for i in range(len(sprite_colors))],
                  "noRotates": [True] * len(sprite_colors)
              },
          },
          {
              "component": "AvatarVesicle",
              "kwargs": {
                  "playerIndex": lua_index,
                  "preInitState": "preInit",
                  "initialState": "empty",
                  "waitState": "vesicleWait"
              }
          },
          {
              "component": "Cell",
              "kwargs": {
                  "numCellStates": len(state_configs),
                  "statesToProperties": states_to_properties,
                  # The radius over which to search for neighbors on every step.
                  "radius": default_reaction_radius,
                  # Query according to L1 (diamond) or L2 (disc) norm.
                  "queryType": default_reaction_query_type,
                  # Layers on which to search for neighbors on every step.
                  "interactionLayers": ["lowerPhysical", "overlay"],
                  # You can override query properties on a per state basis.
                  "stateSpecificQueryConfig": query_configs,
              },
          },
          {
              "component": "Reactant",
              "kwargs": {
                  "name": "Reactant",
                  "reactivities": reactivities,
                  "priorityMode": priority_mode,
              }
          },
          {
              "component": "Product",
              "kwargs": {
                  "name": "Product",
              }
          },
      ]
  }
  return prefab


def create_avatar_constant_self_view(
    rewarding_reactions,
    player_idx: int,
    target_sprite_self_empty: Dict[str, Any],
    target_sprite_self_holds_one: Dict[str, Any],
    randomize_initial_orientation: bool = True,
    add_location_observer: bool = False) -> Dict[str, Any]:
  """Create an avatar prefab rewarded by reactions in `rewarding_reactions`."""
  # Lua is 1-indexed.
  lua_index = player_idx + 1

  # Setup the self vs other sprite mapping.
  source_sprite_self_empty = f"Avatar_{lua_index}_empty"
  source_sprite_self_holds_one = f"Avatar_{lua_index}_holds_one"

  custom_sprite_map = {
      source_sprite_self_empty: target_sprite_self_empty["name"],
      source_sprite_self_holds_one: target_sprite_self_holds_one["name"],
  }

  # Part of the avatar is partially transparent so molecules can be seen below.
  cytoavatar_palette = get_cytoavatar_palette((0, 0, 0, 75))

  live_state_name_empty = f"player{lua_index}_empty"
  live_state_name_holds_one = f"player{lua_index}_holds_one"
  avatar_object = {
      "name": "avatar",
      "components": [
          {
              "component": "StateManager",
              "kwargs": {
                  "initialState": live_state_name_empty,
                  "stateConfigs": [
                      {"state": live_state_name_empty,
                       "layer": "upperPhysical",
                       "sprite": source_sprite_self_empty,
                       "contact": "avatar",
                       "groups": ["players"]},
                      {"state": live_state_name_holds_one,
                       "layer": "upperPhysical",
                       "sprite": source_sprite_self_holds_one,
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
                  "renderMode": "ascii_shape",
                  "spriteNames": [source_sprite_self_empty,
                                  source_sprite_self_holds_one],
                  "spriteShapes": [shapes.CYTOAVATAR_EMPTY,
                                   shapes.CYTOAVATAR_HOLDING_ONE],
                  "palettes": [cytoavatar_palette] * 2,
                  "noRotates": [True] * 2
              }
          },
          {
              "component": "AdditionalSprites",
              "kwargs": {
                  "renderMode": "ascii_shape",
                  "customSpriteNames": [
                      target_sprite_self_empty["name"],
                      target_sprite_self_holds_one["name"],
                  ],
                  "customSpriteShapes": [
                      target_sprite_self_empty["shape"],
                      target_sprite_self_holds_one["shape"],
                  ],
                  "customPalettes": [
                      cytoavatar_palette,
                      cytoavatar_palette,
                  ],
                  "customNoRotates": [
                      target_sprite_self_empty["noRotate"],
                      target_sprite_self_holds_one["noRotate"],
                  ],
              }
          },
          {
              "component": "Avatar",
              "kwargs": {
                  "index": lua_index,
                  "spawnGroup": "spawnPoints",
                  "aliveState": live_state_name_empty,
                  "additionalLiveStates": [live_state_name_holds_one],
                  "waitState": "playerWait",
                  "actionOrder": ["move", "turn", "ioAction"],
                  "actionSpec": {
                      "move": {"default": 0, "min": 0, "max": 4},
                      "turn": {"default": 0, "min": -1, "max": 1},
                      "ioAction": {"default": 0, "min": 0, "max": 1},
                  },
                  "view": {
                      "left": 5,
                      "right": 5,
                      "forward": 9,
                      "backward": 1,
                      "centered": False
                  },
                  "spriteMap": custom_sprite_map,
                  "randomizeInitialOrientation": randomize_initial_orientation,
              }
          },
          {
              "component": "IOBeam",
              "kwargs": {
                  "cooldownTime": 2,
              }
          },
          {
              "component": "VesicleManager",
              "kwargs": {
                  "orderedVesicles": ["vesicleOne",],
                  "cytoavatarStates": {
                      "empty": live_state_name_empty,
                      "holdingOne": live_state_name_holds_one,
                  },
              }
          },
          {
              "component": "ReactionsToRewards",
              "kwargs": {
                  # Specify rewards for specific reactions.
                  "rewardingReactions": rewarding_reactions
              }
          },
      ]
  }
  if add_location_observer:
    avatar_object["components"].append({
        "component": "LocationObserver",
        "kwargs": {"objectIsAvatar": True, "alsoReportOrientation": True},
    })

  return avatar_object


def create_scene(reactions, stochastic_episode_ending=False):
  """Construct the global scene prefab."""
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
              "component": "ReactionAlgebra",
              "kwargs": {
                  "reactions": reactions
              }
          },
          {
              "component": "GlobalMetricTracker",
              "kwargs": {
                  "name": "GlobalMetricTracker",
              }
          },
      ]
  }
  if stochastic_episode_ending:
    scene["components"].append({
        "component": "StochasticIntervalEpisodeEnding",
        "kwargs": {
            "minimumFramesPerEpisode": 1000,
            "intervalLength": 100,  # Set equal to unroll length.
            "probabilityTerminationPerInterval": 0.2
        }
    })
  return scene
