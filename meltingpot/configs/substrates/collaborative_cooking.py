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
"""Configuration for Collaborative Cooking.

A pure common interest cooking game inspired by Carroll et al. (2019) and
Strouse et al. (2021).

Carroll, M., Shah, R., Ho, M.K., Griffiths, T.L., Seshia, S.A., Abbeel, P. and
Dragan, A., 2019. On the utility of learning about humans for human-AI
coordination. arXiv preprint arXiv:1910.05789.

Strouse, D.J., McKee, K.R., Botvinick, M., Hughes, E. and Everett, R., 2021.
Collaborating with Humans without Human Data. arXiv preprint arXiv:2110.08176.
"""

import copy
from typing import Any, Dict, Mapping, Sequence

from meltingpot.utils.substrates import colors
from meltingpot.utils.substrates import game_object_utils
from meltingpot.utils.substrates import shapes
from meltingpot.utils.substrates import specs
from ml_collections import config_dict as configdict

# Warning: setting `_ENABLE_DEBUG_OBSERVATIONS = True` may cause slowdown.
_ENABLE_DEBUG_OBSERVATIONS = False

COOKING_TIME = 20
items = ["empty", "tomato", "dish", "soup"]

# Map a character to the prefab it represents in the ASCII map.
CHAR_PREFAB_MAP = {
    "P": "spawn_point",
    "#": "counter",
    "O": "tomato_dispenser",
    "D": "dish_dispenser",
    "T": "delivery_location",
    "C": "cooking_pot",
}

###########
# SPRITES #
###########

TARGET_SPRITE_SELF = {
    "name": "Self",
    "shape": shapes.CUTE_AVATAR,
    "palette": shapes.get_palette((50, 100, 200)),
    "noRotate": True,
}

BACKGROUND_LIGHT = (255, 255, 255, 255)
BACKGROUND_DARK = (82, 82, 82, 255)
OUTLINE = (85, 58, 23, 255)
OUTLINE_DARK = (49, 49, 49, 255)
INVISIBLE = (0, 0, 0, 0)
COUNTER = (115, 81, 39, 255)

SPRITES = {}
PALETTES = {}

SPRITES["interact"] = """
PPPPPPPPPPPPPPPP
PPPPPPPPPPPPPPPP
PP~~~~~~~~~~~~PP
PP~~~~~~~~~~~~PP
PP~~~~~~~~~~~~PP
PP~~~~~~~~~~~~PP
PP~~~~~~~~~~~~PP
PP~~~~~~~~~~~~PP
PP~~~~~~~~~~~~PP
PP~~~~~~~~~~~~PP
PP~~~~~~~~~~~~PP
PP~~~~~~~~~~~~PP
PP~~~~~~~~~~~~PP
PP~~~~~~~~~~~~PP
PPPPPPPPPPPPPPPP
PPPPPPPPPPPPPPPP
"""

SPRITES["empty"] = """
~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~
"""
PALETTES["empty"] = {"~": INVISIBLE, "@": BACKGROUND_LIGHT}

SPRITES["counter"] = """
&&&&&&&&&&&&&&&&
&**************&
&**************&
&**************&
&**************&
&**************&
&**************&
&**************&
&**************&
&**************&
&**************&
&**************&
&**************&
&**************&
&**************&
&&&&&&&&&&&&&&&&
"""
PALETTES["counter"] = {"*": COUNTER, "&": OUTLINE}

SPRITES["delivery_location"] = SPRITES["counter"]
PALETTES["delivery_location"] = {"*": BACKGROUND_DARK, "&": OUTLINE_DARK}

SPRITES["dish"] = """
~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~
~~~~~~++++~~~~~~
~~~~~+^^^^+~~~~~
~~~~~+^^^^+~~~~~
~~~~~+^^^^+~~~~~
~~~~~&++++&~~~~~
~~~~~~&&&&~~~~~~
~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~
"""
PALETTES["dish"] = {
    "~": [0, 0, 0, 0],
    "+": [255, 255, 255, 255],
    "^": [233, 239, 248, 255],
    "&": [221, 222, 238, 255]
}

SPRITES["soup"] = SPRITES["dish"]
PALETTES["soup"] = {
    "~": [0, 0, 0, 0],
    "+": [255, 255, 255, 255],
    "^": [236, 58, 74, 255],
    "&": [221, 222, 238, 255]
}

SPRITES["dish_dispenser"] = """
&&&&&&&&&&&&&&&&
&~~~~~~~~~~~~~~&
&~~~~~~~~~~~~~~&
&~~~~~~~~~~~~~~&
&~~~~~~~~~~~~~~&
&~~~~~++++~~~~~&
&~~~~+^^^^+~~~~&
&~~~~+^^^^+~~~~&
&~~~~+^^^^+~~~~&
&~~~~X++++X~~~~&
&~~~~~XXXX~~~~~&
&~~~~~~~~~~~~~~&
&~~~~~~~~~~~~~~&
&~~~~~~~~~~~~~~&
&~~~~~~~~~~~~~~&
&&&&&&&&&&&&&&&&
"""

PALETTES["dish_dispenser"] = {
    "&": OUTLINE_DARK,
    "~": BACKGROUND_DARK,
    "+": [255, 255, 255, 255],
    "^": [233, 239, 248, 255],
    "X": [221, 222, 238, 255]
}

SPRITES["tomato"] = """
~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~
~~~~~~++^+~~~~~~
~~~~~&O^---~~~~~
~~~~~O-----~~~~~
~~~~~O&-@--~~~~~
~~~~~~OO&&~~~~~~
~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~
"""
PALETTES["tomato"] = {
    "~": [0, 0, 0, 0],
    "+": [239, 81, 90, 255],
    "^": [29, 139, 43, 255],
    "&": [190, 53, 62, 255],
    "O": [151, 47, 52, 255],
    "-": [236, 58, 74, 255],
    "@": [240, 57, 75, 255]
}

SPRITES["tomato_dispenser"] = """
&&&&&&&&&&&&&&&&
&,,,,,,,,,,,,,,&
&,,,,,,,,,,,,,,&
&,,,,,,,,,,,,,,&
&,,,,,,,,,,,,,,&
&,,,,,,,,,,,,,,&
&,,,,,++^+,,,,,&
&,,,,XO^---,,,,&
&,,,,O-----,,,,&
&,,,,O&-@--,,,,&
&,,,,,OOXX,,,,,&
&,,,,,,,,,,,,,,&
&,,,,,,,,,,,,,,&
&,,,,,,,,,,,,,,&
&,,,,,,,,,,,,,,&
&&&&&&&&&&&&&&&&
"""

PALETTES["tomato_dispenser"] = {
    ",": BACKGROUND_DARK,
    "&": OUTLINE_DARK,
    "~": [0, 0, 0, 0],
    "+": [239, 81, 90, 255],
    "^": [29, 139, 43, 255],
    "X": [190, 53, 62, 255],
    "O": [151, 47, 52, 255],
    "-": [236, 58, 74, 255],
    "@": [240, 57, 75, 255]
}

SPRITES["cooking_pot_empty"] = """
&&&&&&&&&&&&&&&&
&~~~++++++++~~~&
&~~+^^^^^^^XO~~&
&~~+^^^^^^XXO~~&
&^^+^^^^^XXXO--&
&^~+^^^^XXXXO~-&
&^~+@@@@AAAAO~-&
&^^+@@@@AAAAO--&
&~~+@@@@AAAAO~~&
&~~@OOOOOOOO-~~&
&~~@MMMMMMMM-~~&
&~~@MMMMMMMM-~~&
&~~@MMMMMMMM-~~&
&~~@MMMMMMMM-~~&
&~~~MMMMMMMM~~~&
&&&&&&&&&&&&&&&&
"""

SPRITES["cooking_pot_1"] = """
&&&&&&&&&&&&&&&&
&~~~++++++++~~~&
&~~+^^^^^^^XO~~&
&~~+^^^^^^XXO~~&
&^^+^^^^^XXXO--&
&^~+KKKKLLLLO~-&
&^~+KKKKLLLLO~-&
&^^+KKKKLLLLO--&
&~~+KKKKLLLLO~~&
&~~@OOOOOOOO-~~&
&~~@MMMMMMMM-~~&
&~~@MMMMMMMM-~~&
&~~@MMMMMMMM-~~&
&~~@MMMMMMMM-~~&
&~~~MMMMMMMM~~~&
&&&&&&&&&&&&&&&&
"""

SPRITES["cooking_pot_2"] = """
&&&&&&&&&&&&&&&&
&~~~++++++++~~~&
&~~+^^^^^^^XO~~&
&~~+^^^^^^XXO~~&
&^^+KKKKKKLLO--&
&^~+KKKKLKLLO~-&
&^~+KLKKKKLLO~-&
&^^+KKKKKKLLO--&
&~~+KKKKKKLLO~~&
&~~@OOOOOOOO-~~&
&~~@MMMMMMMM-~~&
&~~@MMMMMMMM-~~&
&~~@MMMMMMMM-~~&
&~~@MMMMMMMM-~~&
&~~~MMMMMMMM~~~&
&&&&&&&&&&&&&&&&
"""

SPRITES["cooking_pot_3"] = """
&&&&&&&&&&&&&&&&
&~~~++++++++~~~&
&~~+KKKKKKKKO~~&
&~~+KNKKKKKKO~~&
&^^+KKKKKKKKO--&
&^~+KKKKKKKKO~-&
&^~+KKKKKKKKO~-&
&^^+KKNKKKNKO--&
&~~+KKKKKKKKO~~&
&~~@OOOOOOOO-~~&
&~~@MMMMMMMM-~~&
&~~@MMMMMMMM-~~&
&~~@MMMMMMMM-~~&
&~~@MMMMMMMM-~~&
&~~~MMMMMMMM~~~&
&&&&&&&&&&&&&&&&
"""

PALETTES["cooking_pot"] = {
    "&": OUTLINE,
    "~": COUNTER,
    "+": [224, 231, 240, 255],
    "^": [140, 155, 181, 255],
    "X": [98, 95, 128, 255],
    "O": [238, 241, 241, 255],
    "-": [194, 206, 222, 255],
    "@": [92, 106, 135, 255],
    "A": [65, 66, 97, 255],
    "M": [139, 155, 181, 255],
    "K": [236, 58, 74, 255],
    "L": [161, 43, 43, 255],
    "N": [242, 226, 187, 255]
}

SPRITES["loading_bar"] = """
~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~
~~~abcdefghij~~~
~~~abcdefghij~~~
~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~
"""


def create_loading_bar_palette(count, finished=False):
  """Creates an incrementally colored loading bar based on count."""
  characters = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]
  bars_palette = {"~": INVISIBLE}
  for idx in range(0, 10):
    if idx < count:
      if finished:
        bars_palette[characters[idx]] = (15, 188, 15, 255)
      else:
        bars_palette[characters[idx]] = (201, 178, 50, 255)
    else:
      bars_palette[characters[idx]] = (255, 255, 255, 255)

  return bars_palette


OFFSET_SIZE = 3
for s in ["empty", "tomato", "dish", "soup"]:
  SPRITES[s + "_offset"] = ("~~~~~~~~~~~~~~~~\n" * OFFSET_SIZE +
                            SPRITES[s][1:-OFFSET_SIZE*17])

###########
# PREFABS #
###########

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

# pylint: disable=g-complex-comprehension
INVENTORY = {
    "name": "inventory",
    "components": [
        {
            "component": "StateManager",
            "kwargs": {
                "initialState":
                    "wait",
                "stateConfigs": [{
                    "state": "wait"
                }] + [{
                    "state": item,
                    "sprite": item,
                    "layer": "overlay"
                } for item in items] + [{
                    "state": item + "_offset",
                    "sprite": item + "_offset",
                    "layer": "overlay"
                } for item in items]
            }
        },
        {
            "component": "Transform",
            "kwargs": {}
        },
        {
            "component": "Appearance",
            "kwargs": {
                "renderMode": "ascii_shape",
                "spriteNames": items + [item + "_offset" for item in items],
                "spriteShapes": ([SPRITES[item] for item in items] +
                                 [SPRITES[item + "_offset"] for item in items]),
                "palettes": ([PALETTES[item] for item in items] +
                             [PALETTES[item] for item in items]),
                "noRotates": [False]
            }
        },
        {
            "component": "Inventory",
            "kwargs": {
                "playerIndex": -1,  # Can be overwritten.
                "emptyState": "empty",
                "waitState": "wait"
            }
        },
    ]
}
# pylint: enable=g-complex-comprehension

loading_palettes = ([create_loading_bar_palette(idx) for idx in range(0, 10)] +
                    [create_loading_bar_palette(10, finished=True)])
LOADING_BAR = {
    "name": "loading_bar",
    "components": [{
        "component": "StateManager",
        "kwargs": {
            "initialState":
                "loading_bar_0",
            "stateConfigs": [{
                "state": "loading_bar_%d" % d, "layer": "overlay",
                "sprite": "loading_bar_%d" % d} for d in range(0, 11)],
        }
    }, {
        "component": "Transform",
        "kwargs": {}
    }, {
        "component": "Appearance",
        "kwargs": {
            "renderMode": "ascii_shape",
            "spriteNames": ["loading_bar_%d" % d for d in range(0, 11)],
            "spriteShapes": [SPRITES["loading_bar"] for _ in range(0, 11)],
            "palettes": loading_palettes,
            "noRotates": [True]
        }
    }, {
        "component": "LoadingBarVisualiser",
        "kwargs": {
            "totalTime": COOKING_TIME,
            "customStateNames": ["loading_bar_%d" % d for d in range(0, 11)],
        }
    }]
}


def create_base_prefab(name, layer="upperPhysical"):
  """Returns a base prefab with a given name on the given layer."""
  return {
      "name": f"{name}",
      "components": [
          {
              "component": "StateManager",
              "kwargs": {
                  "initialState": f"{name}",
                  "stateConfigs": [{
                      "state": f"{name}",
                      "layer": layer,
                      "sprite": f"{name}",
                  }],
              }
          },
          {
              "component": "Transform",
              "kwargs": {}
          },
          {
              "component": "Appearance",
              "kwargs": {
                  "renderMode": "ascii_shape",
                  "spriteNames": [name],
                  "spriteShapes": [SPRITES[name]],
                  "palettes": [PALETTES[name]],
                  "noRotates": [True]
              }
          }]
  }


def create_counter():
  """Returns a prefab which can contain one of any item."""
  base_prefab = create_base_prefab("counter")
  base_prefab["components"] += [{
      "component": "Container",
      "kwargs": {
          "reward": 0.0
      }
  }]
  return base_prefab


def create_dispenser(prefab_name, item_name):
  """Returns a prefab which dispenses items to avatars upon interaction."""
  base_prefab = create_base_prefab(prefab_name)
  base_prefab["components"] += [{
      "component": "Container",
      "kwargs": {
          "startingItem": item_name,
          "infinite": True,
          "reward": 0.0
      }
  }]
  return base_prefab


def create_receiver(prefab_name,
                    item_name,
                    reward=0,
                    global_reward=False):
  """Returns a prefab which can receive items from avatars.

  Args:
    prefab_name: the name of the prefab.
    item_name: the name of the accepted item.
    reward: value of reward given to avatar when object receives the item.
    global_reward: if true, reward all avatars.

  Returns:
    A prefab which can receive items.
  """
  base_prefab = create_base_prefab(prefab_name)
  base_prefab["components"] += [{
      "component": "Receiver",
      "kwargs": {
          "acceptedItems": item_name,
          "reward": reward,
          "globalReward": global_reward
      }
  }]
  return base_prefab


def create_cooking_pot(time_to_cook, reward=1):
  """Creates a cooking pot for tomatoes."""

  state_configs = []
  sprite_names = []
  pot_sprites = []
  cooking_pot_palettes = []
  custom_state_names = []
  available_foods = ["tomato"]

  # Create state for each combination of available foods.
  foods_in_pot = ["empty"] + available_foods
  for food1 in foods_in_pot:
    for food2 in foods_in_pot:
      for food3 in foods_in_pot:
        sprite_name = "CookingPot_%s_%s_%s" % (food1, food2, food3)
        name = "cooking_pot_%s_%s_%s" % (food1, food2, food3)
        entry = {"state": name,
                 "layer": "upperPhysical",
                 "sprite": sprite_name,
                 "groups": ["cooking_pot"]}
        state_configs.append(entry)
        sprite_names.append(sprite_name)
        pots_palette = PALETTES["cooking_pot"]
        cooking_pot_palettes.append(pots_palette)
        custom_state_names.append(name)

        if food1 == "empty":
          pot_sprites.append(SPRITES["cooking_pot_empty"])
        elif food2 == "empty":
          pot_sprites.append(SPRITES["cooking_pot_1"])
        elif food3 == "empty":
          pot_sprites.append(SPRITES["cooking_pot_2"])
        else:
          pot_sprites.append(SPRITES["cooking_pot_3"])

  # Create cooked state.
  sprite_name = "CookingPot_cooked"
  name = "cooking_pot_cooked"
  entry = {"state": "cooking_pot_cooked",
           "layer": "upperPhysical",
           "sprite": sprite_name,
           "groups": ["cooking_pot"]}
  state_configs.append(entry)
  sprite_names.append(sprite_name)
  pot_sprites.append(SPRITES["cooking_pot_3"])
  pots_palette = PALETTES["cooking_pot"]
  cooking_pot_palettes.append(pots_palette)
  custom_state_names.append(name)

  cooking_pot = {
      "name": "cooking_pot",
      "components": [
          {
              "component": "StateManager",
              "kwargs": {
                  "initialState": (
                      "cooking_pot_empty_empty_empty"
                  ),
                  "stateConfigs": state_configs
              }
          },
          {
              "component": "Transform",
              "kwargs": {}
          },
          {
              "component": "Appearance",
              "kwargs": {
                  "renderMode": "ascii_shape",
                  "spriteNames": sprite_names,
                  "spriteShapes": pot_sprites,
                  "palettes": cooking_pot_palettes,
                  "noRotates": [True for _ in sprite_names]
              }
          },
          {
              "component": "CookingPot",
              "kwargs": {
                  "acceptedItems": available_foods,
                  "cookingTime": time_to_cook,
                  "reward": reward,
                  "customStateNames": custom_state_names
              }
          },
      ]
  }

  return cooking_pot


def create_prefabs(cooking_pot_pseudoreward: float = 0.0):
  """Creates a dictionary mapping names to template game objects."""
  prefabs = {
      "spawn_point": SPAWN_POINT,
      "inventory": INVENTORY,
      "loading_bar": LOADING_BAR,
      "counter": create_counter(),
      "dish_dispenser": create_dispenser(prefab_name="dish_dispenser",
                                         item_name="dish"),
      "tomato_dispenser": create_dispenser(prefab_name="tomato_dispenser",
                                           item_name="tomato"),
      "delivery_location": create_receiver(prefab_name="delivery_location",
                                           item_name="soup",
                                           reward=20,
                                           global_reward=True),
      "cooking_pot": create_cooking_pot(time_to_cook=COOKING_TIME,
                                        reward=cooking_pot_pseudoreward),
  }
  return prefabs

###########
# ACTIONS #
###########

# Primitive action components.
# pylint: disable=bad-whitespace
# pyformat: disable
NOOP       = {"move": 0, "turn":  0, "interact": 0}
FORWARD    = {"move": 1, "turn":  0, "interact": 0}
STEP_RIGHT = {"move": 2, "turn":  0, "interact": 0}
BACKWARD   = {"move": 3, "turn":  0, "interact": 0}
STEP_LEFT  = {"move": 4, "turn":  0, "interact": 0}
TURN_LEFT  = {"move": 0, "turn": -1, "interact": 0}
TURN_RIGHT = {"move": 0, "turn":  1, "interact": 0}
INTERACT   = {"move": 0, "turn":  0, "interact": 1}
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
    INTERACT,
)

###########
# OBJECTS #
###########


def create_game_objects(ascii_map_string, prefabs):
  """Returns list of game objects from 'ascii_map' and 'char_prefab' mapping."""

  # Create all game objects.
  game_objects = []
  for char, _ in CHAR_PREFAB_MAP.items():
    transforms = game_object_utils.get_game_object_positions_from_map(
        ascii_map_string, char)
    for transform in transforms:
      # Add inventory game object for holding and visualising items.
      if char == "#" or char == "O" or char == "D":
        inventory_object = copy.deepcopy(prefabs["inventory"])
        go_transform = game_object_utils.get_first_named_component(
            inventory_object, "Transform")
        go_transform["kwargs"]["position"] = (transform.position.x,
                                              transform.position.y)
        game_objects.append(inventory_object)

      # Add loading bar object to cooking pots.
      if char == "C":
        loading_object = copy.deepcopy(prefabs["loading_bar"])
        go_transform = game_object_utils.get_first_named_component(
            loading_object, "Transform")
        go_transform["kwargs"]["position"] = (transform.position.x,
                                              transform.position.y)
        game_objects.append(loading_object)

  return game_objects


def create_avatar_object(player_idx: int,
                         target_sprite_self: Dict[str, Any]) -> Dict[str, Any]:
  """Create an avatar object that always sees itself as blue."""
  # Lua is 1-indexed.
  lua_index = player_idx + 1

  # Setup the self vs other sprite mapping.
  source_sprite_self = "Avatar" + str(lua_index)
  custom_sprite_map = {source_sprite_self: target_sprite_self["name"]}

  interact_palette = {
      "P": colors.palette[player_idx] + (255,),
      "~": INVISIBLE,
  }

  live_state_name = "player{}".format(lua_index)
  avatar_object = {
      "name": "avatar",
      "components": [
          {
              "component": "StateManager",
              "kwargs": {
                  "initialState": live_state_name,
                  "stateConfigs": [
                      {"state": live_state_name,
                       "layer": "upperPhysical",
                       "sprite": source_sprite_self,
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
                  "spriteNames": [source_sprite_self],
                  "spriteShapes": [shapes.CUTE_AVATAR],
                  "palettes": [shapes.get_palette(colors.palette[player_idx])],
                  "noRotates": [True]
              }
          },
          {
              "component": "AdditionalSprites",
              "kwargs": {
                  "renderMode": "ascii_shape",
                  "customSpriteNames": [target_sprite_self["name"]],
                  "customSpriteShapes": [target_sprite_self["shape"]],
                  "customPalettes": [target_sprite_self["palette"]],
                  "customNoRotates": [target_sprite_self["noRotate"]],
              }
          },
          {
              "component": "Avatar",
              "kwargs": {
                  "index": lua_index,
                  "spawnGroup": "spawnPoints",
                  "aliveState": live_state_name,
                  "waitState": "playerWait",
                  "speed": 1.0,
                  "actionOrder": ["move", "turn", "interact"],
                  "actionSpec": {
                      "move": {"default": 0, "min": 0, "max": 4},
                      "turn": {"default": 0, "min": -1, "max": 1},
                      "interact": {"default": 0, "min": 0, "max": 1},
                  },
                  "view": {
                      "left": 2,
                      "right": 2,
                      "forward": 3,
                      "backward": 1,
                      "centered": False
                  },
                  "spriteMap": custom_sprite_map,
              }
          },
          {
              "component": "InteractBeam",
              "kwargs": {
                  "cooldownTime": 1,
                  "shapes": [SPRITES["interact"]],
                  "palettes": [interact_palette]
              }
          },
          {"component": "AvatarCumulants",},
      ]
  }
  if _ENABLE_DEBUG_OBSERVATIONS:
    avatar_object["components"].append({
        "component": "LocationObserver",
        "kwargs": {"objectIsAvatar": True, "alsoReportOrientation": True},
    })
    avatar_object["components"].append({
        "component": "AvatarMetricReporter",
        "kwargs": {
            "metrics": [
                {
                    "name": "ADDED_INGREDIENT_TO_COOKING_POT",
                    "type": "Doubles",
                    "shape": [],
                    "component": "AvatarCumulants",
                    "variable": "addedIngredientToCookingPot",
                },
                {
                    "name": "COLLECTED_SOUP_FROM_COOKING_POT",
                    "type": "Doubles",
                    "shape": [],
                    "component": "AvatarCumulants",
                    "variable": "collectedSoupFromCookingPot",
                },
            ]
        }
    })

  return avatar_object


def create_avatar_objects(num_players, prefabs):
  """Returns list of avatar objects of length 'num_players'."""
  game_objects = []
  for player_idx in range(0, num_players):
    lua_index = player_idx + 1
    game_object = create_avatar_object(player_idx,
                                       TARGET_SPRITE_SELF)
    game_objects.append(game_object)

    # Add inventory game object which will be connected to player at init.
    inventory_object = copy.deepcopy(prefabs["inventory"])
    game_object_utils.get_first_named_component(
        inventory_object,
        "Inventory")["kwargs"]["playerIndex"] = lua_index
    game_objects.append(inventory_object)

  return game_objects


def get_config():
  """Default configuration for training on the collaborative cooking level."""
  config = configdict.ConfigDict()

  # Cooking pot pseudoreward should be 0.0 for the canonical version of this
  # environment, but in order to train background bots it is sometimes useful
  # to give them a pseudoreward when they put items in the cooking pot. It has
  # the effect of shaping their behavior a bit in the right direction.
  config.cooking_pot_pseudoreward = 0.0

  # Action set configuration.
  config.action_set = ACTION_SET
  # Observation format configuration.
  config.individual_observation_names = [
      "RGB",
  ]
  config.global_observation_names = [
      "WORLD.RGB",
  ]

  config.action_spec = specs.action(len(ACTION_SET))

  return config


def build(
    roles: Sequence[str],
    config: configdict.ConfigDict,
) -> Mapping[str, Any]:
  """Build the substrate given player roles."""
  num_players = len(roles)
  ascii_map = config.layout.ascii_map
  prefabs = create_prefabs(
      cooking_pot_pseudoreward=config.cooking_pot_pseudoreward)
  game_objects = create_game_objects(ascii_map, prefabs)
  extra_game_objects = create_avatar_objects(num_players, prefabs)
  game_objects += extra_game_objects
  # Build the rest of the substrate definition.
  substrate_definition = dict(
      levelName="collaborative_cooking",
      levelDirectory="meltingpot/lua/levels",
      numPlayers=num_players,
      maxEpisodeLengthFrames=1000,
      spriteSize=8,
      topology="BOUNDED",  # Choose from ["BOUNDED", "TORUS"],
      simulation={
          "map": ascii_map,
          "gameObjects": game_objects,
          "prefabs": prefabs,
          "charPrefabMap": CHAR_PREFAB_MAP,
      },
  )
  return substrate_definition
