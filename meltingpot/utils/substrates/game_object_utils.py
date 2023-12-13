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
"""Utilities for DMLab2D Game Objects."""

import copy
import enum
from typing import List, Mapping, NamedTuple, Optional, Sequence, Tuple, Union
from meltingpot.utils.substrates import colors
from meltingpot.utils.substrates import shapes
import numpy as np

# Type of a GameObject prefab configuration: A recursive string mapping.
# pytype: disable=not-supported-yet
PrefabConfig = Mapping[str, "PrefabConfigValue"]
PrefabConfigValue = Union[str, float, List["PrefabConfigValue"], PrefabConfig]
# pytype: enable=not-supported-yet


class Position(NamedTuple):
  x: int
  y: int


class Orientation(enum.Enum):
  NORTH = "N"
  EAST = "E"
  SOUTH = "S"
  WEST = "W"


class Transform(NamedTuple):
  position: Position
  orientation: Orientation


# Special char to prefab mappings
TYPE_ALL = "all"
TYPE_CHOICE = "choice"


def get_named_components(
    game_object_config: PrefabConfig,
    name: str):
  return [component for component in game_object_config["components"]
          if component["component"] == name]


def get_first_named_component(
    game_object_config: PrefabConfig,
    name: str):
  named = get_named_components(game_object_config, name)
  if not named:
    raise ValueError(f"No component with name '{name}' found.")
  return named[0]


def build_game_objects(
    num_players: int,
    ascii_map: str,
    prefabs: Optional[Mapping[str, PrefabConfig]] = None,
    char_prefab_map: Optional[PrefabConfig] = None,
    player_palettes: Optional[Sequence[shapes.Color]] = None,
    use_badges: bool = False,
    badge_palettes: Optional[Sequence[shapes.Color]] = None,
) -> Tuple[List[PrefabConfig], List[PrefabConfig]]:
  """Build all avatar and normal game objects based on the config and map."""
  game_objects = get_game_objects_from_map(ascii_map, char_prefab_map, prefabs)
  avatar_objects = build_avatar_objects(num_players, prefabs, player_palettes)
  if use_badges:
    game_objects += build_avatar_badges(num_players, prefabs, badge_palettes)
  return game_objects, avatar_objects


def build_avatar_objects(
    num_players: int,
    prefabs: Optional[Mapping[str, PrefabConfig]] = None,
    player_palettes: Optional[Sequence[shapes.Color]] = None,
) -> List[PrefabConfig]:
  """Build all avatar and their associated game objects from the prefabs."""
  if not prefabs or "avatar" not in prefabs:
    raise ValueError(
        "Building avatar objects requested, but no avatar prefab provided.")

  if not player_palettes:
    player_palettes = [
        shapes.get_palette(colors.palette[i]) for i in range(num_players)]

  avatar_objects = []
  for idx in range(0, num_players):
    game_object = copy.deepcopy(prefabs["avatar"])
    color_palette = player_palettes[idx]
    # Lua is 1-indexed.
    lua_index = idx + 1
    # First, modify the prefab's sprite name.
    sprite_name = get_first_named_component(
        game_object, "Appearance")["kwargs"]["spriteNames"][0]
    new_sprite_name = sprite_name + str(lua_index)
    get_first_named_component(
        game_object,
        "Appearance")["kwargs"]["spriteNames"][0] = new_sprite_name
    # Second, name the same sprite in the prefab's stateManager.
    state_configs = get_first_named_component(
        game_object,
        "StateManager")["kwargs"]["stateConfigs"]
    for state_config in state_configs:
      if "sprite" in state_config and state_config["sprite"] == sprite_name:
        state_config["sprite"] = new_sprite_name
    # Third, override the prefab's color palette for this sprite.
    get_first_named_component(
        game_object, "Appearance")["kwargs"]["palettes"][0] = color_palette
    # Fourth, override the avatar's player id.
    get_first_named_component(
        game_object, "Avatar")["kwargs"]["index"] = lua_index
    avatar_objects.append(game_object)

  return avatar_objects


def build_avatar_badges(
    num_players: int,
    prefabs: Optional[Mapping[str, PrefabConfig]] = None,
    badge_palettes: Optional[Sequence[shapes.Color]] = None,
) -> List[PrefabConfig]:
  """Build all avatar and their associated game objects from the prefabs."""
  if not prefabs or "avatar_badge" not in prefabs:
    raise ValueError(
        "Building avatar badges requested, but no avatar_badge prefab " +
        "provided.")
  game_objects = []

  if badge_palettes is None:
    badge_palettes = [
        shapes.get_palette(colors.palette[i]) for i in range(num_players)]

  for idx in range(0, num_players):
    lua_index = idx + 1
    # Add the overlaid badge on top of each avatar.
    badge_object = copy.deepcopy(prefabs["avatar_badge"])
    sprite_name = get_first_named_component(
        badge_object, "Appearance")["kwargs"]["spriteNames"][0]
    new_sprite_name = sprite_name + str(lua_index)
    get_first_named_component(
        badge_object,
        "Appearance")["kwargs"]["spriteNames"][0] = new_sprite_name
    get_first_named_component(
        badge_object,
        "StateManager")["kwargs"]["stateConfigs"][0]["sprite"] = (
            new_sprite_name)
    get_first_named_component(
        badge_object, "AvatarConnector")["kwargs"]["playerIndex"] = lua_index
    get_first_named_component(
        badge_object,
        "Appearance")["kwargs"]["palettes"][0] = badge_palettes[idx]
    game_objects.append(badge_object)

  return game_objects


def get_game_object_positions_from_map(
    ascii_map: str, char: str, orientation_mode: str = "always_north"
    ) -> Sequence[Transform]:
  """Extract the occurrences of a character in the ascii map into transforms.

  For all occurrences of the given `char`, retrieves a Transform containing the
  position and orientation of the instance.

  Args:
    ascii_map: the ascii map.
    char: the character to extract transforms from the ascii map.
    orientation_mode: select a method for choosing orientations.

  Returns:
    A list of Transforms containing all the positions and orientations of all
    occurrences of the character in the map.
  """
  transforms = []
  rows = ascii_map.split("\n")
  # Assume the first line of the string consists only of '\n'. This means we
  # need to skip the first row.
  for i, row in enumerate(rows[1:]):
    indices = [i for i, c in enumerate(row) if char == c]
    for j in indices:
      if orientation_mode == "always_north":
        orientation = Orientation.NORTH
      else:
        raise ValueError("Other orientation modes are not yet implemented.")
      transform = Transform(position=Position(j, i), orientation=orientation)
      transforms.append(transform)

  return transforms


def _create_game_object(
    prefab: PrefabConfig, transform: Transform) -> PrefabConfig:
  game_object = copy.deepcopy(prefab)
  go_transform = get_first_named_component(game_object, "Transform")
  go_transform["kwargs"] = {
      "position": (transform.position.x, transform.position.y),
      "orientation": transform.orientation.value,
    }
  return game_object


def get_game_objects_from_map(
    ascii_map: str,
    char_prefab_map: Mapping[str, str],
    prefabs: Mapping[str, PrefabConfig],
    random: np.random.RandomState = np.random.RandomState()
) -> List[PrefabConfig]:
  """Returns a list of game object configurations from the map and prefabs.

  Each prefab will have its `Transform` component overwritten to its actual
  location (and orientation, although it is all 'N' by default) in the ASCII
  map.

  Args:
    ascii_map: The map for the level. Defines which prefab to use at each
        position in the map, which is a string defining a matrix of characters.
    char_prefab_map: A dictionary mapping characters in the ascii_map to prefab
        names.
    prefabs: A collection of named prefabs that define a GameObject
        configuration.
    random: An optional random number generator.

  Returns:
    A list of game object configurations from the map and prefabs.
  """
  game_objects = []
  for char, prefab in char_prefab_map.items():
    transforms = get_game_object_positions_from_map(ascii_map, char)
    for transform in transforms:
      if hasattr(prefab, "items"):
        assert "type" in prefab
        assert "list" in prefab
        if prefab["type"] == TYPE_ALL:
          for p in prefab["list"]:
            game_objects.append(_create_game_object(prefabs[p], transform))
        elif prefab["type"] == TYPE_CHOICE:
          game_objects.append(
              _create_game_object(prefabs[random.choice(prefab["list"])],
                                  transform))
      else:  # Typical case, since named prefab.
        game_objects.append(_create_game_object(prefabs[prefab], transform))
  return game_objects
