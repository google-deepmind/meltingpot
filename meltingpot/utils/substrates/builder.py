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
"""Multi-player environment builder for Melting Pot levels."""

from collections.abc import Mapping
import copy
import itertools
import os
import random
from typing import Any, Optional, Union

from absl import logging
import dmlab2d
from dmlab2d import runfiles_helper
from dmlab2d import settings_helper
from meltingpot.utils.substrates import game_object_utils
from meltingpot.utils.substrates.wrappers import reset_wrapper
from ml_collections import config_dict
import tree

Settings = Union[config_dict.ConfigDict, Mapping[str, Any]]

_MAX_SEED = 2 ** 32 - 1
_DMLAB2D_ROOT = runfiles_helper.find()


def _find_root() -> str:
  import re  # pylint: disable=g-import-not-at-top
  return re.sub('^(.*)/meltingpot/.*?$', r'\1', __file__)


_MELTINGPOT_ROOT = _find_root()


# Although to_dict in ConfigDict is recursive, it is not enough for our use case
# because the recursion will _not_ go into the list elements. And we have plenty
# of those in our configs.
def _config_dict_to_dict(value):
  if isinstance(value, config_dict.ConfigDict):
    return tree.map_structure(_config_dict_to_dict, value.to_dict())
  return value


def parse_python_settings_for_dmlab2d(
    lab2d_settings: config_dict.ConfigDict) -> dict[str, Any]:
  """Flatten lab2d_settings into Lua-friendly properties."""
  # Since config_dicts disallow "." in keys, we must use a different character,
  # "$", in our config and then convert it to "." here. This is particularly
  # important for levels with config keys like 'player.%default' in DMLab2D.
  lab2d_settings = _config_dict_to_dict(lab2d_settings)
  lab2d_settings = settings_helper.flatten_args(lab2d_settings)
  lab2d_settings_dict = {}
  for key, value in lab2d_settings.items():
    converted_key = key.replace("$", ".")
    lab2d_settings_dict[converted_key] = str(value)
  return lab2d_settings_dict


def apply_prefab_overrides(
    lab2d_settings: config_dict.ConfigDict,
    prefab_overrides: Optional[Settings] = None) -> None:
  """Apply prefab overrides to lab2d_settings."""
  if "gameObjects" not in lab2d_settings.simulation:
    lab2d_settings.simulation.gameObjects = []

  # Edit prefabs with the overrides, both in lab2d_settings and in prefabs.
  if prefab_overrides:
    for prefab, override in prefab_overrides.items():
      for component, arg_overrides in override.items():
        for arg_name, arg_override in arg_overrides.items():
          if prefab not in lab2d_settings.simulation.prefabs:
            raise ValueError(f"Prefab override for '{prefab}' given, but not " +
                             "available in `prefabs`.")
          game_object_utils.get_first_named_component(
              lab2d_settings.simulation.prefabs[prefab],
              component)["kwargs"][arg_name] = arg_override


def maybe_build_and_add_avatar_objects(
    lab2d_settings: config_dict.ConfigDict) -> None:
  """If requested, build the avatar objects and add them to lab2d_settings.

  Avatars will be built here if and only if:
  1) An 'avatar' prefab is supplied in lab2d_settings.simulation.prefabs; and
  2) lab2d_settings.simulation.buildAvatars is not True.

  Avatars built here will have their colors set from the palette provided in
  lab2d_settings.simulation.playerPalettes, or if none is provided, using the
  first num_players colors in the colors.py module.

  Args:
    lab2d_settings: A writable version of the lab2d_settings. Avatar objects,
      if they are to be built here, will be added as game objects in
      lab2d_settings.simulation.gameObjects.
  """
  # Whether the avatars will be built in Lua (False) or here (True). This is
  # roughly the opposite of the `buildAvatars` setting.
  build_avatars_here = ("avatar" in lab2d_settings.simulation.prefabs)
  if ("buildAvatars" in lab2d_settings.simulation
      and lab2d_settings.simulation.buildAvatars):
    build_avatars_here = False
    if "avatar" not in lab2d_settings.simulation.prefabs:
      raise ValueError(
          "Deferring avatar building to Lua, yet no 'avatar' prefab given.")
  if build_avatars_here:
    palettes = (lab2d_settings.simulation.playerPalettes
                if "playerPalettes" in lab2d_settings.simulation else None)
    if "gameObjects" not in lab2d_settings.simulation:
      lab2d_settings.simulation.gameObjects = []
    # Create avatars.
    logging.info("Building avatars in `meltingpot.builder` with palettes: %s",
                 lab2d_settings.simulation.playerPalettes)
    avatar_objects = game_object_utils.build_avatar_objects(
        int(lab2d_settings.numPlayers),
        lab2d_settings.simulation.prefabs,
        palettes)
    lab2d_settings.simulation.gameObjects += avatar_objects


def locate_and_overwrite_level_directory(
    lab2d_settings: config_dict.ConfigDict) -> None:
  """Locates the run files, and overwrites the levelDirectory with it."""
  # Locate runfiles.
  level_name = lab2d_settings.get("levelName")
  level_dir = lab2d_settings.get("levelDirectory")
  if level_dir:
    lab2d_settings.levelName = os.path.join(level_dir, level_name)
    lab2d_settings.levelDirectory = _MELTINGPOT_ROOT


def builder(
    lab2d_settings: Settings,
    prefab_overrides: Optional[Settings] = None,
    env_seed: Optional[int] = None,
    **settings) -> dmlab2d.Environment:
  """Builds a Melting Pot environment.

  Args:
    lab2d_settings: a dict of environment designation args.
    prefab_overrides: overrides for prefabs.
    env_seed: the seed to pass to the environment.
    **settings: Other settings which are not used by Melting Pot but can still
      be passed from the environment builder.

  Returns:
    A multi-player Melting Pot environment.
  """
  del settings  #  Not currently used by DMLab2D.

  assert "simulation" in lab2d_settings

  # Copy config, so as not to modify it.
  lab2d_settings = config_dict.ConfigDict(
      copy.deepcopy(lab2d_settings)).unlock()

  apply_prefab_overrides(lab2d_settings, prefab_overrides)
  maybe_build_and_add_avatar_objects(lab2d_settings)
  locate_and_overwrite_level_directory(lab2d_settings)

  # Convert settings from python to Lua format.
  lab2d_settings_dict = parse_python_settings_for_dmlab2d(lab2d_settings)

  if env_seed is None:
    # Select a long seed different than zero.
    env_seed = random.randint(1, _MAX_SEED)
  env_seeds = (seed % (_MAX_SEED + 1) for seed in itertools.count(env_seed))

  def build_environment():
    seed = next(env_seeds)
    lab2d_settings_dict["env_seed"] = str(seed)  # Sets the Lua seed.
    env_raw = dmlab2d.Lab2d(_DMLAB2D_ROOT, lab2d_settings_dict)
    observation_names = env_raw.observation_names()
    return dmlab2d.Environment(
        env=env_raw,
        observation_names=observation_names,
        seed=seed)

  # Add a wrapper that rebuilds the environment when reset is called.
  env = reset_wrapper.ResetWrapper(build_environment)

  return env
