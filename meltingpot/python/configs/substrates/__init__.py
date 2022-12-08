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
"""Configs for substrates."""

from collections.abc import Mapping, Sequence, Set
import dataclasses
import functools
import importlib
from typing import Any

from ml_collections import config_dict


def _validated(build):
  """And adds validation checks to build function."""

  def lab2d_settings_builder(
      *,
      config: config_dict.ConfigDict,
      roles: Sequence[str],
  ) -> Mapping[str, Any]:
    """Builds the lab2d settings for the specified config and roles.

    Args:
      config: the meltingpot substrate config.
      roles: the role for each corresponding player.

    Returns:
      The lab2d settings for the substrate.
    """
    invalid_roles = set(roles) - config.valid_roles
    if invalid_roles:
      raise ValueError(f'Invalid roles: {invalid_roles!r}. Must be one of '
                       f'{config.valid_roles!r}')
    return build(config=config, roles=roles)

  return lab2d_settings_builder


def get_config(substrate: str) -> config_dict.ConfigDict:
  """Returns the specified config.

  Args:
    substrate: the name of the substrate. Must be in SUBSTRATES.

  Raises:
    ModuleNotFoundError: the config does not exist.
  """
  if substrate not in SUBSTRATES:
    raise ValueError(f'{substrate} not in {SUBSTRATES}.')
  path = f'{__name__}.{substrate}'
  module = importlib.import_module(path)
  config = module.get_config()
  with config.unlocked():
    config.lab2d_settings_builder = _validated(module.build)
  return config.lock()


SUBSTRATES: Set[str] = frozenset({
    # keep-sorted start
    'allelopathic_harvest__open',
    'bach_or_stravinsky_in_the_matrix__arena',
    'bach_or_stravinsky_in_the_matrix__repeated',
    'boat_race__eight_races',
    'chemistry__three_metabolic_cycles',
    'chemistry__three_metabolic_cycles_with_plentiful_distractors',
    'chemistry__two_metabolic_cycles',
    'chemistry__two_metabolic_cycles_with_distractors',
    'chicken_in_the_matrix__arena',
    'chicken_in_the_matrix__repeated',
    'clean_up',
    'coins',
    'collaborative_cooking__asymmetric',
    'collaborative_cooking__circuit',
    'collaborative_cooking__cramped',
    'collaborative_cooking__crowded',
    'collaborative_cooking__figure_eight',
    'collaborative_cooking__forced',
    'collaborative_cooking__ring',
    'commons_harvest__closed',
    'commons_harvest__open',
    'commons_harvest__partnership',
    'coop_mining',
    'daycare',
    'externality_mushrooms__dense',
    'factory_commons__either_or',
    'fruit_market__concentric_rivers',
    'gift_refinements',
    'hidden_agenda',
    'paintball__capture_the_flag',
    'paintball__king_of_the_hill',
    'predator_prey__alley_hunt',
    'predator_prey__open',
    'predator_prey__orchard',
    'predator_prey__random_forest',
    'prisoners_dilemma_in_the_matrix__arena',
    'prisoners_dilemma_in_the_matrix__repeated',
    'pure_coordination_in_the_matrix__arena',
    'pure_coordination_in_the_matrix__repeated',
    'rationalizable_coordination_in_the_matrix__arena',
    'rationalizable_coordination_in_the_matrix__repeated',
    'running_with_scissors_in_the_matrix__arena',
    'running_with_scissors_in_the_matrix__one_shot',
    'running_with_scissors_in_the_matrix__repeated',
    'stag_hunt_in_the_matrix__arena',
    'stag_hunt_in_the_matrix__repeated',
    'territory__inside_out',
    'territory__open',
    'territory__rooms',
    # keep-sorted end
})
