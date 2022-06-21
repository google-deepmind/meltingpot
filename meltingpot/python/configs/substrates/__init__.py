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

"""Configs for substrates."""

import importlib
from typing import AbstractSet

from ml_collections import config_dict


def get_config(substrate: str) -> config_dict.ConfigDict:
  """Returns the specified config.

  Args:
    substrate: the name of the substrate.

  Raises:
    ModuleNotFoundError: the config does not exist.
  """
  if substrate not in SUBSTRATES:
    raise ValueError(f'{substrate} not in {SUBSTRATES}.')
  path = f'{__name__}.{substrate}'
  module = importlib.import_module(path)
  return module.get_config().lock()


SUBSTRATES: AbstractSet[str] = frozenset({
    # keep-sorted start
    'allelopathic_harvest',
    'arena_running_with_scissors_in_the_matrix',
    'bach_or_stravinsky_in_the_matrix',
    'capture_the_flag',
    'chemistry_branched_chain_reaction',
    'chemistry_metabolic_cycles',
    'chicken_in_the_matrix',
    'clean_up',
    'collaborative_cooking_impassable',
    'collaborative_cooking_passable',
    'commons_harvest_closed',
    'commons_harvest_open',
    'commons_harvest_partnership',
    'king_of_the_hill',
    'prisoners_dilemma_in_the_matrix',
    'pure_coordination_in_the_matrix',
    'rationalizable_coordination_in_the_matrix',
    'running_with_scissors_in_the_matrix',
    'stag_hunt_in_the_matrix',
    'territory_open',
    'territory_rooms',
    # keep-sorted end
    'commons_harvest_open_simple',
})
