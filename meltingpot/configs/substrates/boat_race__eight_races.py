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
"""Configuration for boat_race__eight_races."""

from meltingpot.configs.substrates import boat_race as base_config
from ml_collections import config_dict as configdict


def get_config() -> configdict.ConfigDict:
  """Configuration for the boat_race substrate."""
  config = base_config.get_config()

  config.num_races = 8

  config.default_player_roles = ("default",) * base_config.MANDATED_NUM_PLAYERS

  return config


build = base_config.build
