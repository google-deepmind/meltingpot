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
"""Configuration for tutorial level: Harvest."""

from ml_collections import config_dict


def get_config():
  """Default configuration for the Harvest level."""
  config = config_dict.ConfigDict()

  # Basic configuration.
  config.individual_observation_names = []
  config.global_observation_names = ["WORLD.RGB"]

  # Lua script configuration.
  config.lab2d_settings = {
      "levelName":
          "harvest",
      "levelDirectory":
          "examples/tutorial/harvest/levels",
      "maxEpisodeLengthFrames":
          100,
      "numPlayers":
          0,
      "spriteSize":
          8,
      "simulation": {
          "map": " ",
          "prefabs": {},
          "charPrefabMap": {},
      },
  }

  return config
