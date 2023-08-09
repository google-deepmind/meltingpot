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
"""Common functions to be used across multiple *_in_the_matrix substrates."""

import copy
from typing import Any, Dict, Mapping, Sequence

from meltingpot.utils.substrates import shapes


def get_cumulant_metric_configs(
    num_resources: int) -> Sequence[Mapping[str, Any]]:
  """Get metric configs to configure AvatarMetricReporter."""
  cumulants = []
  # One cumulant tracks from frame to frame whether the player participated in
  # an interaction.
  cumulants.append({
      "name": "INTERACTED_THIS_STEP",
      "type": "Doubles",
      "shape": [],
      "component": "GameInteractionZapper",
      "variable": "interacted_this_step",
  })
  for py_idx in range(num_resources):
    lua_idx = py_idx + 1
    # Several cumulants track when resources are collected. There will be one
    # such cumulant per resource type.
    cumulants.append({
        "name": f"COLLECTED_RESOURCE_{lua_idx}",
        "type": "Doubles",
        "shape": [],
        "component": "GameInteractionZapper",
        "variable": f"collected_resource_{lua_idx}",
    })
    # Several cumulants track when resources are destroyed. There will be one
    # such cumulant per resource type.
    cumulants.append({
        "name": f"DESTROYED_RESOURCE_{lua_idx}",
        "type": "Doubles",
        "shape": [],
        "component": "GameInteractionZapper",
        "variable": f"destroyed_resource_{lua_idx}",
    })
    # Sevaral cumulants track which resource was maximal in the interaction on
    # the current frame. There will be one such cumulant per resource type.
    cumulants.append({
        "name": f"ARGMAX_INTERACTION_INVENTORY_WAS_{lua_idx}",
        "type": "Doubles",
        "shape": [],
        "component": "GameInteractionZapper",
        "variable": f"argmax_interaction_inventory_was_{lua_idx}",
    })
  return cumulants


def get_indicator_color_palette(color_rgba):
  indicator_palette = copy.deepcopy(shapes.GOLD_CROWN_PALETTE)
  indicator_palette["#"] = color_rgba
  slightly_darker_color = [round(value * 0.9) for value in color_rgba[:-1]]
  slightly_darker_color.append(150)  # Add a half transparent alpha channel.
  indicator_palette["@"] = slightly_darker_color
  return indicator_palette


def create_ready_to_interact_marker(player_idx: int) -> Dict[str, Any]:
  """Create a ready-to-interact marker overlay object."""
  # Lua is 1-indexed.
  lua_idx = player_idx + 1

  marking_object = {
      "name": "avatarReadyToInteractMarker",
      "components": [
          {
              "component": "StateManager",
              "kwargs": {
                  "initialState": "avatarMarkingWait",
                  "stateConfigs": [
                      # Use `overlay` layer for ready and nonready states, both
                      # are used for live avatars and are always connected.
                      {"state": "ready",
                       "layer": "overlay",
                       "sprite": "Ready"},
                      {"state": "notReady",
                       "layer": "overlay"},

                      # Result indication colors.
                      {"state": "resultIndicatorColor1",
                       "layer": "overlay",
                       "sprite": "ResultIndicatorColor1"},
                      {"state": "resultIndicatorColor2",
                       "layer": "overlay",
                       "sprite": "ResultIndicatorColor2"},
                      {"state": "resultIndicatorColor3",
                       "layer": "overlay",
                       "sprite": "ResultIndicatorColor3"},
                      {"state": "resultIndicatorColor4",
                       "layer": "overlay",
                       "sprite": "ResultIndicatorColor4"},
                      {"state": "resultIndicatorColor5",
                       "layer": "overlay",
                       "sprite": "ResultIndicatorColor5"},

                      # Invisible inactive overlay type.
                      {"state": "avatarMarkingWait",
                       "groups": ["avatarMarkingWaits"]},
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
                  "spriteNames": [
                      "Ready",
                      "ResultIndicatorColor1",
                      "ResultIndicatorColor2",
                      "ResultIndicatorColor3",
                      "ResultIndicatorColor4",
                      "ResultIndicatorColor5",
                  ],
                  "spriteShapes": [shapes.BRONZE_CAP,] * 6,
                  "palettes": [
                      shapes.SILVER_CROWN_PALETTE,
                      # Colors are in rainbow order (more or less).
                      get_indicator_color_palette((139, 0, 0, 255)),  # red
                      get_indicator_color_palette((253, 184, 1, 255)),  # yellow
                      get_indicator_color_palette((0, 102, 0, 255)),  # green
                      get_indicator_color_palette((2, 71, 254, 255)),  # blue
                      get_indicator_color_palette((127, 0, 255, 255)),  # violet
                  ],
                  "noRotates": [True,] * 6,
              }
          },
          {
              "component": "AvatarConnector",
              "kwargs": {
                  "playerIndex": lua_idx,
                  "aliveState": "notReady",  # state `notReady` is invisible.
                  "waitState": "avatarMarkingWait"
              }
          },
          {
              "component": "ReadyToInteractMarker",
              "kwargs": {
                  "playerIndex": lua_idx,
              }
          },
      ]
  }
  return marking_object
