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
"""Puppeteers for puppet bots."""

from typing import Tuple

import dm_env
import numpy as np

from meltingpot.python.utils.puppeteers import puppeteer

# Note: This assumes resource 0 is "good" and resource 1 is "bad". Thus:
# For PrisonersDilemma, resource 0 is `cooperate` and resource 1 is `defect`.
# For Stag hunt, resource 0 is `stag` and resource 1 is `hare`.
# For Chicken, resource 0 is `dove` and resource 1 is `hawk`.
_TWO_RESOURCE_GOALS = puppeteer.puppet_goals([
    'COLLECT_COOPERATE',
    'COLLECT_DEFECT',
    'DESTROY_COOPERATE',
    'DESTROY_DEFECT',
    'INTERACT',
], dtype=np.float64)


class GrimTwoResource(puppeteer.Puppeteer[int]):
  """Puppeteer function for a GRIM strategy in two resource *_in_the_matrix."""

  def __init__(self, threshold: int) -> None:
    """Initializes the puppeteer.

    Args:
      threshold: number of defections after which it will switch behavior.
    """
    self._threshold = threshold
    self._cooperate_resource_index = 0
    self._defect_resource_index = 1

  def initial_state(self) -> int:
    """See base class."""
    partner_defections = 0
    return partner_defections

  def _get_focal_and_partner_inventory(self, timestep: dm_env.TimeStep):
    """Returns the focal and partner inventories from the latest interaction."""
    interaction_inventories = timestep.observation['INTERACTION_INVENTORIES']
    focal_inventory = interaction_inventories[0]
    partner_inventory = interaction_inventories[1]
    return focal_inventory, partner_inventory

  def _is_defection(self, inventory: np.ndarray) -> bool:
    """Returns True if `inventory` constitutes defection."""
    num_cooperate_resources = inventory[self._cooperate_resource_index]
    num_defect_resources = inventory[self._defect_resource_index]
    return num_defect_resources > num_cooperate_resources

  def step(self, timestep: dm_env.TimeStep,
           prev_state: int) -> Tuple[dm_env.TimeStep, int]:
    """See base class."""
    if timestep.first():
      prev_state = self.initial_state()
    partner_defections = prev_state

    # Accumulate partner defections over the episode.
    _, partner_inventory = self._get_focal_and_partner_inventory(timestep)
    if self._is_defection(partner_inventory):
      partner_defections += 1

    inventory = timestep.observation['INVENTORY']
    num_cooperate_resources = inventory[self._cooperate_resource_index]
    num_defect_resources = inventory[self._defect_resource_index]

    # Ready to interact if collected more of either resource than the other.
    ready_to_interact = False
    if np.abs(num_defect_resources - num_cooperate_resources) > 0:
      ready_to_interact = True

    if not ready_to_interact:
      # Collect either C or D when not ready to interact.
      if partner_defections < self._threshold:
        # When defection is below threshold, then collect cooperate resources.
        goal = _TWO_RESOURCE_GOALS['COLLECT_COOPERATE']
      else:
        # When defection exceeds threshold, then collect D resources.
        goal = _TWO_RESOURCE_GOALS['COLLECT_DEFECT']
    else:
      # Interact when ready.
      goal = _TWO_RESOURCE_GOALS['INTERACT']
    timestep = puppeteer.puppet_timestep(timestep, goal)
    next_state = partner_defections
    return timestep, next_state
