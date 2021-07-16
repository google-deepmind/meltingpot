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
"""Puppeteer functions for puppet bots."""

from typing import Callable, Mapping

import numpy as np
import tree

Observation = Mapping[str, tree.Structure[np.ndarray]]
PuppeteerFn = Callable[[int, Observation], np.ndarray]

CLEAN_UP_CLEAN_GOAL = np.array([1., 0.])
CLEAN_UP_EAT_GOAL = np.array([0., 1.])
CLEAN_UP_CLEAN_ACTION = 8


def cleanup_alternate_clean_first(step_count: int,
                                  observation: Observation) -> np.ndarray:
  """Cleanup puppeteer.

  Args:
    step_count: steps since episode started.
    observation: current observation.

  Returns:
    A goal for the puppet. Alternates cleaning and eating starting with
    cleaning.
  """
  del observation
  if step_count < 250:
    return CLEAN_UP_CLEAN_GOAL
  elif step_count < 500:
    return CLEAN_UP_EAT_GOAL
  elif step_count < 750:
    return CLEAN_UP_CLEAN_GOAL
  else:
    return CLEAN_UP_EAT_GOAL


def cleanup_alternate_eat_first(step_count: int,
                                observation: Observation) -> np.ndarray:
  """Cleanup puppeteer.

  Args:
    step_count: steps since episode started.
    observation: current observation.

  Returns:
    A goal for the puppet. Alternates cleaning and eating starting with
    eating.
  """
  del observation
  if step_count < 250:
    return CLEAN_UP_EAT_GOAL
  elif step_count < 500:
    return CLEAN_UP_CLEAN_GOAL
  elif step_count < 750:
    return CLEAN_UP_EAT_GOAL
  else:
    return CLEAN_UP_CLEAN_GOAL


class ConditionalCleaner(object):
  """Cleanup puppeteer for a reciprocating agent."""

  def __init__(self, threshold: int) -> None:
    """Initializes the puppeteer.

    Args:
      threshold: number of other cleaners below which it will switch to
        cleaning.
    """
    self._clean_until = 0
    self._threshold = threshold
    self._prev_cleaning = None

  def __call__(self, step_count: int, observation: Observation) -> np.ndarray:
    """Puppeteer step.

    Args:
      step_count: steps since episode started.
      observation: current observation.

    Returns:
      A goal for the puppet.
    """
    if step_count == 0:
      self._clean_until = 0

    not_me = 1 - observation['agent_slot']
    # Must have at least 1 other agent cleaning, then I'll help for a while.
    near_river = (observation['global']['observations']['POSITION'][..., 1] < 9)

    # Smooth the cleaning binary vector across 2 timesteps.
    cleaning = observation['global']['actions'] == CLEAN_UP_CLEAN_ACTION
    if self._prev_cleaning is None:
      self._prev_cleaning = cleaning
    smooth_cleaning = np.logical_or(cleaning, self._prev_cleaning)

    # AND together the cleaning, the near river, and the negated identity
    # vectors to figure out the number of other cleaners. Compare to threshold.
    if np.logical_and(not_me, np.logical_and(
        smooth_cleaning, near_river)).sum() >= self._threshold:
      self._clean_until = step_count + 100

    self._prev_cleaning = cleaning
    if step_count < self._clean_until:
      return CLEAN_UP_CLEAN_GOAL
    else:
      return CLEAN_UP_EAT_GOAL


# Note: This assumes resource 0 is "good" and resource 1 is "bad". Thus:
# For PrisonersDilemma, resource 0 is `cooperate` and resource 1 is `defect`.
# For Stag hunt, resource 0 is `stag` and resource 1 is `hare`.
# For Chicken, resource 0 is `dove` and resource 1 is `hawk`.
TWO_RESOURCE_IN_THE_MATRIX_COLLECT_C = np.array([1., 0., 0., 0., 0.])
TWO_RESOURCE_IN_THE_MATRIX_COLLECT_D = np.array([0., 1., 0., 0., 0.])
TWO_RESOURCE_IN_THE_MATRIX_DESTROY_C = np.array([0., 0., 1., 0., 0.])
TWO_RESOURCE_IN_THE_MATRIX_DESTROY_D = np.array([0., 0., 0., 1., 0.])
TWO_RESOURCE_IN_THE_MATRIX_INTERACT = np.array([0., 0., 0., 0., 1.])


class GrimTwoResourceInTheMatrix(object):
  """Puppeteer function for a GRIM strategy in two resource *_in_the_matrix."""

  def __init__(self, threshold: int) -> None:
    """Initializes the puppeteer.

    Args:
      threshold: number of defections after which it will switch behavior.
    """
    self._threshold = threshold
    self._partner_defections = 0
    self._ready_to_interact = False
    self._cooperate_resource_index = 0
    self._defect_resource_index = 1
    self._column_player_is_focal = True

  def _get_focal_and_partner_inventory(self, observation: Observation):
    """Returns the focal and partner inventories from the latest interaction."""
    interaction_inventories = observation['INTERACTION_INVENTORIES']
    row_inventory = interaction_inventories[0]
    col_inventory = interaction_inventories[1]
    if self._column_player_is_focal:
      focal_inventory = col_inventory
      partner_inventory = row_inventory
    else:
      focal_inventory = row_inventory
      partner_inventory = col_inventory
    return focal_inventory, partner_inventory

  def _is_defection(self, inventory: np.ndarray) -> bool:
    """Returns True if `inventory` constitutes defection."""
    num_cooperate_resources = inventory[self._cooperate_resource_index]
    num_defect_resources = inventory[self._defect_resource_index]
    return num_defect_resources > num_cooperate_resources

  def __call__(self, step_count: int, observation: Observation) -> np.ndarray:
    """Puppeteer step.

    Args:
      step_count: steps since episode started.
      observation: current observation.

    Returns:
      A goal for the puppet.
    """
    if step_count == 0:
      self._partner_defections = 0

    # Accumulate partner defections over the episode.
    _, partner_inventory = self._get_focal_and_partner_inventory(observation)
    if self._is_defection(partner_inventory):
      self._partner_defections += 1

    inventory = observation['INVENTORY']
    num_cooperate_resources = inventory[self._cooperate_resource_index]
    num_defect_resources = inventory[self._defect_resource_index]

    # Ready to interact if collected more of either resource than the other.
    self._ready_to_interact = False
    if np.abs(num_defect_resources - num_cooperate_resources) > 0:
      self._ready_to_interact = True

    if not self._ready_to_interact:
      # Collect either C or D when not ready to interact.
      if self._partner_defections < self._threshold:
        # When defection is below threshold, then collect cooperate resources.
        return TWO_RESOURCE_IN_THE_MATRIX_COLLECT_C
      else:
        # When defection exceeds threshold, then collect D resources.
        return TWO_RESOURCE_IN_THE_MATRIX_COLLECT_D
    else:
      # Interact when ready.
      return TWO_RESOURCE_IN_THE_MATRIX_INTERACT
