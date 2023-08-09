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
"""Puppeteers for *_in_the_matrix."""

from collections.abc import Mapping, Sequence
import dataclasses
import random
from typing import Optional, TypeVar

import dm_env
from meltingpot.utils.puppeteers import puppeteer
import numpy as np
import tree

State = TypeVar("State")
Observation = Mapping[str, tree.Structure[np.ndarray]]


def get_inventory(timestep: dm_env.TimeStep) -> np.ndarray:
  """Returns player's current inventory."""
  return timestep.observation["INVENTORY"]


def get_partner_interaction_inventory(
    timestep: dm_env.TimeStep) -> Optional[np.ndarray]:
  """Returns the partner inventory from previous interaction."""
  _, partner_inventory = timestep.observation["INTERACTION_INVENTORIES"]
  if np.all(partner_inventory < 0):
    return None  # No interaction occurred.
  else:
    return partner_inventory


def has_interaction(timestep: dm_env.TimeStep) -> bool:
  """Returns True if the timestep contains an interaction."""
  return get_partner_interaction_inventory(timestep) is not None


def max_resource_and_margin(inventory: np.ndarray) -> tuple[int, int]:
  """Returns the index of the maximum resource and the margin of its lead."""
  sorted_resources = np.argsort(inventory)
  maximum_resource = sorted_resources[-1]
  margin = (
      int(inventory[sorted_resources[-1]]) -
      int(inventory[sorted_resources[-2]]))
  return maximum_resource, margin


def has_collected_sufficient(
    inventory: np.ndarray,
    resource: int,
    margin: int,
) -> bool:
  """Returns True if a sufficient amount of the resource has been collected.

  Args:
    inventory: the inventory of collected resources.
    resource: the index of the resource being collected.
    margin: the required margin for "sufficiency".
  """
  max_resource, current_margin = max_resource_and_margin(inventory)
  return max_resource == resource and current_margin >= margin


def partner_max_resource(timestep: dm_env.TimeStep) -> Optional[int]:
  """Returns partner's maximum resource at previous interaction."""
  partner_inventory = get_partner_interaction_inventory(timestep)
  if partner_inventory is None:
    return None  # No interaction occurred.
  resource, margin = max_resource_and_margin(partner_inventory)
  if margin == 0:
    return None  # Intent is unclear (no unique maximum).
  else:
    return resource


def tremble(tremble_probability: float):
  """Returns True if the hand trembles."""
  return random.random() < tremble_probability


@dataclasses.dataclass(frozen=True)
class Resource:
  """A resource that can be collected by a puppet.

  Attributes:
    index: the index of the resource in the INVENTORY vector.
    collect_goal: the goal that directs the puppet to collect the resource.
    interact_goal: the goal that directs the puppet to interact with another
      player while playing the resource.
  """
  index: int
  collect_goal: puppeteer.PuppetGoal
  interact_goal: puppeteer.PuppetGoal

  def __eq__(self, obj):
    if not isinstance(obj, Resource):
      return NotImplemented
    else:
      return self is obj

  def __hash__(self):
    return hash(id(self))


def collect_or_interact_puppet_timestep(
    timestep: dm_env.TimeStep,
    target: Resource,
    margin: int,
) -> dm_env.TimeStep:
  """Returns a timestep for a *_in_the_matrix puppet.

  Args:
    timestep: the timestep without any goal added.
    target: the resource for the collector to target.
    margin: the threshold at which the puppet switches from collecting to
      interacting.

  Returns:
    A timestep with a goal added for the puppet. If the puppet has already
    collected enough of the targeted resource, will add the resource's
    interact_goal. Otherwise will add the resource's collect_goal.
  """
  inventory = get_inventory(timestep)
  if has_collected_sufficient(inventory, target.index, margin):
    goal = target.interact_goal
  else:
    goal = target.collect_goal
  return puppeteer.puppet_timestep(timestep, goal)


class Specialist(puppeteer.Puppeteer[tuple[()]]):
  """Puppeteer that targets a single resource."""

  def __init__(self, *, target: Resource, margin: int) -> None:
    """Initializes the puppeteer.

    Args:
      target: the resource to target.
      margin: the margin at which the specialist will switch from collecting to
        interacting.
    """
    self._target = target
    if margin > 0:
      self._margin = margin
    else:
      raise ValueError("Margin must be positive.")

  def initial_state(self) -> tuple[()]:
    """See base class."""
    return ()

  def step(self, timestep: dm_env.TimeStep,
           prev_state: tuple[()]) -> tuple[dm_env.TimeStep, tuple[()]]:
    """See base class."""
    timestep = collect_or_interact_puppet_timestep(
        timestep, self._target, self._margin)
    return timestep, prev_state


class AlternatingSpecialist(puppeteer.Puppeteer[int]):
  """Puppeteer that cycles targeted resource on a fixed schedule."""

  def __init__(self,
               *,
               targets: Sequence[Resource],
               interactions_per_target: int,
               margin: int) -> None:
    """Initializes the puppeteer.

    Args:
      targets: circular sequence of resources to target. Targets correspond to
        pure strategies in the underlying matrix game.
      interactions_per_target: how many interactions to select each target
        before switching to the next one in the `targets` sequence.
      margin: Try to collect `margin` more of the target resource than the other
        resources before interacting.
    """
    if targets:
      self._targets = tuple(targets)
    else:
      raise ValueError("targets must not be empty")

    if interactions_per_target > 0:
      self._interactions_per_target = interactions_per_target
    else:
      raise ValueError("interactions_per_target must be positive.")

    if margin > 0:
      self._margin = margin
    else:
      raise ValueError("margin must be positive.")

  def initial_state(self) -> int:
    """See base class."""
    return 0

  def step(self, timestep: dm_env.TimeStep,
           prev_state: int) -> tuple[dm_env.TimeStep, int]:
    """See base class."""
    if timestep.first():
      prev_state = self.initial_state()

    if has_interaction(timestep):
      total_interactions = prev_state + 1
    else:
      total_interactions = prev_state

    target_index = (total_interactions // self._interactions_per_target) % len(
        self._targets)
    target = self._targets[target_index]

    timestep = collect_or_interact_puppet_timestep(
        timestep, target, self._margin)

    return timestep, total_interactions


class ScheduledFlip(puppeteer.Puppeteer[int]):
  """Puppeteer that targets one resource then switches to another."""

  def __init__(
      self,
      *,
      threshold: int,
      initial_target: Resource,
      final_target: Resource,
      initial_margin: int,
      final_margin: int,
  ) -> None:
    """Initializes the puppeteer.

    Args:
      threshold: Switch targeted resource once this many interactions have
        occurred.
      initial_target: The initial resource to target.
      final_target: The resource to target after the switch.
      initial_margin: How much more of the target resource to collect before
        interacting.
      final_margin: The margin after the flip.
    """
    self._initial_target = initial_target
    self._final_target = final_target

    if threshold > 0:
      self._threshold = threshold
    else:
      raise ValueError("threshold must be positive.")

    if initial_margin > 0:
      self._initial_margin = initial_margin
    else:
      raise ValueError("initial_margin must be positive.")

    if final_margin > 0:
      self._final_margin = final_margin
    else:
      raise ValueError("final_margin must be positive.")

  def initial_state(self) -> int:
    """See base class."""
    return 0

  def step(self, timestep: dm_env.TimeStep,
           prev_state: int) -> tuple[dm_env.TimeStep, int]:
    """See base class."""
    if timestep.first():
      prev_state = self.initial_state()

    if has_interaction(timestep):
      total_interactions = prev_state + 1
    else:
      total_interactions = prev_state

    if total_interactions < self._threshold:
      timestep = collect_or_interact_puppet_timestep(
          timestep, self._initial_target, self._initial_margin)
    else:
      timestep = collect_or_interact_puppet_timestep(
          timestep, self._final_target, self._final_margin)

    return timestep, total_interactions


class GrimTrigger(puppeteer.Puppeteer[int]):
  """Puppeteer for a grim trigger.

  This bot will always try to play cooperate until other players have defected
  against it more than `threshold` times. After enduring `threshold` defections,
  it switches to a triggered mode where it always plays defect. It never leaves
  this mode, i.e. it is grim. It defects in all future interactions, not only
  those interactions with the players who originally defected on it.
  """

  def __init__(
      self,
      *,
      threshold: int,
      cooperate_resource: Resource,
      defect_resource: Resource,
      margin: int,
  ) -> None:
    """Initializes the puppeteer.

    Args:
      threshold: How many defections cause this agent to switch to its triggered
        mode. Once triggered it will try to defect in all future interactions.
      cooperate_resource: the cooperation resource.
      defect_resource: the defection resource.
      margin: Try to collect `margin` more of the target resource than the other
        resource before interacting.
    """
    if threshold > 0:
      self._threshold = threshold
    else:
      raise ValueError("threshold must be positive")

    self._cooperate_resource = cooperate_resource
    self._defect_resource = defect_resource

    if margin > 0:
      self._margin = margin
    else:
      raise ValueError("margin must be positive")

  def initial_state(self) -> int:
    """See base class."""
    return 0

  def step(self, timestep: dm_env.TimeStep,
           prev_state: int) -> tuple[dm_env.TimeStep, int]:
    """See base class."""
    if timestep.first():
      prev_state = self.initial_state()

    partner_resource = partner_max_resource(timestep)
    partner_defected = partner_resource == self._defect_resource.index
    if partner_defected:
      partner_defections = prev_state + 1
    else:
      partner_defections = prev_state

    if partner_defections < self._threshold:
      timestep = collect_or_interact_puppet_timestep(
          timestep, self._cooperate_resource, self._margin)
    else:
      timestep = collect_or_interact_puppet_timestep(
          timestep, self._defect_resource, self._margin)
    return timestep, partner_defections


class TitForTat(puppeteer.Puppeteer[bool]):
  """Puppeteer for a tit-for-tat bot.

  This bot will always try to cooperate if its partner cooperated in the last
  round and defect if its partner defected in the last round. It cooperates
  on the first round.

  Important note: this puppeteer function assumes there is only one other player
  in the game. So it only makes sense for two player substrates like those we
  called *_in_the_matrix__repeated.
  """

  def __init__(
      self,
      *,
      cooperate_resource: Resource,
      defect_resource: Resource,
      margin: int,
      tremble_probability: float,
  ) -> None:
    """Initializes the puppeteer.

    Args:
      cooperate_resource: the cooperation resource.
      defect_resource: the defection resource.
      margin: Try to collect `margin` more of the target resource than the other
        resource before interacting.
      tremble_probability: When deciding to cooperate/defect, switch to
        defect/cooperate with this probability.
    """
    self._cooperate_resource = cooperate_resource
    self._defect_resource = defect_resource

    if margin > 0:
      self._margin = margin
    else:
      raise ValueError("margin must be positive")

    if 0 <= tremble_probability <= 1:
      self._tremble_probability = tremble_probability
    else:
      raise ValueError("tremble_probability must be a probability.")

  def initial_state(self) -> bool:
    """See base class."""
    is_cooperative = True if not tremble(self._tremble_probability) else False
    return is_cooperative

  def step(self, timestep: dm_env.TimeStep,
           prev_state: bool) -> tuple[dm_env.TimeStep, bool]:
    """See base class."""
    if timestep.first():
      prev_state = self.initial_state()

    partner_resource = partner_max_resource(timestep)
    partner_defected = partner_resource == self._defect_resource.index
    partner_cooperated = partner_resource == self._cooperate_resource.index

    if partner_cooperated:
      is_cooperative = True if not tremble(self._tremble_probability) else False
    elif partner_defected:
      is_cooperative = False if not tremble(self._tremble_probability) else True
    else:
      is_cooperative = prev_state

    if is_cooperative:
      timestep = collect_or_interact_puppet_timestep(
          timestep, self._cooperate_resource, self._margin)
    else:
      timestep = collect_or_interact_puppet_timestep(
          timestep, self._defect_resource, self._margin)
    return timestep, is_cooperative


@dataclasses.dataclass(frozen=True)
class CorrigableState:
  """State of Corrigable puppeteer.

  Attributes:
    partner_defections: the number of times the partner has defected.
    is_cooperative: whether the puppeteer is currently cooperating (as opposed
       to defecting).
  """
  partner_defections: int
  is_cooperative: bool


class Corrigible(puppeteer.Puppeteer[CorrigableState]):
  """Puppeteer that defects until you punish it, then switches to tit-for-tat.

  Important note: this puppeteer function assumes there is only one other player
  in the game. So it only makes sense for two player substrates like those we
  called *_in_the_matrix__repeated.
  """

  def __init__(
      self,
      threshold: int,
      cooperate_resource: Resource,
      defect_resource: Resource,
      margin: int,
      tremble_probability: float,
  ) -> None:
    """Initializes the puppeteer.

    Args:
      threshold: How many times this bot must be punished for it to change its
        behavior from 'always defect' to 'tit-for-tat'.
      cooperate_resource: the cooperation resource.
      defect_resource: the defection resource.
      margin: Try to collect `margin` more of the target resource than the other
        resource before interacting.
      tremble_probability: Once playing tit-for-tat, when deciding to
        cooperate/defect, switch to defect/cooperate with this probability.
    """
    if threshold > 0:
      self._threshold = threshold
    else:
      raise ValueError("threshold must be positive.")

    self._cooperate_resource = cooperate_resource
    self._defect_resource = defect_resource

    if margin > 0:
      self._margin = margin
    else:
      raise ValueError("margin must be positive")

    if 0 <= tremble_probability <= 1:
      self._tremble_probability = tremble_probability
    else:
      raise ValueError("tremble_probability must be a probability.")

  def initial_state(self) -> CorrigableState:
    """See base class."""
    return CorrigableState(partner_defections=0, is_cooperative=False)

  def step(
      self,
      timestep: dm_env.TimeStep,
      prev_state: CorrigableState,
  ) -> tuple[dm_env.TimeStep, CorrigableState]:
    """See base class."""
    if timestep.first():
      prev_state = self.initial_state()

    partner_resource = partner_max_resource(timestep)
    partner_defected = partner_resource == self._defect_resource.index
    partner_cooperated = partner_resource == self._cooperate_resource.index

    if partner_defected:
      partner_defections = prev_state.partner_defections + 1
      switching_now = partner_defections == self._threshold
    else:
      partner_defections = prev_state.partner_defections
      switching_now = False

    insufficiently_punished = partner_defections < self._threshold
    if insufficiently_punished:
      is_cooperative = False
    elif switching_now or partner_cooperated:
      is_cooperative = True if not tremble(self._tremble_probability) else False
    elif partner_defected:
      is_cooperative = False if not tremble(self._tremble_probability) else True
    else:
      is_cooperative = prev_state.is_cooperative

    if is_cooperative:
      timestep = collect_or_interact_puppet_timestep(
          timestep, self._cooperate_resource, self._margin)
    else:
      timestep = collect_or_interact_puppet_timestep(
          timestep, self._defect_resource, self._margin)
    next_state = CorrigableState(
        is_cooperative=is_cooperative, partner_defections=partner_defections)
    return timestep, next_state


class RespondToPrevious(puppeteer.Puppeteer[Resource]):
  """Puppeteer for responding to opponents previous move.

  At the start of an episode, RespondToPrevious targets a random resource up
  until the first interaction occurs. Thereafter RespondToPrevious selects the
  resource to target based on the maximum resource held by the coplayer at the
  last interaction. If the coplayer held no single maximum resource,
  RespondToPrevious will continue to target the resource it was previously
  targeting.
  """

  def __init__(
      self,
      responses: Mapping[Resource, Resource],
      margin: int,
  ) -> None:
    """Initializes the puppeteer.

    Args:
      responses: Mapping from the maximum resource in the partner inventory to
        the resource to target in response.
      margin: Try to collect `margin` more of the target resource than the other
        resource before interacting.
    """
    self._responses = {
        resource.index: response for resource, response in responses.items()
    }
    if margin > 0:
      self._margin = margin
    else:
      raise ValueError("margin must be positive.")

  def initial_state(self) -> Resource:
    """See base class."""
    return random.choice(list(self._responses.values()))

  def step(
      self,
      timestep: dm_env.TimeStep,
      prev_state: Resource,
  ) -> tuple[dm_env.TimeStep, Resource]:
    """See base class."""
    if timestep.first():
      prev_state = self.initial_state()
    partner_resource = partner_max_resource(timestep)
    response = self._responses.get(partner_resource, prev_state)
    timestep = collect_or_interact_puppet_timestep(
        timestep, response, self._margin)
    return timestep, response
