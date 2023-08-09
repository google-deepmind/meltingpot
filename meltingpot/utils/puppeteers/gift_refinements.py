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
"""Puppeteers for gift_refinements."""

from collections.abc import Mapping

import dm_env
from meltingpot.utils.puppeteers import puppeteer
import numpy as np
import tree

Observation = Mapping[str, tree.Structure[np.ndarray]]


class GiftRefinementsCooperator(puppeteer.Puppeteer[tuple[()]]):
  """Cooperator puppeteer for gift refinements.

  This puppeteer expresses a cooperative high level policy:
  1.  Collect tokens when the inventory is empty.
  2.  If the inventory is not empty, check if there are any refined tokens, if
      not, the gift some tokens.
  3.  If there the player has refined tokens, consume.

  This means that a GiftRefinementsCooperator will start by grabbing a token,
  and then gift it. As soon as they receive any gift from anyone, they would
  consume.
  """

  def __init__(
      self,
      *,
      collect_goal: puppeteer.PuppetGoal,
      gift_goal: puppeteer.PuppetGoal,
      consume_goal: puppeteer.PuppetGoal,
  ):
    """Initializes the puppeteer.

    Args:
      collect_goal: goal to emit to puppet when "collecting"
      gift_goal: goal to emit to puppet when "gifting"
      consume_goal: goal to emit to puppet when "consuming"
    """
    self._collect_goal = collect_goal
    self._gift_goal = gift_goal
    self._consume_goal = consume_goal

  def initial_state(self) -> tuple[()]:
    """See base class."""
    return ()

  def should_consume(self, observation: Observation) -> bool:
    """Decides whether we should consume tokens in our inventory."""
    _, refined, twice_refined = observation['INVENTORY']
    return bool(refined) or bool(twice_refined)

  def step(self, timestep: dm_env.TimeStep,
           prev_state: tuple[()]) -> tuple[dm_env.TimeStep, tuple[()]]:
    """See base class."""
    if np.sum(timestep.observation['INVENTORY']):
      if self.should_consume(timestep.observation):
        goal = self._consume_goal
      else:
        goal = self._gift_goal
    else:
      goal = self._collect_goal

    # Return the encoded cumulant associated with the current goal.
    timestep = puppeteer.puppet_timestep(timestep, goal)
    return timestep, prev_state


class GiftRefinementsExtremeCooperator(GiftRefinementsCooperator):
  """Cooperator that gifts until it has tokens of type 2 (double refinement).

  This means that a GiftRefinementsExtremeCooperator, like the cooperator above,
  will start by grabbing a token, and then gift it. However, upon receiving a
  gift, they would gift back. Only will they consume if they receive a doubly
  refined token.
  """

  def should_consume(self, observation: Observation) -> bool:
    """See base class."""
    _, _, twice_refined = observation['INVENTORY']
    return bool(twice_refined > 0)
