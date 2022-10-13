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
"""Puppeteers for clean_up puppets."""

from typing import Any, Mapping, Tuple

import dm_env
import numpy as np

from meltingpot.python.utils.puppeteers import puppeteer

_GOALS = puppeteer.puppet_goals(['CLEAN', 'EAT'], dtype=np.float64)
_CLEAN_ACTION = 8


class AlternateCleanFirst(puppeteer.Puppeteer[int]):
  """Alternates cleaning and eating goals, starting with cleaning."""

  def initial_state(self) -> int:
    """See base class."""
    return 0  # step count

  def _goal(self, step_count):
    if step_count < 250:
      return _GOALS['CLEAN']
    elif step_count < 500:
      return _GOALS['EAT']
    elif step_count < 750:
      return _GOALS['CLEAN']
    else:
      return _GOALS['EAT']

  def step(self, timestep: dm_env.TimeStep,
           prev_state: int) -> Tuple[dm_env.TimeStep, int]:
    """See base class."""
    if timestep.first():
      prev_state = self.initial_state()
    goal = self._goal(prev_state)
    next_state = prev_state + 1
    return puppeteer.puppet_timestep(timestep, goal), next_state


class AlternateEatFirst(puppeteer.Puppeteer[int]):
  """Alternates cleaning and eating goals, starting with eating."""

  def initial_state(self) -> int:
    """See base class."""
    return 0

  def _goal(self, step_count):
    if step_count < 250:
      return _GOALS['EAT']
    elif step_count < 500:
      return _GOALS['CLEAN']
    elif step_count < 750:
      return _GOALS['EAT']
    else:
      return _GOALS['CLEAN']

  def step(self, timestep: dm_env.TimeStep,
           prev_state: int) -> Tuple[dm_env.TimeStep, int]:
    """See base class."""
    if timestep.first():
      prev_state = self.initial_state()
    goal = self._goal(prev_state)
    next_state = prev_state + 1
    return puppeteer.puppet_timestep(timestep, goal), next_state


class ConditionalCleaner(puppeteer.Puppeteer[Mapping[str, Any]]):
  """Cleanup puppeteer for a reciprocating agent.

  Requires the agent_slot to be in the observations.
  """

  def __init__(self, threshold: int) -> None:
    """Initializes the puppeteer.

    Args:
      threshold: number of other cleaners below which it will switch to
        cleaning.
    """
    self._threshold = threshold

  def initial_state(self) -> Mapping[str, Any]:
    """See base class."""
    return dict(step_count=0, clean_until=0, cleaning=None)

  def step(
      self,
      timestep: dm_env.TimeStep,
      prev_state: Mapping[str, Any],
  ) -> Tuple[dm_env.TimeStep, Mapping[str, Any]]:
    """See base class."""
    if timestep.first():
      prev_state = self.initial_state()
    observation = timestep.observation
    step_count = prev_state['step_count']
    clean_until = prev_state['clean_until']
    prev_cleaning = prev_state['cleaning']

    not_me = 1 - observation['agent_slot']
    # Must have at least 1 other agent cleaning, then I'll help for a while.
    near_river = (observation['global']['observations']['POSITION'][..., 1] < 9)

    # Smooth the cleaning binary vector across 2 timesteps.
    cleaning = observation['global']['actions'] == _CLEAN_ACTION
    if prev_cleaning is None:
      prev_cleaning = cleaning
    smooth_cleaning = np.logical_or(cleaning, prev_cleaning)

    # AND together the cleaning, the near river, and the negated identity
    # vectors to figure out the number of other cleaners. Compare to threshold.
    if np.logical_and(not_me, np.logical_and(
        smooth_cleaning, near_river)).sum() >= self._threshold:
      clean_until = step_count + 100

    if step_count < clean_until:
      goal = _GOALS['CLEAN']
    else:
      goal = _GOALS['EAT']
    timestep = puppeteer.puppet_timestep(timestep, goal)

    next_state = dict(
        step_count=step_count + 1,
        clean_until=clean_until,
        cleaning=cleaning,
    )
    return timestep, next_state
