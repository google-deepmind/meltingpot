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
"""Puppeteer that alternates between goals."""

from collections.abc import Sequence

import dm_env
from meltingpot.utils.puppeteers import puppeteer


class Alternator(puppeteer.Puppeteer[int]):
  """Puppeteer that cycles over a list of goals on a fixed schedule."""

  def __init__(self,
               *,
               goals: Sequence[puppeteer.PuppetGoal],
               steps_per_goal: int) -> None:
    """Initializes the puppeteer.

    Args:
      goals: circular sequence of goals to emit.
      steps_per_goal: how many steps to use each goal before switching to the
        next one in the sequence.
    """
    if steps_per_goal > 0:
      self._steps_per_goal = steps_per_goal
    else:
      raise ValueError('steps_per_goal must be positive.')

    if goals:
      self._goals = list(goals)
    else:
      raise ValueError('goals must not be empty.')

  def initial_state(self) -> int:
    """See base class."""
    return 0  # step count.

  def step(self, timestep: dm_env.TimeStep,
           prev_state: int) -> tuple[dm_env.TimeStep, int]:
    """See base class."""
    if timestep.first():
      prev_state = self.initial_state()
    step_count = prev_state
    goal_index = step_count // self._steps_per_goal % len(self._goals)
    timestep = puppeteer.puppet_timestep(timestep, self._goals[goal_index])
    return timestep, step_count + 1
