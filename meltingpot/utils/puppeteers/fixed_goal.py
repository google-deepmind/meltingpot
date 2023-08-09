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
"""Puppeteer the emits a fixed goal."""

import dm_env
from meltingpot.utils.puppeteers import puppeteer


class FixedGoal(puppeteer.Puppeteer[tuple[()]]):
  """Puppeteer that emits the same goal on every step."""

  def __init__(self, goal: puppeteer.PuppetGoal) -> None:
    """Initializes the puppeteer.

    Args:
      goal: goal to pass to the puppet.
    """
    self._goal = goal

  def initial_state(self) -> tuple[()]:
    """See base class."""
    return ()

  def step(self, timestep: dm_env.TimeStep,
           prev_state: tuple[()]) -> tuple[dm_env.TimeStep, tuple[()]]:
    """See base class."""
    timestep = puppeteer.puppet_timestep(timestep, self._goal)
    return timestep, prev_state
