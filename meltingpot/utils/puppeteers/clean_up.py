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
"""Puppeteers for clean_up."""

import dataclasses

import dm_env
from meltingpot.utils.puppeteers import puppeteer


@dataclasses.dataclass(frozen=True)
class ConditionalCleanerState:
  """Current state of the ConditionalCleaner.

  Attributes:
    step_count: number of timesteps previously seen in this episode.
    clean_until: step_count after which to stop cleaning.
    recent_cleaning: number of others cleaning on previous timesteps (ordered
      from oldest to most recent).
  """
  step_count: int
  clean_until: int
  recent_cleaning: tuple[int, ...]


class ConditionalCleaner(puppeteer.Puppeteer[ConditionalCleanerState]):
  """Puppeteer for a reciprocating agent.

  This puppeteer's behavior depends on the behavior of others. In particular, it
  tracks the total amount of others' "cleaning", and integrates this signal
  using a rolling window.

  Initially, the puppet will be in a "nice" mode where it will direct the
  puppet to clean the river for a fixed period. Once this period is over, the
  puppeteer will fall into a "eating" mode where it will direct the puppet to
  only eat apples. However, once the total level of others' cleaning reaches a
  threshold, the puppeteer will temporarily switch to a "cleaning" mode. Once
  the total level of others' cleaning drops back below threshold, the puppeteer
  will clean for fixed number of steps before falling back into the "eating"
  mode.
  """

  def __init__(self,
               *,
               clean_goal: puppeteer.PuppetGoal,
               eat_goal: puppeteer.PuppetGoal,
               coplayer_cleaning_signal: str,
               recency_window: int,
               threshold: int,
               reciprocation_period: int,
               niceness_period: int) -> None:
    """Initializes the puppeteer.

    Args:
      clean_goal: goal to emit to puppet when "cleaning".
      eat_goal: goal to emit to puppet when "eating".
      coplayer_cleaning_signal: key in observations that provides the
        privileged observation of number of others cleaning in the previous
        timestep.
      recency_window: number of steps over which to remember others' behavior.
      threshold: if the total number of (nonunique) cleaners over the
        remembered period reaches this threshold, the puppeteer will direct the
        puppet to clean.
      reciprocation_period: the number of steps to clean for once others'
        cleaning has been forgotten and fallen back below threshold.
      niceness_period: the number of steps to unconditionally clean for at
        the start of the episode.
    """
    self._clean_goal = clean_goal
    self._eat_goal = eat_goal
    self._coplayer_cleaning_signal = coplayer_cleaning_signal

    if threshold > 0:
      self._threshold = threshold
    else:
      raise ValueError('threshold must be positive')

    if recency_window > 0:
      self._recency_window = recency_window
    else:
      raise ValueError('recency_window must be positive')

    if reciprocation_period > 0:
      self._reciprocation_period = reciprocation_period
    else:
      raise ValueError('reciprocation_period must be positive')

    if niceness_period >= 0:
      self._niceness_period = niceness_period
    else:
      raise ValueError('niceness_period must be nonnegative')

  def initial_state(self) -> ConditionalCleanerState:
    """See base class."""
    return ConditionalCleanerState(
        step_count=0, clean_until=self._niceness_period, recent_cleaning=())

  def step(
      self, timestep: dm_env.TimeStep, prev_state: ConditionalCleanerState
  ) -> tuple[dm_env.TimeStep, ConditionalCleanerState]:
    """See base class."""
    if timestep.first():
      prev_state = self.initial_state()
    step_count = prev_state.step_count
    clean_until = prev_state.clean_until
    recent_cleaning = prev_state.recent_cleaning

    coplayers_cleaning = int(
        timestep.observation[self._coplayer_cleaning_signal])
    recent_cleaning += (coplayers_cleaning,)
    recent_cleaning = recent_cleaning[-self._recency_window:]

    smooth_cleaning = sum(recent_cleaning)
    if smooth_cleaning >= self._threshold:
      clean_until = max(clean_until, step_count + self._reciprocation_period)
      # Do not clear the recent_cleaning history after triggering.
      # TODO(b/237058204): clear history in future versions.

    if step_count < clean_until:
      goal = self._clean_goal
    else:
      goal = self._eat_goal
    timestep = puppeteer.puppet_timestep(timestep, goal)

    next_state = ConditionalCleanerState(
        step_count=step_count + 1,
        clean_until=clean_until,
        recent_cleaning=recent_cleaning)
    return timestep, next_state
