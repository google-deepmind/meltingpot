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
import numpy as np


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


@dataclasses.dataclass(frozen=True)
class CorrigibleReciprocatorState:
  """Current state of the CorrigibleReciprocator.

  Attributes:
    clean_until: step_count after which to stop cleaning.
    nice: whether the puppeteer will cooperate.
    step_count: number of timesteps previously seen in this episode.
    recent_cleaning: number of others cleaning on previous timesteps (ordered
      from oldest to most recent).
    sanctioned_steps: how long has the player been in time-out (after zap).
  """
  clean_until: int
  nice: bool
  step_count: int
  recent_cleaning: tuple[int, ...]
  sanctioned_steps: int


@dataclasses.dataclass(frozen=True)
class SanctionerAlternatorState:
  """Current state of the SanctionerAlternator.

  Attributes:
    step_count: number of timesteps previously seen in this episode.
    recent_cleaning: number of others cleaning on previous timesteps (ordered
      from oldest to most recent).
    sanction_until: the number of steps to sanction others.
  """
  step_count: int
  recent_cleaning: tuple[int, ...]
  sanction_until: int


class CorrigibleReciprocator(puppeteer.Puppeteer[CorrigibleReciprocatorState]):
  """Cleanup puppeteer for a corrigible reciprocator."""

  def __init__(
      self,
      cooperate_goal: puppeteer.PuppetGoal,
      defect_goal: puppeteer.PuppetGoal,
      num_others_cooperating_cumulant: str,
      threshold: int,
      recency_window: int = 5,
      steps_to_cooperate_when_motivated: int = 75,
      timeout_steps: int = 50,
      corrigible_threshold: int = 2) -> None:
    """Initializes the puppeteer."""
    self._cooperate_goal = cooperate_goal
    self._defect_goal = defect_goal
    self._num_others_cooperating_cumulant = num_others_cooperating_cumulant

    self._steps_to_cooperate_when_motivated = steps_to_cooperate_when_motivated

    self._timeout_steps = timeout_steps
    self._corrigible_threshold = corrigible_threshold

    self._threshold = threshold
    self._recency_window = recency_window

  def initial_state(self):
    return CorrigibleReciprocatorState(
        clean_until=0, nice=False, step_count=0, recent_cleaning=(),
        sanctioned_steps=0)

  def step(
      self, timestep: dm_env.TimeStep, prev_state: CorrigibleReciprocatorState,
  ) -> tuple[dm_env.TimeStep, CorrigibleReciprocatorState]:
    """Puppeteer step.

    Args:
      timestep: The current timestep from the environment.
      prev_state: The previous state of the puppeteer.

    Returns:
      The transformed timestep (with a goal), and the new state.
    """
    if timestep.first():
      prev_state = self.initial_state()

    sanctioned_steps = prev_state.sanctioned_steps
    clean_until = prev_state.clean_until
    nice = prev_state.nice
    step_count = prev_state.step_count
    recent_cleaning = prev_state.recent_cleaning
    if np.all(timestep.observation['RGB'] == 0):
      sanctioned_steps += 1
      if (sanctioned_steps >=
          self._timeout_steps * self._corrigible_threshold):
        nice = True
        clean_until = step_count + self._steps_to_cooperate_when_motivated
        return (puppeteer.puppet_timestep(timestep, self._cooperate_goal),
                CorrigibleReciprocatorState(clean_until, nice, step_count + 1,
                                            recent_cleaning, sanctioned_steps))
      return (puppeteer.puppet_timestep(timestep, self._defect_goal),
              CorrigibleReciprocatorState(clean_until, nice, step_count + 1,
                                          recent_cleaning, sanctioned_steps))

    if not nice:
      return (puppeteer.puppet_timestep(timestep, self._defect_goal),
              CorrigibleReciprocatorState(clean_until, nice, step_count + 1,
                                          recent_cleaning, sanctioned_steps))

    num_others_cooperating = float(
        timestep.observation[self._num_others_cooperating_cumulant])
    if len(recent_cleaning) < self._recency_window:
      recent_cleaning = tuple([num_others_cooperating, *recent_cleaning])
    else:
      recent_cleaning = tuple([num_others_cooperating, *recent_cleaning[:-1]])

    smooth_cleaning = np.sum(recent_cleaning)
    if smooth_cleaning >= self._threshold:
      clean_until = step_count + self._steps_to_cooperate_when_motivated

    if step_count < clean_until:
      goal = self._cooperate_goal
    else:
      goal = self._defect_goal

    return (puppeteer.puppet_timestep(timestep, goal),
            CorrigibleReciprocatorState(clean_until, nice, step_count + 1,
                                        recent_cleaning, sanctioned_steps))


class SanctionerAlternator(puppeteer.Puppeteer[SanctionerAlternatorState]):
  """Cleanup puppeteer for a thresholded sanctioner alternator."""

  def __init__(
      self,
      cooperate_goal: puppeteer.PuppetGoal,
      defect_goal: puppeteer.PuppetGoal,
      sanction_goal: puppeteer.PuppetGoal,
      num_others_cooperating_cumulant: str,
      threshold: int,
      recency_window: int = 50,
      steps_to_sanction_when_motivated: int = 100,
      alternating_steps: int = 200,
      nice: bool = True) -> None:
    """Initializes the puppeteer.
    
    Args:
      cooperate_goal: the goal to emit when cooperating.
      defect_goal: the goal to emit when defecting.
      sanction_goal: the goal to emit when sanctioning.
      num_others_cooperating_cumulant: name of the observation for others
        cooperating.
      threshold: threshold of others cooperating over time under which it
        triggers a sanctioning burst.
      recency_window: the size of the window to keep track of others
        cooperating. Used in conjunction with the threshold.
      steps_to_sanction_when_motivated: when a sactioning burst happens, it
        lasts this long.
      alternating_steps: change goals between cooperation and defection every
        this number of steps.
      nice: whether it starts with cooperation or defection.
    """
    self._cooperate_goal = cooperate_goal
    self._defect_goal = defect_goal
    self._sanction_goal = sanction_goal
    self._num_others_cooperating_cumulant = num_others_cooperating_cumulant
    self._threshold = threshold
    self._recency_window = recency_window

    self._alternating_steps = alternating_steps
    self._nice = nice

    self._steps_to_sanction_when_motivated = steps_to_sanction_when_motivated

  def initial_state(self) -> SanctionerAlternatorState:
    return SanctionerAlternatorState(
        sanction_until=0,
        recent_cleaning=tuple([1] * self._recency_window),
        step_count=0)

  def step(
      self,
      timestep: dm_env.TimeStep,
      prev_state: SanctionerAlternatorState,
  ) -> tuple[dm_env.TimeStep, SanctionerAlternatorState]:
    """Puppeteer step.

    Args:
      timestep: The current timestep from the environment.
      prev_state: The previous state of the puppeteer.

    Returns:
      The transformed timestep (with a goal), and the new state.
    """
    if timestep.first():
      prev_state = self.initial_state()

    num_others_cooperating = float(
        timestep.observation[self._num_others_cooperating_cumulant])
    recent_cleaning = tuple(
        [1 if num_others_cooperating else 0] +
        list(prev_state.recent_cleaning[:-1]))
    sanction_until = prev_state.sanction_until

    smooth_cleaning = np.sum(recent_cleaning)
    if smooth_cleaning < self._threshold:
      sanction_until = (
          prev_state.step_count + self._steps_to_sanction_when_motivated)

    if prev_state.step_count < sanction_until:
      goal = self._sanction_goal
    else:
      if ((prev_state.step_count // self._alternating_steps) % 2
          ) == 0 and self._nice:
        goal = self._cooperate_goal
      else:
        goal = self._defect_goal

    return puppeteer.puppet_timestep(timestep, goal), SanctionerAlternatorState(
        sanction_until=sanction_until,
        recent_cleaning=recent_cleaning,
        step_count=prev_state.step_count + 1)
