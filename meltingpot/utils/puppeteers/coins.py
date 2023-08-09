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
"""Puppeteers for coins."""

import dataclasses

import dm_env
from meltingpot.utils.puppeteers import puppeteer


@dataclasses.dataclass(frozen=True)
class ReciprocatorState:
  """Current state of the Reciprocator.

  Attributes:
    step_count: number of timesteps previously seen in this episode.
    spite_until: earliest step_count after which to stop being spiteful.
    defect_until: earliest step_count after which to stop defecting.
    recent_defection: level of defection on previous timesteps (ordered from
      oldest to most recent).
  """
  step_count: int
  spite_until: int
  defect_until: int
  recent_defection: tuple[int, ...]


class Reciprocator(puppeteer.Puppeteer[ReciprocatorState]):
  """Puppeteer for a reciprocating agent.

  This puppeteer's behavior depends on the behavior of others. In particular, it
  tracks the total amount of others' defection, and integrates this signal
  using a rolling window.

  Initially, the puppet will be in a cooperation mode where it will direct the
  puppet to cooperate with others. However, once the total level of
  defection reaches threshold, the puppeteer will switch to a defection
  routine. This routine starts with some amount of spite, then plain defection.
  Once the routine is complete, the puppeteer will return to the cooperative
  mode.

  At any point, if the integrated level of defection again reaches threshold,
  the defection routine will be triggered again from the beginning.
  """

  def __init__(
      self,
      *,
      cooperate_goal: puppeteer.PuppetGoal,
      defect_goal: puppeteer.PuppetGoal,
      spite_goal: puppeteer.PuppetGoal,
      partner_defection_signal: str,
      recency_window: int,
      threshold: int,
      frames_to_punish: int,
      spiteful_punishment_window: int,
  ) -> None:
    """Initializes the puppeteer.

    Args:
      cooperate_goal: goal to emit to puppet when "cooperating".
      defect_goal: goal to emit to puppet when "defecting".
      spite_goal: goal to emit to puppet when being "spiteful".
      partner_defection_signal: key in observations that provides the level of
        partner defection in the previous timestep.
      recency_window: number of steps over which to remember others' behavior.
      threshold: if the total number of (nonunique) cooperators over the
        remembered period reaches this threshold, the puppeteer will direct the
        puppet to cooperate.
      frames_to_punish: the number of steps to not cooperate for when triggered
        by others' behavior.
      spiteful_punishment_window: the number of steps to bne spiteful for when
        triggered by others' behavior.
    """
    self._cooperate_goal = cooperate_goal
    self._defect_goal = defect_goal
    self._spite_goal = spite_goal
    self._partner_defection_signal = partner_defection_signal

    if threshold > 0:
      self._threshold = threshold
    else:
      raise ValueError('threshold must be positive')

    if recency_window > 0:
      self._recency_window = recency_window
    else:
      raise ValueError('recency_window must be positive')

    if frames_to_punish > 0:
      self._frames_to_punish = frames_to_punish
    else:
      raise ValueError('frames_to_punish must be positive.')

    if 0 <= spiteful_punishment_window <= frames_to_punish:
      self._spiteful_punishment_window = spiteful_punishment_window
    else:
      raise ValueError('spiteful_punishment_window must nonegative and lower '
                       'than frames_to_punish')

  def initial_state(self) -> ReciprocatorState:
    """See base class."""
    return ReciprocatorState(
        step_count=0, spite_until=0, defect_until=0, recent_defection=())

  def step(
      self, timestep: dm_env.TimeStep, prev_state: ReciprocatorState
  ) -> tuple[dm_env.TimeStep, ReciprocatorState]:
    """See base class."""
    if timestep.first():
      prev_state = self.initial_state()
    step_count = prev_state.step_count
    spite_until = prev_state.spite_until
    defect_until = prev_state.defect_until
    recent_defection = prev_state.recent_defection

    partner_defection = int(
        timestep.observation[self._partner_defection_signal])
    recent_defection += (partner_defection,)
    recent_defection = recent_defection[-self._recency_window:]

    total_recent_defection = sum(recent_defection)
    if total_recent_defection >= self._threshold:
      spite_until = step_count + self._spiteful_punishment_window
      defect_until = step_count + self._frames_to_punish
      recent_defection = ()

    if step_count < spite_until:
      goal = self._spite_goal
    elif step_count < defect_until:
      goal = self._defect_goal
    else:
      goal = self._cooperate_goal
    timestep = puppeteer.puppet_timestep(timestep, goal)

    next_state = ReciprocatorState(
        step_count=step_count + 1,
        spite_until=spite_until,
        defect_until=defect_until,
        recent_defection=recent_defection)
    return timestep, next_state
