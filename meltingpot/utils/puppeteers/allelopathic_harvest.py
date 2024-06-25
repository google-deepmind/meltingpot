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
"""Puppeteers for allelopathic_harvest."""

import dataclasses
from typing import Sequence

import dm_env
from meltingpot.utils.puppeteers import puppeteer
import numpy as np


@dataclasses.dataclass(frozen=True)
class ConventionFollowerState:
  """Current state of the ConventionFollower.

  Attributes:
    step_count: number of timesteps previously seen in this episode.
    current_goal: the goal last used by the puppeteer.
    recent_frames: buffer of recent observation frames.
  """
  step_count: int
  current_goal: puppeteer.PuppetGoal
  recent_frames: tuple[np.ndarray, ...]


class ConventionFollower(puppeteer.Puppeteer[ConventionFollowerState]):
  """Allelopathic Harvest puppeteer for a convention follower."""

  def __init__(
      self,
      initial_goal: puppeteer.PuppetGoal,
      preference_goals: Sequence[puppeteer.PuppetGoal],
      color_threshold: int,
      recency_window: int = 5) -> None:
    """Initializes the puppeteer.
    
    Args:
      initial_goal: the initial goal to pursue.
      preference_goals: sequence of goals corresponding to the R, G, B, goals
        for when that color becomes dominant (on average) over the last
        `recency_window` frames.
      color_threshold: threshold for a color to become dominant.
      recency_window: number of frames to check for a dominant color.
    """
    self._initial_goal = initial_goal
    self._preference_goals = preference_goals
    self._color_threshold = color_threshold
    self._recency_window = recency_window

  def initial_state(self) -> ConventionFollowerState:
    return ConventionFollowerState(
        step_count=0, current_goal=self._initial_goal, recent_frames=())

  def step(
      self,
      timestep: dm_env.TimeStep,
      prev_state: ConventionFollowerState
  ) -> tuple[dm_env.TimeStep, ConventionFollowerState]:
    """Puppeteer step.

    Args:
      timestep: the timestep.
      prev_state: the state of the pupeeteer.

    Returns:
      Modified timestep and new state.
    """
    if timestep.first():
      prev_state = self.initial_state()

    recent_frames = list(prev_state.recent_frames)
    current_goal = prev_state.current_goal
    if len(recent_frames) < self._recency_window:
      recent_frames = tuple([timestep.observation['RGB']] + recent_frames)
    else:
      recent_frames = tuple([timestep.observation['RGB']] + recent_frames[:-1])

    average_color = np.array(recent_frames).mean(axis=(0, 1, 2))
    index = np.argmax(average_color)
    if average_color[index] > self._color_threshold:
      current_goal = self._preference_goals[index]

    return puppeteer.puppet_timestep(timestep, current_goal), (
        ConventionFollowerState(
            step_count=prev_state.step_count + 1,
            current_goal=current_goal,
            recent_frames=recent_frames))
