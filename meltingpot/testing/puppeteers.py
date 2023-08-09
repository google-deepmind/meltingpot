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
"""Puppeteer test utilities."""

from typing import Any, Iterable, Iterator, Mapping, Optional, Sequence, TypeVar

import dm_env
from meltingpot.utils.puppeteers import puppeteer as puppeteer_lib

GOAL_KEY = puppeteer_lib._GOAL_OBSERVATION_KEY  # pylint: disable=protected-access
State = TypeVar('State')


def step_many(
    puppeteer: puppeteer_lib.Puppeteer[State],
    timesteps: Iterable[dm_env.TimeStep],
    state: Optional[State] = None,
) -> Iterator[tuple[dm_env.TimeStep, State]]:
  """Yields multiple puppeteeer steps."""
  if state is None:
    state = puppeteer.initial_state()
  for timestep in timesteps:
    transformed_timestep, state = puppeteer.step(timestep, state)
    yield transformed_timestep, state


def goals_from_timesteps(
    puppeteer: puppeteer_lib.Puppeteer[State],
    timesteps: Iterable[dm_env.TimeStep],
    state: Optional[State] = None,
) -> tuple[Sequence[puppeteer_lib.PuppetGoal], State]:
  """Returns puppet goals for each timestep."""
  goals = []
  for timestep, state in step_many(puppeteer, timesteps, state):
    goals.append(timestep.observation[GOAL_KEY])
  return goals, state


def episode_timesteps(
    observations: Sequence[Mapping[str, Any]]) -> Iterator[dm_env.TimeStep]:
  """Yields an episode timestep for each observation."""
  for n, observation in enumerate(observations):
    if n == 0:
      yield dm_env.restart(observation=observation)
    elif n == len(observations) - 1:
      yield dm_env.termination(observation=observation, reward=0)
    else:
      yield dm_env.transition(observation=observation, reward=0)


def goals_from_observations(
    puppeteer: puppeteer_lib.Puppeteer[State],
    observations: Sequence[Mapping[str, Any]],
    state: Optional[State] = None,
) -> tuple[Sequence[puppeteer_lib.PuppetGoal], State]:
  """Returns puppet goals from an episode of the provided observations."""
  timesteps = episode_timesteps(observations)
  return goals_from_timesteps(puppeteer, timesteps, state)
