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
"""Puppeteers for puppet bots."""

import abc
from typing import Generic, Mapping, NewType, Sequence, Tuple, TypeVar

import dm_env
import immutabledict
import numpy as np

State = TypeVar('State')
PuppetGoal = NewType('PuppetGoal', np.ndarray)

_GOAL_OBSERVATION_KEY = 'GOAL'
_GOAL_DTYPE = np.int32


class Puppeteer(Generic[State], metaclass=abc.ABCMeta):
  """A puppeteer that controls the timestep forwarded to the puppet.

  Must not possess any mutable state not in `initial_state`.
  """

  @abc.abstractmethod
  def initial_state(self) -> State:
    """Returns the initial state of the puppeteer.

    Must not have any side effects.
    """

  @abc.abstractmethod
  def step(self, timestep: dm_env.TimeStep,
           prev_state: State) -> Tuple[dm_env.TimeStep, State]:
    """Steps the puppeteer.

    Must not have any side effects.

    Args:
      timestep: information from the environment.
      prev_state: the previous state of the puppeteer.

    Returns:
      timestep: the timestep to forward to the puppet.
      next_state: the state for the next step call.
    """


def puppet_timestep(timestep: dm_env.TimeStep,
                    goal: PuppetGoal) -> dm_env.TimeStep:
  """Returns a timestep with a goal observation added."""
  puppet_observation = immutabledict.immutabledict(
      timestep.observation, **{_GOAL_OBSERVATION_KEY: goal})
  return timestep._replace(observation=puppet_observation)


def puppet_goals(names: Sequence[str],
                 dtype: ... = _GOAL_DTYPE) -> Mapping[str, PuppetGoal]:
  """Returns a mapping from goal name to a one-hot goal vector for a puppet.

  Args:
    names: names for each of the corresponding goals.
    dtype: dtype of the one-hot goals to return.
  """
  goals = np.eye(len(names), dtype=dtype)
  goals.setflags(write=False)
  return immutabledict.immutabledict(zip(names, goals))
