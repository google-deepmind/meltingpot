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
"""Bot policy implementations."""

import abc
from typing import Generic, Tuple, TypeVar

import dm_env

State = TypeVar('State')


class Policy(Generic[State], metaclass=abc.ABCMeta):
  """Abstract base class for a policy.

  Must not possess any mutable state not in `initial_state`.
  """

  @abc.abstractmethod
  def initial_state(self) -> State:
    """Returns the initial state of the agent.

    Must not have any side effects.
    """
    raise NotImplementedError()

  @abc.abstractmethod
  def step(self, timestep: dm_env.TimeStep,
           prev_state: State) -> Tuple[int, State]:
    """Steps the agent.

    Must not have any side effects.

    Args:
      timestep: information from the environment
      prev_state: the previous state of the agent.

    Returns:
      action: the action to send to the environment.
      next_state: the state for the next step call.
    """
    raise NotImplementedError()

  @abc.abstractmethod
  def close(self) -> None:
    """Closes the policy."""
    raise NotImplementedError()

  def __enter__(self):
    return self

  def __exit__(self, *args, **kwargs):
    del args, kwargs
    self.close()
