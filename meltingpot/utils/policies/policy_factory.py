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
"""Factory for constructing policies."""

import abc
from typing import Callable

import dm_env
from meltingpot.utils.policies import policy


class PolicyFactory(metaclass=abc.ABCMeta):
  """Factory for producing instances of a specific policy."""

  def __init__(
      self,
      *,
      timestep_spec: dm_env.TimeStep,
      action_spec: dm_env.specs.DiscreteArray,
      builder: Callable[[], policy.Policy],
  ) -> None:
    """Initializes the object.

    Args:
      timestep_spec: spec of the timestep expected by the policy.
      action_spec: spec of the action returned by the policy.
      builder: callable that builds the policy.
    """
    self._timestep_spec = timestep_spec
    self._action_spec = action_spec
    self._builder = builder

  def timestep_spec(self) -> dm_env.TimeStep:
    """Returns spec of the timestep expected by the policy."""
    return self._timestep_spec

  def action_spec(self) -> dm_env.specs.DiscreteArray:
    """Returns spec of the action returned by the policy."""
    return self._action_spec

  def build(self) -> policy.Policy:
    """Returns a policy for the bot."""
    return self._builder()
