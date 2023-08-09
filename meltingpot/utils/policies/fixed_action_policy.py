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
"""Policy that always returns a fixed action."""

from typing import Tuple

import dm_env
from meltingpot.utils.policies import policy


class FixedActionPolicy(policy.Policy[Tuple[()]]):
  """Always performs the same action, regardless of observations."""

  def __init__(self, action: int):
    """Initializes the policy.

    Args:
      action: The action that that the policy will always emit, regardless of
        its observations.
    """
    self._action = action

  def step(self, timestep: dm_env.TimeStep,
           prev_state: Tuple[()]) -> Tuple[int, Tuple[()]]:
    """See base class."""
    return self._action, prev_state

  def initial_state(self) -> Tuple[()]:
    """See base class."""
    return ()

  def close(self) -> None:
    """See base class."""
