# Copyright 2024 DeepMind Technologies Limited.
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
"""Subject that emits the player returns at the end of each episode."""

import dm_env
import numpy as np
from reactivex import subject


class ReturnSubject(subject.Subject):
  """Subject that emits the player returns at the end of each episode."""

  _return: np.ndarray | None = None

  def on_next(self, timestep: dm_env.TimeStep) -> None:
    """Called on each timestep.

    Args:
      timestep: the most recent timestep.
    """
    if timestep.step_type.first():
      self._return = np.zeros_like(timestep.reward)
    elif self._return is None:
      raise ValueError('First timestep must be StepType.FIRST.')
    self._return += timestep.reward
    if timestep.step_type.last():
      super().on_next(self._return)
      self._return = None
