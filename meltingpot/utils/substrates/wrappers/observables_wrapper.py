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
"""Wrapper that exposes Lab2d timesteps, actions, and events as observables."""

from typing import Mapping, Union

import dm_env
import dmlab2d
from meltingpot.utils.substrates.wrappers import observables
import numpy as np
from reactivex import subject

Action = Union[int, float, np.ndarray]


class ObservablesWrapper(observables.ObservableLab2dWrapper):
  """Wrapper exposes timesteps, actions, and events as observables."""

  def __init__(self, env: dmlab2d.Environment):
    """Initializes the object.

    Args:
      env: The environment to wrap.
    """
    super().__init__(env)
    self._action_subject = subject.Subject()
    self._timestep_subject = subject.Subject()
    self._events_subject = subject.Subject()
    self._observables = observables.Lab2dObservables(
        action=self._action_subject,
        events=self._events_subject,
        timestep=self._timestep_subject,
    )

  def reset(self) -> dm_env.TimeStep:
    """See base class."""
    timestep = super().reset()
    self._timestep_subject.on_next(timestep)
    for event in super().events():
      self._events_subject.on_next(event)
    return timestep

  def step(self, action: Mapping[str, Action]) -> dm_env.TimeStep:
    """See base class."""
    self._action_subject.on_next(action)
    timestep = super().step(action)
    self._timestep_subject.on_next(timestep)
    for event in super().events():
      self._events_subject.on_next(event)
    return timestep

  def close(self) -> None:
    """See base class."""
    super().close()
    self._action_subject.on_completed()
    self._timestep_subject.on_completed()
    self._events_subject.on_completed()

  def observables(self) -> observables.Lab2dObservables:
    """See base class."""
    return self._observables
