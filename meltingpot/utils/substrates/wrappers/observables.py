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
"""Base class for wrappers.

Wrappers are assumed to own the wrapped environment and that they have the
**only** reference to it. This means that they will:

1.   Close the environment when they close.
2.   Modify the environment specs and timesteps inplace.
"""

import abc
from typing import Any, Sequence

import chex
import dm_env
import dmlab2d
from meltingpot.utils.substrates.wrappers import base
import reactivex


@chex.dataclass(frozen=True)
class Lab2dObservables:
  """Observables for a Lab2D environment.

  Attributes:
    action: emits actions sent to the substrate from players.
    timestep: emits timesteps sent from the substrate to players.
    events: emits environment-specific events resulting from any interactions
      with the Substrate. Each individual event is emitted as a single element:
      (event_name, event_item).
  """
  action: reactivex.Observable[Sequence[int]]
  timestep: reactivex.Observable[dm_env.TimeStep]
  events: reactivex.Observable[tuple[str, Any]]


class ObservableLab2d(dmlab2d.Environment):
  """A DM Lab2D environment which is observable."""

  @abc.abstractmethod
  def observables(self) -> Lab2dObservables:
    """The observables of the Lab2D environment."""


class ObservableLab2dWrapper(base.Lab2dWrapper, ObservableLab2d):
  """Base class for wrappers of ObservableLab2d."""

  def observables(self, *args, **kwargs) -> ...:
    """See base class."""
    return self._env.observables(*args, **kwargs)
