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
"""Substrate builder."""

from collections.abc import Collection, Mapping, Sequence
from typing import Any

import chex
import dm_env
from meltingpot.utils.substrates import builder
from meltingpot.utils.substrates.wrappers import base
from meltingpot.utils.substrates.wrappers import collective_reward_wrapper
from meltingpot.utils.substrates.wrappers import discrete_action_wrapper
from meltingpot.utils.substrates.wrappers import multiplayer_wrapper
from meltingpot.utils.substrates.wrappers import observables
from meltingpot.utils.substrates.wrappers import observables_wrapper
import reactivex
from reactivex import subject


@chex.dataclass(frozen=True)
class SubstrateObservables:
  """Observables for a substrate.

  Attributes:
    action: emits actions sent to the substrate from players.
    timestep: emits timesteps sent from the substrate to players.
    events: emits environment-specific events resulting from any interactions
      with the Substrate. Each individual event is emitted as a single element:
      (event_name, event_item).
    dmlab2d: Observables from the underlying dmlab2d environment.
  """
  action: reactivex.Observable[Sequence[int]]
  timestep: reactivex.Observable[dm_env.TimeStep]
  events: reactivex.Observable[tuple[str, Any]]
  dmlab2d: observables.Lab2dObservables


class Substrate(base.Lab2dWrapper):
  """Specific subclass of Wrapper with overridden spec types."""

  def __init__(self, env: observables.ObservableLab2d) -> None:
    """See base class."""
    super().__init__(env)
    self._action_subject = subject.Subject()
    self._timestep_subject = subject.Subject()
    self._events_subject = subject.Subject()
    self._observables = SubstrateObservables(
        action=self._action_subject,
        events=self._events_subject,
        timestep=self._timestep_subject,
        dmlab2d=env.observables(),
    )

  def reset(self) -> dm_env.TimeStep:
    """See base class."""
    timestep = super().reset()
    self._timestep_subject.on_next(timestep)
    for event in super().events():
      self._events_subject.on_next(event)
    return timestep

  def step(self, action: Sequence[int]) -> dm_env.TimeStep:
    """See base class."""
    self._action_subject.on_next(action)
    timestep = super().step(action)
    self._timestep_subject.on_next(timestep)
    for event in super().events():
      self._events_subject.on_next(event)
    return timestep

  def reward_spec(self) -> Sequence[dm_env.specs.Array]:
    """See base class."""
    return self._env.reward_spec()

  def observation_spec(self) -> Sequence[Mapping[str, dm_env.specs.Array]]:
    """See base class."""
    return self._env.observation_spec()

  def action_spec(self) -> Sequence[dm_env.specs.DiscreteArray]:
    """See base class."""
    return self._env.action_spec()

  def close(self) -> None:
    """See base class."""
    super().close()
    self._action_subject.on_completed()
    self._timestep_subject.on_completed()
    self._events_subject.on_completed()

  def observables(self) -> SubstrateObservables:
    """Returns observables for the substrate."""
    return self._observables


def build_substrate(
    *,
    lab2d_settings: builder.Settings,
    individual_observations: Collection[str],
    global_observations: Collection[str],
    action_table: Sequence[Mapping[str, int]],
) -> Substrate:
  """Builds a Melting Pot substrate.

  Args:
    lab2d_settings: the lab2d settings for building the lab2d environment.
    individual_observations: names of the player-specific observations to make
      available to each player.
    global_observations: names of the dmlab2d observations to make available to
      all players.
    action_table: the possible actions. action_table[i] defines the dmlab2d
      action that will be forwarded to the wrapped dmlab2d environment for the
      discrete Melting Pot action i.

  Returns:
    The constructed substrate.
  """
  env = builder.builder(lab2d_settings)
  env = observables_wrapper.ObservablesWrapper(env)
  env = multiplayer_wrapper.Wrapper(
      env,
      individual_observation_names=individual_observations,
      global_observation_names=global_observations)
  env = discrete_action_wrapper.Wrapper(env, action_table=action_table)
  # Add a wrapper that augments adds an observation of the collective
  # reward (sum of all players' rewards).
  env = collective_reward_wrapper.CollectiveRewardWrapper(env)
  return Substrate(env)
