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
"""Scenario class."""

from typing import Any, Collection, Iterable, Mapping, Sequence, Tuple, TypeVar

import chex
import dm_env
import immutabledict
import numpy as np
from rx import subject

from meltingpot.python.utils.scenarios import population
from meltingpot.python.utils.scenarios.wrappers import base
from meltingpot.python.utils.substrates import substrate as substrate_lib

T = TypeVar('T')


def _restrict_observation(
    observation: Mapping[str, T],
    permitted_observations: Collection[str],
) -> Mapping[str, T]:
  """Restricts an observation to only the permitted keys."""
  return immutabledict.immutabledict({
      key: observation[key]
      for key in observation if key in permitted_observations
  })


def _restrict_observations(
    observations: Iterable[Mapping[str, T]],
    permitted_observations: Collection[str],
) -> Sequence[Mapping[str, T]]:
  """Restricts multiple observations to only the permitted keys."""
  return tuple(
      _restrict_observation(observation, permitted_observations)
      for observation in observations
  )


def _partition(
    values: Sequence[T],
    is_focal: Sequence[bool],
) -> Tuple[Sequence[T], Sequence[T]]:
  """Partitions a sequence into focal and background sequences."""
  focal_values = []
  background_values = []
  for focal, value in zip(is_focal, values):
    if focal:
      focal_values.append(value)
    else:
      background_values.append(value)
  return tuple(focal_values), tuple(background_values)


def _merge(
    focal_values: Sequence[T],
    background_values: Sequence[T],
    is_focal: Sequence[bool],
) -> Sequence[T]:
  """Merges focal and background sequences into one."""
  focal_values = iter(focal_values)
  background_values = iter(background_values)
  return tuple(
      next(focal_values if focal else background_values) for focal in is_focal
  )


@chex.dataclass(frozen=True)  # works with tree.
class ScenarioObservables(substrate_lib.SubstrateObservables):
  """Observables for a Scenario.

  Attributes:
    action: emits actions sent to the scenario from (focal) players.
    timestep: emits timesteps sent from the scenario to (focal) players.
    events: will never emit any events since things like player index are hard
      to interpret for a Scenario. Use substrate.events instead.
    background: observables from the perspective of the background players.
    substrate: observables for the underlying substrate.
  """
  background: population.PopulationObservables
  substrate: substrate_lib.SubstrateObservables


class Scenario(base.SubstrateWrapper):
  """An substrate where a number of player slots are filled by bots."""

  def __init__(
      self,
      substrate: substrate_lib.Substrate,
      background_population: population.Population,
      is_focal: Sequence[bool],
      permitted_observations: Collection[str]) -> None:
    """Initializes the scenario.

    Args:
      substrate: the substrate to add bots to. Will be closed with the scenario.
      background_population: the background population to sample bots from. Will
        be closed with the scenario.
      is_focal: which player slots are allocated to focal players.
      permitted_observations: the substrate observation keys permitted to be
        exposed by the scenario to focal agents.
    """
    num_players = len(substrate.action_spec())
    if len(is_focal) != num_players:
      raise ValueError(f'is_focal is length {len(is_focal)} but substrate is '
                       f'{num_players}-player.')

    super().__init__(substrate)
    self._background_population = background_population
    self._is_focal = is_focal
    self._permitted_observations = frozenset(permitted_observations)

    self._focal_action_subject = subject.Subject()
    self._focal_timestep_subject = subject.Subject()
    self._background_action_subject = subject.Subject()
    self._background_timestep_subject = subject.Subject()
    self._events_subject = subject.Subject()
    self._observables = ScenarioObservables(  # pylint: disable=unexpected-keyword-arg
        action=self._focal_action_subject,
        events=self._events_subject,
        timestep=self._focal_timestep_subject,
        background=self._background_population.observables(),
        substrate=super().observables(),
    )

  def close(self) -> None:
    """See base class."""
    self._background_population.close()
    super().close()
    self._focal_action_subject.on_completed()
    self._focal_timestep_subject.on_completed()
    self._events_subject.on_completed()

  def _await_full_action(self, focal_action: Sequence[int]) -> Sequence[int]:
    """Returns full action after awaiting bot actions."""
    self._focal_action_subject.on_next(focal_action)
    background_action = self._background_population.await_action()
    return _merge(focal_action, background_action, self._is_focal)

  def _split_timestep(
      self, timestep: dm_env.TimeStep
  ) -> Tuple[dm_env.TimeStep, dm_env.TimeStep]:
    """Splits multiplayer timestep as needed by agents and bots."""
    focal_rewards, background_rewards = _partition(timestep.reward,
                                                   self._is_focal)
    focal_observations, background_observations = _partition(
        timestep.observation, self._is_focal)
    focal_observations = _restrict_observations(focal_observations,
                                                self._permitted_observations)
    focal_timestep = timestep._replace(
        reward=focal_rewards, observation=focal_observations)
    background_timestep = timestep._replace(
        reward=background_rewards, observation=background_observations)
    return focal_timestep, background_timestep

  def _send_full_timestep(self, timestep: dm_env.TimeStep) -> dm_env.TimeStep:
    """Returns focal timestep and sends background timestep to bots."""
    focal_timestep, background_timestep = self._split_timestep(timestep)
    self._background_population.send_timestep(background_timestep)
    self._focal_timestep_subject.on_next(focal_timestep)
    return focal_timestep

  def reset(self) -> dm_env.TimeStep:
    """See base class."""
    self._background_population.reset()
    timestep = super().reset()
    focal_timestep = self._send_full_timestep(timestep)
    for event in self.events():
      self._events_subject.on_next(event)
    return focal_timestep

  def step(self, action: Sequence[int]) -> dm_env.TimeStep:
    """See base class."""
    action = self._await_full_action(focal_action=action)
    timestep = super().step(action)
    focal_timestep = self._send_full_timestep(timestep)
    for event in self.events():
      self._events_subject.on_next(event)
    return focal_timestep

  def observation(self) -> Sequence[Mapping[str, np.ndarray]]:
    observations = super().observation()
    focal_observations, _ = _partition(observations, self._is_focal)
    focal_observations = _restrict_observations(focal_observations,
                                                self._permitted_observations)
    return focal_observations

  def events(self) -> Sequence[Tuple[str, Any]]:
    """See base class."""
    # Do not emit substrate events as these may not make sense in the context
    # of a scenario (e.g. player indices may have changed).
    return ()

  def action_spec(self) -> Sequence[dm_env.specs.DiscreteArray]:
    """See base class."""
    action_spec = super().action_spec()
    focal_action_spec, _ = _partition(action_spec, self._is_focal)
    return focal_action_spec

  def observation_spec(self) -> Sequence[Mapping[str, dm_env.specs.Array]]:
    """See base class."""
    observation_spec = super().observation_spec()
    focal_observation_spec, _ = _partition(observation_spec, self._is_focal)
    return _restrict_observations(focal_observation_spec,
                                  self._permitted_observations)

  def reward_spec(self) -> Sequence[dm_env.specs.Array]:
    """See base class."""
    reward_spec = super().reward_spec()
    focal_reward_spec, _ = _partition(reward_spec, self._is_focal)
    return focal_reward_spec

  def observables(self) -> ScenarioObservables:
    """Returns the observables for the scenario."""
    return self._observables
