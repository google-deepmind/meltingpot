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

from typing import Any, Mapping, Sequence, Tuple

import chex
import dm_env
from ml_collections import config_dict
import rx
from rx import subject

from meltingpot.python.configs import substrates as substrate_configs
from meltingpot.python.utils.substrates import builder
from meltingpot.python.utils.substrates.wrappers import base
from meltingpot.python.utils.substrates.wrappers import discrete_action_wrapper
from meltingpot.python.utils.substrates.wrappers import multiplayer_wrapper

AVAILABLE_SUBSTRATES = substrate_configs.SUBSTRATES


@chex.dataclass(frozen=True)
class SubstrateObservables:
  """Observables for a substrate.

  Attributes:
    action: emits actions sent to the substrate from players.
    timestep: emits timesteps sent from the substrate to players.
    events: emits environment-specific events resulting from any interactions
      with the Substrate. Each individual event is emitted as a single element:
      (event_name, event_item).
  """
  action: rx.typing.Observable[Sequence[int]]
  timestep: rx.typing.Observable[dm_env.TimeStep]
  events: rx.typing.Observable[Tuple[str, Any]]


class Substrate(base.Wrapper):
  """Specific subclass of Wrapper with overridden spec types."""

  def __init__(self, env: ...) -> None:
    """See base class."""
    super().__init__(env)
    self._action_subject = subject.Subject()
    self._timestep_subject = subject.Subject()
    self._events_subject = subject.Subject()
    self._observables = SubstrateObservables(
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


def get_config(substrate_name: str) -> config_dict.ConfigDict:
  """Returns the configs for the substrate.

  Args:
    substrate_name: name of the substrate. Must be in AVAILABLE_SUBSTRATES.
  """
  if substrate_name not in AVAILABLE_SUBSTRATES:
    raise ValueError(f'Unknown substrate {substrate_name!r}.')
  return substrate_configs.get_config(substrate_name).lock()


def build(config: config_dict.ConfigDict) -> Substrate:
  """Builds the substrate given the config.

  Args:
    config: config resulting from `get_config`.

  Returns:
    The training substrate.
  """
  env = builder.builder(**config)
  env = multiplayer_wrapper.Wrapper(
      env,
      individual_observation_names=config.individual_observation_names,
      global_observation_names=config.global_observation_names)
  env = discrete_action_wrapper.Wrapper(env, action_table=config.action_set)
  return Substrate(env)
