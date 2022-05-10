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
"""Scenario factory."""

import concurrent
import random
from typing import Callable, List, Mapping, Sequence

import chex
import dm_env
import rx

from meltingpot.python import bot as bot_factory


def _step_fn(policy: bot_factory.Policy) -> Callable[[dm_env.TimeStep], int]:
  """Returns a stateful step function where the state is encapsulated.

  Args:
    policy: the underlying policy to use.

  Returns:
    A step function that returns an action in response to a timestep.
  """
  state = policy.initial_state()

  def step(timestep: dm_env.TimeStep) -> int:
    nonlocal state
    action, state = policy.step(timestep=timestep, prev_state=state)
    return action

  return step


class Population:
  """A population of policies to use in a scenario."""

  def __init__(self, policies: Mapping[str, bot_factory.Policy],
               population_size: int) -> None:
    """Initializes the population.

    Args:
      policies: the policies to sample from (with replacement) each episode.
      population_size: the number of policies to sample on each reset.
    """
    self._policies = dict(policies)
    self._population_size = population_size
    self._executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=self._population_size)
    self._step_fns: List[Callable[[dm_env.TimeStep], int]] = []
    self._action_futures: List[concurrent.futures.Future[int]] = []

  def close(self):
    """Closes the population."""
    for future in self._action_futures:
      future.cancel()
    self._executor.shutdown(wait=False)
    for policy in self._policies.values():
      policy.close()

  def _sample_names(self) -> Sequence[str]:
    """Returns a sample of policy names for the population."""
    return random.choices(tuple(self._policies), k=self._population_size)

  def reset(self) -> None:
    """Resamples the population."""
    names = self._sample_names()
    self._step_fns = [_step_fn(self._policies[name]) for name in names]
    for future in self._action_futures:
      future.cancel()
    self._action_futures.clear()

  def send_timestep(self, timestep: dm_env.TimeStep) -> None:
    """Sends timestep to population for asynchronous processing.

    Args:
      timestep: The substrate timestep for the population.

    Raises:
      RuntimeError: previous action has not been awaited.
    """
    if self._action_futures:
      raise RuntimeError('Previous action not retrieved.')
    for n, step_fn in enumerate(self._step_fns):
      bot_timestep = timestep._replace(
          observation=timestep.observation[n], reward=timestep.reward[n])
      future = self._executor.submit(step_fn, bot_timestep)
      self._action_futures.append(future)

  def await_action(self) -> Sequence[int]:
    """Waits for the population action in response to last timestep.

    Returns:
      The action for the population.

    Raises:
      RuntimeError: no timestep has been sent.
    """
    if not self._action_futures:
      raise RuntimeError('No timestep sent.')
    actions = tuple(future.result() for future in self._action_futures)
    self._action_futures.clear()
    return actions


@chex.dataclass(frozen=True)
class PopulationObservables:
  """Observables for a population.

  Attributes:
    action: emits actions sent to the substrate by the poulation.
    timestep: emits timesteps sent from the substrate to the population.
  """
  action: rx.typing.Observable[Sequence[int]]
  timestep: rx.typing.Observable[dm_env.TimeStep]
