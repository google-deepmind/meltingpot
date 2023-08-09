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
import threading
from typing import Callable, Collection, List, Mapping, Sequence

import chex
import dm_env
from meltingpot.utils.policies import policy as policy_lib
import reactivex
from reactivex import subject


def _step_fn(policy: policy_lib.Policy,
             lock: threading.Lock) -> Callable[[dm_env.TimeStep], int]:
  """Threadsafe stateful step function where the state is encapsulated.

  Args:
    policy: the underlying policy to use.
    lock: a lock that controls access to the policy.

  Returns:
    A step function that returns an action in response to a timestep.
  """
  with lock:
    state = policy.initial_state()

  def step(timestep: dm_env.TimeStep) -> int:
    nonlocal state
    with lock:
      action, state = policy.step(timestep=timestep, prev_state=state)
    return action

  return step


@chex.dataclass(frozen=True)  # works with tree.
class PopulationObservables:
  """Observables for a population.

  Attributes:
    names: emits the names of the sampled population on a reset.
    action: emits actions sent to the substrate by the poulation.
    timestep: emits timesteps sent from the substrate to the population.
  """
  names: reactivex.Observable[Sequence[str]]
  action: reactivex.Observable[Sequence[int]]
  timestep: reactivex.Observable[dm_env.TimeStep]


class Population:
  """A population of policies to use in a scenario."""

  def __init__(
      self,
      *,
      policies: Mapping[str, policy_lib.Policy],
      names_by_role: Mapping[str, Collection[str]],
      roles: Sequence[str]) -> None:
    """Initializes the population.

    Args:
      policies: the policies to sample from (with replacement) each episode.
        Will be closed when the Population is closed.
      names_by_role: dict mapping role to bot names that can fill it.
      roles: specifies which role should fill the corresponding player slot.
    """
    self._policies = dict(policies)
    self._names_by_role = {
        role: tuple(set(names)) for role, names in names_by_role.items()}
    self._roles = tuple(roles)

    self._locks = {name: threading.Lock() for name in self._policies}
    self._executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=len(roles))
    self._step_fns: List[Callable[[dm_env.TimeStep], int]] = []
    self._action_futures: List[concurrent.futures.Future[int]] = []

    self._names_subject = subject.Subject()
    self._action_subject = subject.Subject()
    self._timestep_subject = subject.Subject()
    self._observables = PopulationObservables(  # pylint: disable=unexpected-keyword-arg
        names=self._names_subject,
        action=self._action_subject,
        timestep=self._timestep_subject,
    )

  def close(self):
    """Closes the population."""
    for future in self._action_futures:
      future.cancel()
    self._executor.shutdown(wait=False)
    for policy in self._policies.values():
      policy.close()
    self._names_subject.on_completed()
    self._action_subject.on_completed()
    self._timestep_subject.on_completed()

  def _sample_names(self) -> Sequence[str]:
    """Returns a sample of policy names for the population."""
    return [random.choice(self._names_by_role[role]) for role in self._roles]

  def reset(self) -> None:
    """Resamples the population."""
    names = self._sample_names()
    self._names_subject.on_next(names)
    self._step_fns = [
        _step_fn(policy=self._policies[name], lock=self._locks[name])
        for name in names
    ]
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
    self._timestep_subject.on_next(timestep)
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
    self._action_subject.on_next(actions)
    return actions

  def observables(self) -> PopulationObservables:
    """Returns the observables for the population."""
    return self._observables
