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
"""Wrapper that converts action dictionary to a one hot vector."""

import functools
from typing import Mapping, Sequence, TypeVar, Union

import dm_env
import immutabledict
from meltingpot.utils.substrates.wrappers import observables
import numpy as np

T = TypeVar('T')
Numeric = Union[int, float, np.ndarray]


def _validate_action(
    action: Mapping[str, np.ndarray],
    action_spec: Mapping[str, dm_env.specs.Array]) -> None:
  """Raises ValueError if action does not matches the action_spec."""
  if set(action) != set(action_spec):
    raise ValueError('Keys do not match.')
  for key, spec in action_spec.items():
    spec.validate(action[key])


def _validate_action_table(
    action_table: Sequence[Mapping[str, np.ndarray]],
    action_spec: Mapping[str, dm_env.specs.Array]) -> None:
  """Raises ValueError if action_table does not matches the action_spec."""
  if not action_table:
    raise ValueError('action_table must not be empty')
  for action_index, action in enumerate(action_table):
    try:
      _validate_action(action, action_spec)
    except ValueError:
      raise ValueError(f'Action {action_index} ({action}) does not match '
                       f'action_spec ({action_spec}).') from None


def _immutable_action(
    action: Mapping[str, Numeric],
    action_spec: Mapping[str, dm_env.specs.Array],
) -> Mapping[str, np.ndarray]:
  """Returns an immutable action."""
  new_action = {}
  for key, value in action.items():
    if isinstance(value, np.ndarray):
      value = np.copy(value)
    else:
      value = np.array(value, dtype=action_spec[key].dtype)
    value.flags.writeable = False
    new_action[key] = value
  return immutabledict.immutabledict(new_action)


def _immutable_action_table(
    action_table: Sequence[Mapping[str, Numeric]],
    action_spec: Mapping[str, dm_env.specs.Array],
) -> Sequence[Mapping[str, np.ndarray]]:
  """Returns an immutable action table."""
  return tuple(
      _immutable_action(action, action_spec) for action in action_table)


class Wrapper(observables.ObservableLab2dWrapper):
  """Wrapper that maps a discrete action to an entry in an a table."""

  def __init__(self, env, action_table: Sequence[Mapping[str, Numeric]]):
    """Constructor.

    Args:
      env: environment to wrap. When the adaptor closes env will also be closed.
        Note that each player must have the same action spec.
      action_table: Actions that are permissable. The same action lookup is
        used by each player. action_table[i] defines the action that will be
        forwarded to the wrapped environment for discrete action i.
    """
    action_spec = env.action_spec()
    if any(action_spec[0] != spec for spec in action_spec[1:]):
      raise ValueError('Environment has heterogeneous action specs.')
    super().__init__(env)
    self._action_table = _immutable_action_table(action_table, action_spec[0])
    _validate_action_table(self._action_table, action_spec[0])

  def step(self, action: Sequence[int]):
    """See base class."""
    action = [self._action_table[player_action] for player_action in action]
    return super().step(action)

  @functools.lru_cache(maxsize=1)
  def action_spec(self) -> Sequence[dm_env.specs.DiscreteArray]:
    """See base class."""
    spec = dm_env.specs.DiscreteArray(
        num_values=len(self._action_table),
        dtype=np.int64,
        name='action')
    return tuple(spec for _ in super().action_spec())
