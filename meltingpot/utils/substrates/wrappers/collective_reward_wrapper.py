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
"""Wrapper that adds the sum of all players' rewards to observations."""

import copy
from typing import Mapping, Sequence, TypeVar

import dm_env
from meltingpot.utils.substrates.wrappers import observables
import numpy as np

T = TypeVar("T")

_COLLECTIVE_REWARD_OBS = "COLLECTIVE_REWARD"


class CollectiveRewardWrapper(observables.ObservableLab2dWrapper):
  """Wrapper that adds an observation of the sum of all players' rewards."""

  def __init__(self, env):
    """Initializes the object.

    Args:
      env: environment to wrap.
    """
    self._env = env

  def _get_timestep(self, input_timestep: dm_env.TimeStep) -> dm_env.TimeStep:
    """Returns timestep augmented with `collective_reward'.

    Args:
      input_timestep: input_timestep before adding `collective_reward'.
    """
    return dm_env.TimeStep(
        step_type=input_timestep.step_type,
        reward=input_timestep.reward,
        discount=input_timestep.discount,
        observation=[{_COLLECTIVE_REWARD_OBS: np.sum(input_timestep.reward),
                      **obs} for obs in input_timestep.observation])

  def reset(self, *args, **kwargs) -> dm_env.TimeStep:
    """See base class."""
    timestep = super().reset()
    return self._get_timestep(timestep)

  def step(
      self, actions: Sequence[Mapping[str, np.ndarray]]) -> dm_env.TimeStep:
    """See base class."""
    timestep = super().step(actions)
    return self._get_timestep(timestep)

  def observation_spec(self) -> Sequence[Mapping[str, dm_env.specs.Array]]:
    """See base class."""
    observation_spec = copy.copy(super().observation_spec())
    for obs in observation_spec:
      obs[_COLLECTIVE_REWARD_OBS] = dm_env.specs.Array(
          shape=(), dtype=np.float64, name=_COLLECTIVE_REWARD_OBS)
    return observation_spec
