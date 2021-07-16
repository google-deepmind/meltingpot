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
"""Removes observations from the environment."""

from typing import Collection, Mapping, TypeVar

import dm_env

from meltingpot.python.utils.scenarios.wrappers import base

T = TypeVar('T')


class Wrapper(base.Wrapper):
  """Removes observations from all players."""

  def __init__(self,
               env: base.Substrate,
               permitted_observations: Collection[str]) -> None:
    """Wraps an environment.

    Args:
      env: environment to wrap.
      permitted_observations: observation keys to allow in all players
        observations.
    """
    super().__init__(env)
    self._permitted_observations = permitted_observations

  def _adjusted_observation(
      self, observation: Mapping[str, T]) -> Mapping[str, T]:
    return {
        key: observation[key]
        for key in self._permitted_observations if key in observation
    }

  def _adjusted_timestep(self, timestep: dm_env.TimeStep) -> dm_env.TimeStep:
    observations = [
        self._adjusted_observation(observation)
        for observation in timestep.observation
    ]
    return timestep._replace(observation=observations)

  def reset(self):
    """See base class."""
    timestep = super().reset()
    return self._adjusted_timestep(timestep)

  def step(self, actions):
    """See base class."""
    timestep = super().step(actions)
    return self._adjusted_timestep(timestep)

  def observation_spec(self):
    """See base class."""
    observation_spec = super().observation_spec()
    return [self._adjusted_observation(spec) for spec in observation_spec]
