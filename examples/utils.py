# Copyright 2022 DeepMind Technologies Limited.
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
"""Shared utils for third-party library examples."""

from typing import Any, Mapping

import dm_env
from gymnasium import spaces
import numpy as np
import tree

PLAYER_STR_FORMAT = 'player_{index}'
_WORLD_PREFIX = 'WORLD.'


def timestep_to_observations(timestep: dm_env.TimeStep) -> Mapping[str, Any]:
  gym_observations = {}
  for index, observation in enumerate(timestep.observation):
    gym_observations[PLAYER_STR_FORMAT.format(index=index)] = {
        key: value
        for key, value in observation.items()
        if _WORLD_PREFIX not in key
    }
  return gym_observations


def remove_world_observations_from_space(
    observation: spaces.Dict) -> spaces.Dict:
  return spaces.Dict({
      key: observation[key] for key in observation if _WORLD_PREFIX not in key
  })


def spec_to_space(spec: tree.Structure[dm_env.specs.Array]) -> spaces.Space:
  """Converts a dm_env nested structure of specs to a Gym Space.

  BoundedArray is converted to Box Gym spaces. DiscreteArray is converted to
  Discrete Gym spaces. Using Tuple and Dict spaces recursively as needed.

  Args:
    spec: The nested structure of specs

  Returns:
    The Gym space corresponding to the given spec.
  """
  if isinstance(spec, dm_env.specs.DiscreteArray):
    return spaces.Discrete(spec.num_values)
  elif isinstance(spec, dm_env.specs.BoundedArray):
    return spaces.Box(spec.minimum, spec.maximum, spec.shape, spec.dtype)
  elif isinstance(spec, dm_env.specs.Array):
    if np.issubdtype(spec.dtype, np.floating):
      return spaces.Box(-np.inf, np.inf, spec.shape, spec.dtype)
    elif np.issubdtype(spec.dtype, np.integer):
      info = np.iinfo(spec.dtype)
      return spaces.Box(info.min, info.max, spec.shape, spec.dtype)
    else:
      raise NotImplementedError(f'Unsupported dtype {spec.dtype}')
  elif isinstance(spec, (list, tuple)):
    return spaces.Tuple([spec_to_space(s) for s in spec])
  elif isinstance(spec, dict):
    return spaces.Dict({key: spec_to_space(s) for key, s in spec.items()})
  else:
    raise ValueError('Unexpected spec of type {}: {}'.format(type(spec), spec))
