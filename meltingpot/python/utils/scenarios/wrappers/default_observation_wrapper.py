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
"""The Default Observation Wrapper adds an empty field to the observation.

For instance, this wrapper can be used to add an INVENTORY observation for all
players, if one is not already there. This is useful to match APIs between
different environments that expose different observations.
"""

from typing import Any, Dict, Optional

import dm_env
import numpy as np

from meltingpot.python.utils.scenarios.wrappers import base
from meltingpot.python.utils.substrates import substrate


def _setdefault(dictionary: Dict[str, Any], key: str,
                value: Any) -> Dict[str, Any]:
  """Sets the default value of `key` to `value` if necessary.

  Args:
    dictionary: the dictionary to add a default for.
    key: The key to add a default for.
    value: The default value to add if key is missing.

  Returns:
    Either dictionary or a copy wiht the default value added.
  """
  if key in dictionary:
    return dictionary
  else:
    return dict(dictionary, **{key: value})


class Wrapper(base.SubstrateWrapper):
  """Wrapper to add observations with default values if not actually present."""

  def __init__(self,
               env: substrate.Substrate,
               key: str,
               default_value: np.ndarray,
               default_spec: Optional[dm_env.specs.Array] = None):
    """Initializer.

    Args:
      env: environment to wrap. When this wrapper closes env will also be
        closed.
      key: field name of the observation to add.
      default_value: The default value to add to the observation.
      default_spec: The default spec for the observation to add. By default,
        this will be set to match default_value. If specified this must match
        the deafult value.
    """
    super().__init__(env)
    self._key = key
    self._default_value = default_value.copy()
    self._default_value.flags.writeable = False
    if default_spec is None:
      self._default_spec = dm_env.specs.Array(
          shape=self._default_value.shape,
          dtype=self._default_value.dtype,
          name=self._key)
    else:
      self._default_spec = default_spec
    self._default_spec.validate(self._default_value)

  def reset(self):
    """See base class."""
    timestep = super().reset()
    observation = [
        _setdefault(obs, self._key, self._default_value)
        for obs in timestep.observation
    ]
    return timestep._replace(observation=observation)

  def step(self, action):
    """See base class."""
    timestep = super().step(action)
    observation = [
        _setdefault(obs, self._key, self._default_value)
        for obs in timestep.observation
    ]
    return timestep._replace(observation=observation)

  def observation_spec(self):
    """See base class."""
    observation_spec = super().observation_spec()
    return [
        _setdefault(obs, self._key, self._default_spec)
        for obs in observation_spec
    ]
