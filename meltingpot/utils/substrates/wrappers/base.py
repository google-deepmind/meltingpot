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
"""Base class for wrappers.

Wrappers are assumed to own the wrapped environment and that they have the
**only** reference to it. This means that they will:

1.   Close the environment when they close.
2.   Modify the environment specs and timesteps inplace.
"""

import dmlab2d


class Lab2dWrapper(dmlab2d.Environment):
  """Base class for wrappers of dmlab2d.Environments."""

  def __init__(self, env):
    """Initializes the wrapper.

    Args:
      env: An environment to wrap. This environment will be closed with this
        wrapper.
    """
    self._env = env

  def reset(self, *args, **kwargs) -> ...:
    """See base class."""
    return self._env.reset(*args, **kwargs)

  def step(self, *args, **kwargs) -> ...:
    """See base class."""
    return self._env.step(*args, **kwargs)

  def reward_spec(self, *args, **kwargs) -> ...:
    """See base class."""
    return self._env.reward_spec(*args, **kwargs)

  def discount_spec(self, *args, **kwargs) -> ...:
    """See base class."""
    return self._env.discount_spec(*args, **kwargs)

  def observation_spec(self, *args, **kwargs) -> ...:
    """See base class."""
    return self._env.observation_spec(*args, **kwargs)

  def action_spec(self, *args, **kwargs) -> ...:
    """See base class."""
    return self._env.action_spec(*args, **kwargs)

  def close(self, *args, **kwargs) -> ...:
    """See base class."""
    return self._env.close(*args, **kwargs)

  def observation(self, *args, **kwargs) -> ...:
    """See base class."""
    return self._env.observation(*args, **kwargs)

  def events(self, *args, **kwargs) -> ...:
    """See base class."""
    return self._env.events(*args, **kwargs)

  def list_property(self, *args, **kwargs) -> ...:
    """See base class."""
    return self._env.list_property(*args, **kwargs)

  def write_property(self, *args, **kwargs) -> ...:
    """See base class."""
    return self._env.write_property(*args, **kwargs)

  def read_property(self, *args, **kwargs) -> ...:
    """See base class."""
    return self._env.read_property(*args, **kwargs)
