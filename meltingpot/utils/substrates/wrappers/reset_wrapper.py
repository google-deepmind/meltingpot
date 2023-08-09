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
"""Wrapper that rebuilds the Lab2d environment on every reset."""

from typing import Callable

import dm_env
import dmlab2d
from meltingpot.utils.substrates.wrappers import base


class ResetWrapper(base.Lab2dWrapper):
  """Wrapper that rebuilds the environment on reset."""

  def __init__(self, build_environment: Callable[[], dmlab2d.Environment]):
    """Initializes the object.

    Args:
      build_environment: Called to build the underlying environment.
    """
    env = build_environment()
    super().__init__(env)
    self._rebuild_environment = build_environment
    self._reset = False

  def reset(self) -> dm_env.TimeStep:
    """Rebuilds the environment and calls reset on it."""
    if self._reset:
      self._env.close()
      self._env = self._rebuild_environment()
    else:
      # Don't rebuild on very first reset call (it's inefficient).
      self._reset = True
    return super().reset()
