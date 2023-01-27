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
"""Helpers for defining substrate specs.

Used to allow substrates to easily define the single-player specs within their
configs.
"""

from typing import Mapping, Optional

import dm_env
import immutabledict
import numpy as np

STEP_TYPE = dm_env.specs.BoundedArray(
    shape=(),
    dtype=np.int64,
    minimum=min(dm_env.StepType),
    maximum=max(dm_env.StepType),
    name='step_type',
)
DISCOUNT = dm_env.specs.BoundedArray(
    shape=(), dtype=np.float64, minimum=0, maximum=1, name='discount')
REWARD = dm_env.specs.Array(shape=(), dtype=np.float64, name='reward')
OBSERVATION = immutabledict.immutabledict({
    'READY_TO_SHOOT': dm_env.specs.Array(
        shape=(), dtype=np.float64, name='READY_TO_SHOOT'),
    'RGB': dm_env.specs.Array(shape=(88, 88, 3), dtype=np.uint8, name='RGB'),
    'POSITION': dm_env.specs.Array(shape=(2,), dtype=np.int32, name='POSITION'),
    'ORIENTATION': dm_env.specs.Array(
        shape=(), dtype=np.int32, name='ORIENTATION'),
})
_ACTION = dm_env.specs.DiscreteArray(
    num_values=1, dtype=np.int64, name='action')


def float32(*shape: int, name: Optional[str] = None) -> dm_env.specs.Array:
  """Returns the spec for an np.float32 tensor.

  Args:
    *shape: the shape of the tensor.
    name: optional name for the spec.
  """
  return dm_env.specs.Array(shape=shape, dtype=np.float32, name=name)


def float64(*shape: int, name: Optional[str] = None) -> dm_env.specs.Array:
  """Returns the spec for an np.float64 tensor.

  Args:
    *shape: the shape of the tensor.
    name: optional name for the spec.
  """
  return dm_env.specs.Array(shape=shape, dtype=np.float64, name=name)


def int32(*shape: int, name: Optional[str] = None) -> dm_env.specs.Array:
  """Returns the spec for an np.int32 tensor.

  Args:
    *shape: the shape of the tensor.
    name: optional name for the spec.
  """
  return dm_env.specs.Array(shape=shape, dtype=np.int32, name=name)


def int64(*shape: int, name: Optional[str] = None) -> dm_env.specs.Array:
  """Returns the spec for an np.int32 tensor.

  Args:
    *shape: the shape of the tensor.
    name: optional name for the spec.
  """
  return dm_env.specs.Array(shape=shape, dtype=np.int64, name=name)


def action(num_actions: int) -> dm_env.specs.DiscreteArray:
  """Returns the spec for an action.

  Args:
    num_actions: the number of actions that can be taken.
  """
  return _ACTION.replace(num_values=num_actions)


def rgb(height: int,
        width: int,
        name: Optional[str] = 'RGB') -> dm_env.specs.Array:
  """Returns the spec for an RGB observation.

  Args:
    height: the height of the observation.
    width: the width of the observation.
    name: optional name for the spec.
  """
  return OBSERVATION['RGB'].replace(shape=(height, width, 3), name=name)


def world_rgb(ascii_map: str,
              sprite_size: int,
              name: Optional[str] = 'WORLD.RGB') -> dm_env.specs.Array:
  """Returns the spec for a WORLD.RGB observation.

  Args:
    ascii_map: the height of the observation.
    sprite_size: the width of the observation.
    name: optional name for the spec.
  """
  lines = ascii_map.strip().split('\n')
  height = len(lines) * sprite_size
  width = len(lines[0]) * sprite_size if height else 0
  return rgb(height, width, name)


def inventory(num_resources: int,
              name: Optional[str] = 'INVENTORY') -> dm_env.specs.Array:
  """Returns the spec for an INVENTORY observation.

  Args:
    num_resources: the number of resource types in the inventory.
    name: optional name for the spec.
  """
  return float64(num_resources, name=name)


def interaction_inventories(
    num_resources: int,
    name: Optional[str] = 'INTERACTION_INVENTORIES') -> dm_env.specs.Array:
  """Returns the spec for an INTERACTION_INVENTORIES observation.

  Args:
    num_resources: the number of resource types in the inventory.
    name: optional name for the spec.
  """
  return float64(2, num_resources, name=name)


def timestep(
    observation_spec: Mapping[str, dm_env.specs.Array]) -> dm_env.TimeStep:
  """Returns the spec for a timestep.

  Args:
    observation_spec: the observation spec. Spec names will be overwritten with
      their key.
  """
  observation_spec = immutabledict.immutabledict({
      name: spec.replace(name=name) for name, spec in observation_spec.items()
  })
  return dm_env.TimeStep(
      step_type=STEP_TYPE,
      discount=DISCOUNT,
      reward=REWARD,
      observation=observation_spec,
  )
