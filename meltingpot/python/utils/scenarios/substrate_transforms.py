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
"""Substrate transforms."""

from typing import TypeVar

import dm_env
import immutabledict
import numpy as np

from meltingpot.python.utils.scenarios.wrappers import agent_slot_wrapper
from meltingpot.python.utils.scenarios.wrappers import all_observations_wrapper
from meltingpot.python.utils.scenarios.wrappers import default_observation_wrapper
from meltingpot.python.utils.substrates import specs

T = TypeVar('T')


def with_tf1_bot_required_observations(substrate: T) -> T:
  """Transforms a substrate to include observations needed by original TF bots.

  We trained the original TF bots with these wrappers present and so we need to
  add them back in so that they execute in the same context as they were
  trained and validated. Newly trained bots should not need these wrappers.

  Args:
    substrate: substrate to add observations to.

  Returns:
    The substrate, with additional wrappers required by the tf1 bots.
  """
  substrate_observations = set(substrate.observation_spec()[0])
  substrate = all_observations_wrapper.Wrapper(
      substrate, observations_to_share=['POSITION'], share_actions=True)
  substrate = agent_slot_wrapper.Wrapper(substrate)
  if 'INVENTORY' not in substrate_observations:
    substrate = default_observation_wrapper.Wrapper(
        substrate, key='INVENTORY', default_value=np.zeros([1]))
  return substrate


def tf1_bot_timestep_spec(
    timestep_spec: dm_env.TimeStep,
    action_spec: dm_env.specs.DiscreteArray,
    num_players: int,
) -> dm_env.TimeStep:
  """Transforms specs to include observations needed by original TF bots.

  Args:
    timestep_spec: substrate timestep spec.
    action_spec: substrate action spec.
    num_players: the number of players.

  Returns:
    The timestep spec, with additional observations required by the tf1 bots.
  """
  global_observations = {}
  if 'POSITION' in timestep_spec.observation:
    position_spec = immutabledict.immutabledict(
        POSITION=specs.int32(num_players, 2))
    global_observations['observations'] = position_spec

  observation_spec = dict(timestep_spec.observation)
  observation_spec['global'] = immutabledict.immutabledict(
      actions=dm_env.specs.BoundedArray(
          shape=[num_players],
          dtype=action_spec.dtype,
          minimum=action_spec.minimum,
          maximum=action_spec.maximum),
      **global_observations)
  observation_spec['agent_slot'] = specs.float32(num_players)
  if 'INVENTORY' not in observation_spec:
    observation_spec['INVENTORY'] = specs.inventory(1)
  return timestep_spec._replace(
      observation=immutabledict.immutabledict(observation_spec))
