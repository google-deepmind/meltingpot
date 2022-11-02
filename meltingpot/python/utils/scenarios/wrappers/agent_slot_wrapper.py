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
"""Wrap environment, adding the agent's player slot to the observation."""

import dm_env
import numpy as np

from meltingpot.python.utils.scenarios.wrappers import base

AGENT_SLOT = 'agent_slot'


def _augment_timestep(timestep: dm_env.TimeStep) -> dm_env.TimeStep:
  """Returns a new timestep with player slot added as a one-hot."""
  player_slots = np.eye(len(timestep.observation), dtype=np.float32)
  observations = [
      dict(observation, **{AGENT_SLOT: player_slots[n]})
      for n, observation in enumerate(timestep.observation)
  ]
  return timestep._replace(observation=observations)


class Wrapper(base.SubstrateWrapper):
  """Adds agent's player slot as a one-hot observation."""

  def reset(self):
    """See base class."""
    timestep = super().reset()
    return _augment_timestep(timestep)

  def step(self, actions):
    """See base class."""
    timestep = super().step(actions)
    return _augment_timestep(timestep)

  def observation_spec(self):
    """See base class."""
    observation_spec = super().observation_spec()
    num_players = len(observation_spec)
    slot_spec = dm_env.specs.Array(
        shape=[num_players], dtype=np.float32, name=AGENT_SLOT)
    return [dict(spec, **{AGENT_SLOT: slot_spec}) for spec in observation_spec]
