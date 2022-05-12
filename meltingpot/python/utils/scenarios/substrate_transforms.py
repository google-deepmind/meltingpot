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
"""Scenario factory."""

from typing import TypeVar

import numpy as np

from meltingpot.python.utils.scenarios.wrappers import agent_slot_wrapper
from meltingpot.python.utils.scenarios.wrappers import all_observations_wrapper
from meltingpot.python.utils.scenarios.wrappers import default_observation_wrapper

T = TypeVar('T')


def with_tf1_bot_required_observations(substrate: T) -> T:
  """Transforms a substrate to include observations needed by tf bots.

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
