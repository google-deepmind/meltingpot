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
"""Tests for running_with_scissors puppeteers."""

import itertools
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from meltingpot.testing import puppeteers
from meltingpot.utils.puppeteers import coordination_in_the_matrix
from meltingpot.utils.puppeteers import in_the_matrix
import numpy as np

_RESOURCE_A = in_the_matrix.Resource(
    index=0,
    collect_goal=mock.sentinel.collect_0,
    interact_goal=mock.sentinel.interact_0,
)
_RESOURCE_B = in_the_matrix.Resource(
    index=1,
    collect_goal=mock.sentinel.collect_1,
    interact_goal=mock.sentinel.interact_1,
)
_RESOURCE_C = in_the_matrix.Resource(
    index=2,
    collect_goal=mock.sentinel.collect_2,
    interact_goal=mock.sentinel.interact_2,
)


def _observation(inventory, interaction):
  return {
      'INVENTORY': np.array(inventory),
      'INTERACTION_INVENTORIES': np.array(interaction),
  }


def _goals_from_observations(puppeteer, inventories, interactions, state=None):
  observations = []
  for inventory, interaction in itertools.zip_longest(inventories,
                                                      interactions):
    observations.append(_observation(inventory, interaction))
  return puppeteers.goals_from_observations(puppeteer, observations, state)


class CounterPrevious(parameterized.TestCase):

  def test_counters(self):
    puppeteer = coordination_in_the_matrix.CoordinateWithPrevious(
        resources=(_RESOURCE_A, _RESOURCE_B, _RESOURCE_C),
        margin=1,
    )
    inventories = [
        (1, 1, 1),
        (1, 2, 1),
        (3, 2, 1),
        (3, 3, 1),
        (2, 3, 1),
        (1, 2, 1),
        (1, 2, 3),
    ]
    interactions = [
        ((-1, -1, -1), (-1, -1, -1)),  # neither
        ((-1, -1, -1), (1, 0, 0)),  # A
        ((-1, -1, -1), (-1, -1, -1)),  # neither
        ((-1, -1, -1), (0, 1, 0)),  # B
        ((-1, -1, -1), (-1, -1, -1)),  # neither
        ((-1, -1, -1), (0, 0, 1)),  # C
        ((-1, -1, -1), (-1, -1, -1)),  # neither
    ]
    expected = [
        mock.ANY,  # random
        _RESOURCE_A.collect_goal,
        _RESOURCE_A.interact_goal,
        _RESOURCE_B.collect_goal,
        _RESOURCE_B.interact_goal,
        _RESOURCE_C.collect_goal,
        _RESOURCE_C.interact_goal,
    ]
    actual, _ = _goals_from_observations(puppeteer, inventories, interactions)
    self.assertEqual(actual, expected)


if __name__ == '__main__':
  absltest.main()
