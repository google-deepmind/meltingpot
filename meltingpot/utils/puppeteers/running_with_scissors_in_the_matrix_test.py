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
from meltingpot.utils.puppeteers import in_the_matrix
from meltingpot.utils.puppeteers import running_with_scissors_in_the_matrix
import numpy as np

_ROCK = in_the_matrix.Resource(
    index=2,
    collect_goal=mock.sentinel.collect_rock,
    interact_goal=mock.sentinel.interact_rock,
)
_PAPER = in_the_matrix.Resource(
    index=1,
    collect_goal=mock.sentinel.collect_rock,
    interact_goal=mock.sentinel.interact_rock,
)
_SCISSORS = in_the_matrix.Resource(
    index=0,
    collect_goal=mock.sentinel.collect_rock,
    interact_goal=mock.sentinel.interact_rock,
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
    puppeteer = running_with_scissors_in_the_matrix.CounterPrevious(
        rock_resource=_ROCK,
        paper_resource=_PAPER,
        scissors_resource=_SCISSORS,
        margin=1,
    )
    inventories = [
        (1, 1, 1),
        (1, 2, 1),
        (1, 2, 3),
        (2, 3, 1),
        (3, 2, 1),
        (3, 2, 1),
        (2, 3, 1),
    ]
    interactions = [
        ((-1, -1, -1), (-1, -1, -1)),  # neither
        ((-1, -1, -1), (1, 0, 0)),  # scissors
        ((-1, -1, -1), (-1, -1, -1)),  # neither
        ((-1, -1, -1), (0, 1, 0)),  # paper
        ((-1, -1, -1), (-1, -1, -1)),  # neither
        ((-1, -1, -1), (0, 0, 1)),  # rock
        ((-1, -1, -1), (-1, -1, -1)),  # neither
    ]
    expected = [
        mock.ANY,  # random
        _ROCK.collect_goal,
        _ROCK.interact_goal,
        _SCISSORS.collect_goal,
        _SCISSORS.interact_goal,
        _PAPER.collect_goal,
        _PAPER.interact_goal,
    ]
    actual, _ = _goals_from_observations(puppeteer, inventories, interactions)
    self.assertEqual(actual, expected)


if __name__ == '__main__':
  absltest.main()
