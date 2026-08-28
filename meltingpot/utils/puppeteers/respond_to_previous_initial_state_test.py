# Copyright 2026 DeepMind Technologies Limited.
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
"""Regression tests for RespondToPrevious initialization."""

from unittest import mock

from absl.testing import absltest
import dm_env
from meltingpot.utils.puppeteers import in_the_matrix
import numpy as np


_RESOURCE_0 = in_the_matrix.Resource(
    index=0,
    collect_goal=mock.sentinel.collect_0,
    interact_goal=mock.sentinel.interact_0,
)
_RESOURCE_1 = in_the_matrix.Resource(
    index=1,
    collect_goal=mock.sentinel.collect_1,
    interact_goal=mock.sentinel.interact_1,
)


def _first_timestep():
  return dm_env.restart({
      'INVENTORY': np.array([0, 0]),
      'INTERACTION_INVENTORIES': np.array([[-1, -1], [-1, -1]]),
  })


class RespondToPreviousInitialStateTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self._puppeteer = in_the_matrix.RespondToPrevious(
        responses={_RESOURCE_0: _RESOURCE_0, _RESOURCE_1: _RESOURCE_1},
        margin=1,
    )

  def test_initial_state_does_not_sample_random_target(self):
    with mock.patch.object(in_the_matrix.random, 'choice') as choice:
      state = self._puppeteer.initial_state()

    choice.assert_not_called()
    self.assertIs(state, _RESOURCE_0)

  def test_first_timestep_samples_opening_target_once(self):
    initial_state = self._puppeteer.initial_state()
    with mock.patch.object(
        in_the_matrix.random, 'choice', return_value=_RESOURCE_1
    ) as choice:
      timestep, state = self._puppeteer.step(_first_timestep(), initial_state)

    choice.assert_called_once_with([_RESOURCE_0, _RESOURCE_1])
    self.assertIs(state, _RESOURCE_1)
    self.assertIs(timestep.observation['GOAL'], _RESOURCE_1.collect_goal)


if __name__ == '__main__':
  absltest.main()
