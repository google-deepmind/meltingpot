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
"""Regression tests for matrix puppeteer initial state behavior."""

from unittest import mock

from absl.testing import absltest
import dm_env
from meltingpot.utils.puppeteers import in_the_matrix
import numpy as np


_COOPERATE = in_the_matrix.Resource(
    index=1,
    collect_goal=mock.sentinel.cooperate_collect,
    interact_goal=mock.sentinel.cooperate_interact,
)
_DEFECT = in_the_matrix.Resource(
    index=0,
    collect_goal=mock.sentinel.defect_collect,
    interact_goal=mock.sentinel.defect_interact,
)


class TitForTatInitialStateTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.puppeteer = in_the_matrix.TitForTat(
        cooperate_resource=_COOPERATE,
        defect_resource=_DEFECT,
        margin=1,
        tremble_probability=0.5,
    )

  @mock.patch.object(in_the_matrix, 'tremble')
  def test_initial_state_has_no_rng_side_effect(self, tremble):
    self.assertTrue(self.puppeteer.initial_state())
    tremble.assert_not_called()

  @mock.patch.object(in_the_matrix, 'tremble', return_value=True)
  def test_first_timestep_samples_opening_tremble_once(self, tremble):
    timestep = dm_env.restart({
        'INVENTORY': np.array([0, 0]),
        'INTERACTION_INVENTORIES': np.array(([-1, -1], [-1, -1])),
    })

    transformed, state = self.puppeteer.step(
        timestep, self.puppeteer.initial_state())

    self.assertFalse(state)
    self.assertIs(transformed.observation['GOAL'], _DEFECT.collect_goal)
    tremble.assert_called_once_with(0.5)


if __name__ == '__main__':
  absltest.main()
