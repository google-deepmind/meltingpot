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
"""Regression tests for DMLab2D player index validation."""

from unittest import mock

from absl.testing import parameterized
import dm_env
import dmlab2d
from meltingpot.utils.substrates.wrappers import multiplayer_wrapper
import numpy as np


ACT_SPEC = dm_env.specs.BoundedArray(
    shape=(), minimum=0, maximum=4, dtype=np.int8)


class MultiplayerPlayerIndexValidationTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('zero_based', {'0.MOVE': ACT_SPEC, '1.MOVE': ACT_SPEC}),
      ('gapped', {'1.MOVE': ACT_SPEC, '3.MOVE': ACT_SPEC}),
  )
  def test_rejects_non_contiguous_player_indices(self, action_spec):
    env = mock.Mock(spec_set=dmlab2d.Environment)
    env.action_spec.return_value = action_spec

    with self.assertRaisesRegex(ValueError, 'contiguous and start at 1'):
      multiplayer_wrapper.Wrapper(
          env,
          individual_observation_names=[],
          global_observation_names=[],
      )

  def test_rejects_empty_action_spec(self):
    env = mock.Mock(spec_set=dmlab2d.Environment)
    env.action_spec.return_value = {}

    with self.assertRaisesRegex(ValueError, 'at least one player action'):
      multiplayer_wrapper.Wrapper(
          env,
          individual_observation_names=[],
          global_observation_names=[],
      )


if __name__ == '__main__':
  parameterized.absltest.main()
