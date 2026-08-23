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
"""Regression tests for substrate test diagnostics."""

from unittest import mock

from absl.testing import absltest
import dm_env
from meltingpot.testing import substrates
import numpy as np


class SubstrateDiagnosticsTest(substrates.SubstrateTestCase):

  def test_observation_key_mismatch_reports_spec_keys(self):
    env = mock.Mock()
    env.action_spec.return_value = (dm_env.specs.DiscreteArray(num_values=1),)
    env.discount_spec.return_value = dm_env.specs.BoundedArray(
        shape=(), dtype=np.float64, minimum=0.0, maximum=1.0)
    env.reward_spec.return_value = (
        dm_env.specs.Array(shape=(), dtype=np.float64),)
    env.observation_spec.return_value = ({
        'expected': dm_env.specs.Array(shape=(), dtype=np.float64),
    },)
    env.step.return_value = dm_env.transition(
        reward=(0.0,), observation=({'actual': np.array(0.0)},))

    with self.assertRaisesRegex(
        AssertionError,
        r"Observation 0 keys \{'actual'\} do not match spec keys \{'expected'\}",
    ):
      self.assert_step_matches_specs(env)


if __name__ == '__main__':
  absltest.main()
