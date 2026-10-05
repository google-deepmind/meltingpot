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
"""Native regression tests for hopper processing isolation test."""

from pathlib import Path

from absl.testing import absltest
from absl.testing import parameterized
import dmlab2d
from dmlab2d import runfiles_helper
from meltingpot import substrate
from meltingpot.utils.substrates import builder
import numpy as np


class HopperProcessingIsolationTest(parameterized.TestCase):

  def test_native_lua_regressions(self):
    root = Path(__file__).resolve().parents[3]
    lab = dmlab2d.Lab2d(
        runfiles_helper.find(),
        {
            'levelDirectory': str(root),
            'levelName': (
                'meltingpot/lua/levels/factory_of_the_commons/hopper_processing_isolation_test'
            ),
        },
    )
    self.assertEmpty(lab.observation_names())

  def test_short_native_episodes_reset_cleanly(self):
    name = 'factory_commons__either_or'
    config = substrate.get_config(name)
    settings = config.lab2d_settings_builder(
        roles=config.default_player_roles, config=config
    )
    settings['maxEpisodeLengthFrames'] = 3
    with builder.builder(settings, env_seed=17) as env:
      actions = {
          key: np.zeros(spec.shape, spec.dtype)
          for key, spec in env.action_spec().items()
      }
      for _ in range(2):
        timestep = env.reset()
        self.assertTrue(timestep.first())
        for step in range(1, 4):
          timestep = env.step(actions)
          self.assertEqual(timestep.last(), step == 3)


if __name__ == '__main__':
  absltest.main()
