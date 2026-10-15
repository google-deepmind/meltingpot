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

"""Native regression tests for avatar observation signals."""

from pathlib import Path

from absl.testing import absltest
from absl.testing import parameterized
import dmlab2d
from dmlab2d import runfiles_helper
from meltingpot import substrate
from meltingpot.utils.substrates import builder
import numpy as np


class ZapperReadinessNativeTest(parameterized.TestCase):

  def test_readiness_rules_in_native_lua(self):
    root = Path(__file__).resolve().parents[3]
    lab = dmlab2d.Lab2d(
        runfiles_helper.find(),
        {
            'levelDirectory': str(root),
            'levelName': 'meltingpot/lua/modules/zapper_readiness_test',
        },
    )
    self.assertEmpty(lab.observation_names())

  @parameterized.parameters(-1, 0, 2)
  def test_ready_to_shoot_observations_in_real_episodes(self, cooldown):
    config = substrate.get_config('commons_harvest__open')
    settings = config.lab2d_settings_builder(roles=('default',), config=config)
    settings['maxEpisodeLengthFrames'] = 5
    for obj in settings['simulation']['gameObjects']:
      for component in obj['components']:
        if component['component'] == 'Zapper':
          component['kwargs']['cooldownTime'] = cooldown
    expected = (
        [0.0] * 6
        if cooldown < 0
        else [1.0] * 6
        if cooldown == 0
        else [1.0, 0.0, 0.5, 1.0, 0.0, 0.5]
    )
    with builder.builder(settings, env_seed=17) as env:
      for _ in range(2):
        timestep = env.reset()
        values = [float(timestep.observation['1.READY_TO_SHOOT'])]
        while not timestep.last():
          timestep = env.step({'1.move': 0, '1.turn': 0, '1.fireZap': 1})
          values.append(float(timestep.observation['1.READY_TO_SHOOT']))
        np.testing.assert_allclose(values, expected, rtol=0, atol=0)


if __name__ == '__main__':
  absltest.main()
