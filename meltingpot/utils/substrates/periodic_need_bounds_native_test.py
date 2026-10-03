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


class PeriodicNeedNativeTest(parameterized.TestCase):

  def test_need_signal_and_reward_schedule_in_native_lua(self):
    root = Path(__file__).resolve().parents[3]
    lab = dmlab2d.Lab2d(
        runfiles_helper.find(),
        {
            'levelDirectory': str(root),
            'levelName': 'meltingpot/lua/modules/periodic_need_bounds_test',
        },
    )
    self.assertEmpty(lab.observation_names())

  @parameterized.parameters(1, 2, 4)
  def test_hunger_observations_saturate_in_real_fruit_market_episodes(
      self, delay
  ):
    config = substrate.get_config('fruit_market__concentric_rivers')
    settings = config.lab2d_settings_builder(
        roles=config.default_player_roles, config=config
    )
    settings['maxEpisodeLengthFrames'] = 6
    for obj in settings['simulation']['gameObjects']:
      for component in obj['components']:
        if component['component'] == 'PeriodicNeed':
          component['kwargs']['delay'] = delay
    expected = np.minimum(np.arange(7) / delay, 1.0)
    with builder.builder(settings, env_seed=7) as env:
      actions = {name: 0 for name in env.action_spec()}
      for _ in range(2):
        timestep = env.reset()
        keys = [key for key in timestep.observation if key.endswith('.HUNGER')]
        self.assertNotEmpty(keys)
        values = [[float(timestep.observation[key]) for key in keys]]
        while not timestep.last():
          timestep = env.step(actions)
          values.append([float(timestep.observation[key]) for key in keys])
        np.testing.assert_allclose(
            values, np.broadcast_to(expected[:, None], (7, len(keys)))
        )


if __name__ == '__main__':
  absltest.main()
