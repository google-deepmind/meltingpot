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
"""Native regression tests for potential reward settlement test."""

from pathlib import Path

from absl.testing import absltest
import dmlab2d
from dmlab2d import runfiles_helper
from meltingpot import substrate
from meltingpot.utils.substrates import builder
import numpy as np


class PotentialRewardSettlementTest(absltest.TestCase):

  def test_native_lua_regressions(self):
    root = Path(__file__).resolve().parents[3]
    lab = dmlab2d.Lab2d(
        runfiles_helper.find(),
        {
            'levelDirectory': str(root),
            'levelName': (
                'meltingpot/lua/levels/hidden_agenda/potential_reward_settlement_test'
            ),
        },
    )
    self.assertEmpty(lab.observation_names())


class NativePotentialRewardTerminationTest(absltest.TestCase):

  def test_voting_win_terminates_with_potential_rewards_enabled(self):
    config = substrate.get_config('hidden_agenda')
    settings = config.lab2d_settings_builder(
        roles=config.default_player_roles, config=config
    )
    settings['maxEpisodeLengthFrames'] = 12
    for component in settings['simulation']['scene']['components']:
      if component['component'] == 'Progress':
        component['kwargs']['potential_pseudorewards'] = True
        component['kwargs']['voting_params']['votingFrameFrequency'] = 1
        component['kwargs']['voting_params']['votingPhaseCooldown'] = 3
    with builder.builder(settings, env_seed=17) as env:
      for _ in range(2):
        timestep = env.reset()
        returns = np.zeros(5)
        steps = 0
        actions = {
            key: np.int64(5 if key.endswith('.vote') else 0)
            for key in env.action_spec()
        }
        while not timestep.last():
          timestep = env.step(actions)
          returns += [
              float(timestep.observation[f'{index}.REWARD'])
              for index in range(1, 6)
          ]
          steps += 1
          self.assertLessEqual(steps, 12)
        self.assertLess(steps, 12)
        np.testing.assert_array_equal(returns, [1, 1, 1, 1, -1])


if __name__ == '__main__':
  absltest.main()
