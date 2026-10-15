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

import copy
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from meltingpot.configs.substrates import running_with_scissors_in_the_matrix__repeated as test_substrate
from meltingpot.utils.substrates import builder
from ml_collections import config_dict
import numpy as np


def _get_test_settings():
  config = test_substrate.get_config()
  return test_substrate.build(config, config.default_player_roles)


_TEST_SETTINGS = _get_test_settings()


def _get_lua_randomization_map():
  """Native component regressions and short environment checks."""


import copy
from pathlib import Path

from absl.testing import absltest
from absl.testing import parameterized
import dmlab2d
from dmlab2d import runfiles_helper
from meltingpot import substrate
from meltingpot.utils.substrates import builder
import numpy as np


class PaintballEliminationOnceNativeTest(parameterized.TestCase):

  def test_native_component_regressions(self):
    root = Path(__file__).resolve().parents[3]
    lab = dmlab2d.Lab2d(
        runfiles_helper.find(),
        {
            'levelDirectory': str(root),
            'levelName': (
                'meltingpot/lua/levels/paintball/paintball_elimination_once_test'
            ),
        },
    )
    self.assertEmpty(lab.observation_names())

  @parameterized.parameters(
      'paintball__king_of_the_hill', 'paintball__capture_the_flag'
  )
  def test_short_public_environment_episodes(self, name):
    config = substrate.get_config(name)
    settings = copy.deepcopy(
        config.lab2d_settings_builder(
            roles=config.default_player_roles, config=config
        )
    )
    settings['maxEpisodeLengthFrames'] = 6
    with builder.builder(settings, env_seed=17) as env:
      actions = {
          name: 1 if name.endswith('.fireZap') else 0
          for name in env.action_spec()
      }
      for _ in range(2):
        timestep = env.reset()
        for _ in range(7):
          for name, value in timestep.observation.items():
            if name.endswith('RGB'):
              env.observation_spec()[name].validate(value)
          if timestep.reward is not None:
            self.assertTrue(np.isfinite(np.asarray(timestep.reward)).all())
          if timestep.last():
            break
          timestep = env.step(actions)
        self.assertTrue(timestep.last())


if __name__ == '__main__':
  absltest.main()
