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


class AvatarMembershipNativeTest(parameterized.TestCase):

  def test_duplicate_query_results_in_native_lua(self):
    root = Path(__file__).resolve().parents[3]
    lab = dmlab2d.Lab2d(
        runfiles_helper.find(),
        {
            'levelDirectory': str(root),
            'levelName': 'meltingpot/lua/modules/avatar_membership_test',
        },
    )
    self.assertEmpty(lab.observation_names())

  @parameterized.parameters(False, True)
  def test_duplicate_layers_produce_binary_observations_in_native_episodes(
      self, duplicate_layer
  ):
    config = substrate.get_config('commons_harvest__open')
    settings = config.lab2d_settings_builder(
        roles=('default', 'default'), config=config
    )
    settings['maxEpisodeLengthFrames'] = 3
    layers = ['upperPhysical'] * (2 if duplicate_layer else 1)
    for obj in settings['simulation']['gameObjects']:
      obj['components'].extend([
          {
              'component': 'AvatarIdsInViewObservation',
              'kwargs': {'layers': layers},
          },
          {'component': 'AvatarIdsInRangeToZapObservation'},
      ])
    with builder.builder(settings, env_seed=5) as env:
      actions = {name: 0 for name in env.action_spec()}
      for _ in range(2):
        timestep = env.reset()
        for _ in range(4):
          for player in (1, 2):
            for suffix in ('AVATAR_IDS_IN_VIEW', 'AVATAR_IDS_IN_RANGE_TO_ZAP'):
              key = f'{player}.{suffix}'
              values = timestep.observation[key]
              env.observation_spec()[key].validate(values)
              self.assertEqual(values.shape, (2,))
              self.assertEqual(values.dtype, np.dtype(np.int32))
              self.assertTrue(np.logical_or(values == 0, values == 1).all())
            self.assertEqual(
                timestep.observation[f'{player}.AVATAR_IDS_IN_VIEW'][
                    player - 1
                ],
                1,
            )
          if not timestep.last():
            timestep = env.step(actions)


if __name__ == '__main__':
  absltest.main()
