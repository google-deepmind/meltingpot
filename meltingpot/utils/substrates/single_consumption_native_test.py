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
"""Native regression tests for single consumption test."""

import copy
from pathlib import Path

from absl.testing import absltest
from absl.testing import parameterized
import dmlab2d
from dmlab2d import runfiles_helper
from meltingpot import substrate
from meltingpot.utils.substrates import builder
import numpy as np


class SingleConsumptionTest(absltest.TestCase):

  def test_native_lua_regressions(self):
    root = Path(__file__).resolve().parents[3]
    lab = dmlab2d.Lab2d(
        runfiles_helper.find(),
        {
            'levelDirectory': str(root),
            'levelName': (
                'meltingpot/lua/levels/predator_prey/single_consumption_test'
            ),
        },
    )
    self.assertEmpty(lab.observation_names())


class NativePredatorSingleConsumptionTest(parameterized.TestCase):

  @parameterized.product(seed=(5, 17, 29), victim=('prey', 'predator'))
  def test_two_simultaneous_beams_consume_one_victim(self, seed, victim):
    config = substrate.get_config('predator_prey__open')
    roles = ('predator', victim, 'predator')
    settings = config.lab2d_settings_builder(roles=roles, config=config)
    settings['maxEpisodeLengthFrames'] = 12
    simulation = settings['simulation']
    simulation['map'] = 'fffffff\nfffffff\nfabcfff\nfffffff\nfffffff'
    simulation['charPrefabMap'] = {'f': 'tiled_floor'}
    for index, role in enumerate(roles, 1):
      name = f'fixed_spawn_{index}'
      spawn = copy.deepcopy(simulation['prefabs'][f'spawn_point_{role}'])
      spawn['components'][0]['kwargs']['stateConfigs'][0]['groups'] = [name]
      simulation['prefabs'][name] = spawn
      simulation['charPrefabMap']['abc'[index - 1]] = {
          'type': 'all',
          'list': ['tiled_floor', name],
      }
    for obj in simulation['gameObjects']:
      for component in list(obj['components']):
        if component['component'] == 'Avatar':
          index = component['kwargs']['index']
          component['kwargs']['spawnGroup'] = f'fixed_spawn_{index}'
          obj['components'].append({
              'component': 'LocationObserver',
              'kwargs': {'objectIsAvatar': True, 'alsoReportOrientation': True},
          })
    with builder.builder(settings, env_seed=seed) as env:
      for _ in range(2):
        timestep = env.reset()
        np.testing.assert_array_equal(
            [timestep.observation[f'{index}.POSITION'] for index in (1, 2, 3)],
            [[1, 2], [2, 2], [3, 2]],
        )
        actions = {
            key: np.zeros(spec.shape, spec.dtype)
            for key, spec in env.action_spec().items()
        }
        for _ in range(3):
          orientations = [
              int(timestep.observation[f'{index}.ORIENTATION'])
              for index in (1, 3)
          ]
          if orientations == [1, 3]:
            break
          for index, desired, current in zip((1, 3), (1, 3), orientations):
            difference = (desired - current) % 4
            actions[f'{index}.turn'][...] = np.int64(
                0 if difference == 0 else -1 if difference == 3 else 1
            )
          timestep = env.step(actions)
        self.assertEqual(
            [
                int(timestep.observation[f'{index}.ORIENTATION'])
                for index in (1, 3)
            ],
            [1, 3],
        )
        for key in actions:
          actions[key][...] = 0
        actions['1.interact'][...] = 1
        actions['3.interact'][...] = 1
        timestep = env.step(actions)
        expected_event = f'{victim}_consumed'
        consumption_events = [
            item for item in env.events() if item[0] == expected_event
        ]
        self.assertLen(consumption_events, 1)
        rewards = [
            float(timestep.observation[f'{index}.REWARD']) for index in (1, 3)
        ]
        self.assertEqual(sum(rewards), 1.0 if victim == 'prey' else 0.0)
        if victim == 'prey':
          self.assertEqual(sorted(rewards), [0.0, 1.0])
        self.assertEqual(float(timestep.observation['2.REWARD']), 0.0)


if __name__ == '__main__':
  absltest.main()
