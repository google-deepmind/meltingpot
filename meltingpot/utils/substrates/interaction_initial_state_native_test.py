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
"""Native regression tests for interaction initial state test."""

from pathlib import Path

from absl.testing import absltest
from absl.testing import parameterized
import dmlab2d
from dmlab2d import runfiles_helper
from meltingpot import substrate
from meltingpot.utils.puppeteers import in_the_matrix
from meltingpot.utils.substrates import builder
import numpy as np


class InteractionInitialStateTest(absltest.TestCase):

  def test_native_lua_regressions(self):
    root = Path(__file__).resolve().parents[3]
    lab = dmlab2d.Lab2d(
        runfiles_helper.find(),
        {
            'levelDirectory': str(root),
            'levelName': (
                'meltingpot/lua/levels/the_matrix/interaction_initial_state_test'
            ),
        },
    )
    self.assertEmpty(lab.observation_names())


class NativeMatrixInitialObservationTest(parameterized.TestCase):

  @parameterized.parameters(
      'running_with_scissors_in_the_matrix__repeated',
      'running_with_scissors_in_the_matrix__one_shot',
      'running_with_scissors_in_the_matrix__arena',
      'prisoners_dilemma_in_the_matrix__repeated',
      'pure_coordination_in_the_matrix__arena',
  )
  def test_new_episodes_do_not_report_or_count_an_interaction(self, name):
    config = substrate.get_config(name)
    settings = config.lab2d_settings_builder(
        roles=config.default_player_roles, config=config
    )
    settings['maxEpisodeLengthFrames'] = 2
    targets = tuple(
        in_the_matrix.Resource(
            index=index,
            collect_goal=np.eye(4)[2 * index],
            interact_goal=np.eye(4)[2 * index + 1],
        )
        for index in range(2)
    )
    puppeteer = in_the_matrix.AlternatingSpecialist(
        targets=targets, interactions_per_target=1, margin=1
    )
    with builder.builder(settings, env_seed=17) as env:
      actions = {
          key: np.zeros(spec.shape, spec.dtype)
          for key, spec in env.action_spec().items()
      }
      for _ in range(2):
        timestep = env.reset()
        self.assertTrue(timestep.first())
        for index in range(1, len(config.default_player_roles) + 1):
          inventory = timestep.observation[f'{index}.INTERACTION_INVENTORIES']
          self.assertEqual(inventory.dtype, np.float64)
          np.testing.assert_array_equal(inventory, -np.ones_like(inventory))
          single = timestep._replace(
              observation={
                  'INVENTORY': timestep.observation[f'{index}.INVENTORY'],
                  'INTERACTION_INVENTORIES': inventory,
              }
          )
          self.assertFalse(in_the_matrix.has_interaction(single))
          _, state = puppeteer.step(single, puppeteer.initial_state())
          self.assertEqual(state, 0)
        timestep = env.step(actions)
        self.assertTrue(timestep.mid())
        for index in range(1, len(config.default_player_roles) + 1):
          inventory = timestep.observation[f'{index}.INTERACTION_INVENTORIES']
          np.testing.assert_array_equal(inventory, -np.ones_like(inventory))
        self.assertTrue(env.step(actions).last())


if __name__ == '__main__':
  absltest.main()
