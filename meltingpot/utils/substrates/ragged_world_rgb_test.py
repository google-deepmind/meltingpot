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
"""WORLD.RGB dimensions follow the full extent of ragged ASCII maps."""

from absl.testing import absltest
from absl.testing import parameterized
from meltingpot import substrate
from meltingpot.utils.substrates import builder
from meltingpot.utils.substrates import specs
import numpy as np


class RaggedWorldRgbTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('longer_last', 'A\nBBB', 2, 3),
      ('longer_middle', '\nA\nBBBB\nCC\n', 3, 4),
      ('longer_first', '\nAAAA\nBB\nC\n', 3, 4),
      ('boundary_spaces', '\n A\n B   \n', 2, 5),
      ('blank_interior_row', 'A\n\nBBB', 3, 3),
      ('rectangular', '\nAB\nCD\n', 2, 2),
      ('single_row', ' ABC ', 1, 5),
  )
  def test_dimensions_use_the_longest_row_without_stripping_cells(
      self, ascii_map, height, width
  ):
    for sprite_size in (1, 4, 8):
      with self.subTest(sprite_size=sprite_size):
        actual = specs.world_rgb(ascii_map, sprite_size, name='world')
        self.assertEqual(
            actual.shape, (height * sprite_size, width * sprite_size, 3)
        )
        self.assertEqual(actual.dtype, np.dtype(np.uint8))
        self.assertEqual(actual.name, 'world')
        actual.validate(np.zeros(actual.shape, np.uint8))

  @parameterized.parameters('', '\n', '\n\n')
  def test_empty_map_convention_remains_unchanged(self, ascii_map):
    self.assertEqual(specs.world_rgb(ascii_map, 8).shape, (8, 0, 3))
    self.assertEqual(specs.world_rgb(ascii_map, 0).shape, (0, 0, 3))

  @parameterized.parameters(False, True)
  def test_spec_matches_native_lab2d_frames_for_ragged_maps(self, extend_last):
    config = substrate.get_config('commons_harvest__open')
    settings = config.lab2d_settings_builder(
        roles=config.default_player_roles, config=config
    )
    rows = settings['simulation']['map'].strip('\n').split('\n')
    rows[0] = rows[0][:-5]
    if extend_last:
      rows[-1] += 'WWWWW'
    ascii_map = '\n' + '\n'.join(rows) + '\n'
    settings['simulation']['map'] = ascii_map
    settings['maxEpisodeLengthFrames'] = 2
    expected = specs.world_rgb(ascii_map, sprite_size=8)
    with builder.builder(settings, env_seed=31) as environment:
      self.assertEqual(
          expected.shape, environment.observation_spec()['WORLD.RGB'].shape
      )
      actions = {
          name: np.zeros(spec.shape, spec.dtype)
          for name, spec in environment.action_spec().items()
      }
      for _ in range(2):
        timestep = environment.reset()
        expected.validate(timestep.observation['WORLD.RGB'])
        while not timestep.last():
          timestep = environment.step(actions)
          expected.validate(timestep.observation['WORLD.RGB'])
    self.assertEqual(settings['simulation']['map'], ascii_map)


if __name__ == '__main__':
  absltest.main()
