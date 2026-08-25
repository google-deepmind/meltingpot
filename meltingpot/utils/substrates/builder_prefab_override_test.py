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
"""Tests for prefab overrides in the substrate builder."""

from absl.testing import absltest
from meltingpot.utils.substrates import builder
from ml_collections import config_dict


class PrefabOverrideTest(absltest.TestCase):

  def test_override_adds_missing_kwargs_mapping(self):
    lab2d_settings = config_dict.ConfigDict({
        'simulation': {
            'prefabs': {
                'item': {
                    'components': [
                        {'component': 'Transform'},
                    ],
                },
            },
        },
    }).unlock()

    builder.apply_prefab_overrides(
        lab2d_settings,
        prefab_overrides={
            'item': {
                'Transform': {
                    'orientation': 'E',
                },
            },
        },
    )

    component = lab2d_settings.simulation.prefabs.item.components[0]
    self.assertEqual(component.kwargs.orientation, 'E')


if __name__ == '__main__':
  absltest.main()
