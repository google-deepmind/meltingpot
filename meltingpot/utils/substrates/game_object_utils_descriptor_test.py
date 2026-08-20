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
"""Tests for prefab map descriptor validation."""

from absl.testing import absltest
from absl.testing import parameterized
from meltingpot.utils.substrates import game_object_utils


_PREFABS = {
    'item': {
        'components': [
            {
                'component': 'Transform',
                'kwargs': {},
            },
        ],
    },
}


class PrefabMapDescriptorTest(parameterized.TestCase):

  @parameterized.parameters(
      {'type': game_object_utils.TYPE_ALL},
      {'list': ['item']},
  )
  def test_rejects_incomplete_descriptor(self, descriptor):
    with self.assertRaisesRegex(ValueError, "both 'type' and 'list'"):
      game_object_utils.get_game_objects_from_map(
          '\nA', {'A': descriptor}, _PREFABS)

  def test_rejects_unknown_descriptor_type(self):
    descriptor = {'type': 'unknown', 'list': ['item']}

    with self.assertRaisesRegex(ValueError, 'Unknown prefab descriptor type'):
      game_object_utils.get_game_objects_from_map(
          '\nA', {'A': descriptor}, _PREFABS)


if __name__ == '__main__':
  absltest.main()
