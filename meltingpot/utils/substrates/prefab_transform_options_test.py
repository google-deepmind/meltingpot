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

"""Map placement retains independent non-positional Transform settings."""

import copy

from absl.testing import absltest
from absl.testing import parameterized
from meltingpot.utils.substrates import game_object_utils


def _prefab(kwargs=None):
  transform = {'component': 'Transform'}
  if kwargs is not None:
    transform['kwargs'] = kwargs
  return {'name': 'item', 'components': [transform]}


def _kwargs(game_object):
  return game_object_utils.get_first_named_component(game_object, 'Transform')[
      'kwargs'
  ]


class PrefabTransformOptionsTest(parameterized.TestCase):

  @parameterized.parameters(
      ('item',),
      ({'type': 'all', 'list': ['item']},),
      ({'type': 'choice', 'list': ['item']},),
  )
  def test_placement_retains_options_without_changing_the_prefab(
      self, descriptor
  ):
    prefab = _prefab({
        'position': (19, 23),
        'orientation': 'S',
        'deferPieceCreation': True,
        'name': 'Transform',
    })
    original = copy.deepcopy(prefab)
    objects = game_object_utils.get_game_objects_from_map(
        '\n.X\nX.\n', {'X': descriptor}, {'item': prefab}
    )
    self.assertLen(objects, 2)
    for obj, position in zip(objects, ((1, 0), (0, 1))):
      self.assertEqual(
          _kwargs(obj),
          {
              'position': position,
              'orientation': 'N',
              'deferPieceCreation': True,
              'name': 'Transform',
          },
      )
    _kwargs(objects[0])['deferPieceCreation'] = False
    self.assertTrue(_kwargs(objects[1])['deferPieceCreation'])
    self.assertEqual(prefab, original)

  @parameterized.parameters((None,), ({},), ({'deferPieceCreation': False},))
  def test_missing_and_default_options_keep_existing_placement(self, kwargs):
    prefab = _prefab(kwargs)
    expected = {} if kwargs is None else dict(kwargs)
    expected.update(position=(0, 0), orientation='N')
    objects = game_object_utils.get_game_objects_from_map(
        'X', {'X': 'item'}, {'item': prefab}
    )
    self.assertEqual(_kwargs(objects[0]), expected)

  def test_nonpositional_nested_values_are_copied_for_each_object(self):
    prefab = _prefab({'custom': {'labels': ['original']}})
    objects = game_object_utils.get_game_objects_from_map(
        'XX', {'X': 'item'}, {'item': prefab}
    )
    _kwargs(objects[0])['custom']['labels'].append('changed')
    self.assertEqual(_kwargs(objects[1])['custom'], {'labels': ['original']})
    self.assertEqual(_kwargs(prefab)['custom'], {'labels': ['original']})


if __name__ == '__main__':
  absltest.main()
