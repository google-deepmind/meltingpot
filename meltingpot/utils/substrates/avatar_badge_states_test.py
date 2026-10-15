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
"""Checks that badge sprite renaming preserves state configuration semantics."""

import copy

from absl.testing import absltest
from absl.testing import parameterized
from meltingpot.utils.substrates import game_object_utils

_STATE_CONFIGS = (
    [{'state': 'badge', 'sprite': 'Badge'}, {'state': 'waiting'}],
    [{'state': 'waiting'}, {'state': 'badge', 'sprite': 'Badge'}],
    [
        {'state': 'static', 'sprite': 'Static'},
        {'state': 'badge', 'sprite': 'Badge'},
    ],
    [
        {'state': 'badge', 'sprite': 'Badge'},
        {'state': 'alternate', 'sprite': 'Badge'},
        {'state': 'waiting'},
    ],
    [
        {'state': 'badge', 'sprite': 'Badge'},
        {'state': 'static', 'sprite': 'Static'},
        {'state': 'waiting'},
    ],
    [{'state': 'waiting'}, {'state': 'static', 'sprite': 'Static'}],
)


def _prefab(states):
  return {
      'name': 'avatar_badge',
      'components': [
          {
              'component': 'StateManager',
              'kwargs': {
                  'initialState': states[0]['state'],
                  'stateConfigs': copy.deepcopy(states),
              },
          },
          {
              'component': 'Transform',
              'kwargs': {'position': (2, 3), 'orientation': 'N'},
          },
          {
              'component': 'Appearance',
              'kwargs': {
                  'renderMode': 'ascii_shape',
                  'spriteNames': ['Badge', 'Static'],
                  'spriteShapes': ['*', '*'],
                  'palettes': [
                      {'*': (255, 0, 0, 255)},
                      {'*': (0, 0, 255, 255)},
                  ],
                  'noRotates': [False, True],
              },
          },
          {
              'component': 'AvatarConnector',
              'kwargs': {
                  'playerIndex': -1,
                  'aliveState': 'badge',
                  'waitState': 'waiting',
              },
          },
      ],
  }


def _kwargs(game_object, component):
  return game_object_utils.get_first_named_component(game_object, component)[
      'kwargs'
  ]


class BadgeSpriteStatesTest(parameterized.TestCase):

  @parameterized.product(states=_STATE_CONFIGS, num_players=(1, 3))
  def test_only_matching_sprite_references_are_renamed(
      self, states, num_players
  ):
    prefab = _prefab(states)
    original = copy.deepcopy(prefab)
    badges = game_object_utils.build_avatar_badges(
        num_players, prefabs={'avatar_badge': prefab}
    )
    self.assertLen(badges, num_players)
    self.assertEqual(prefab, original)
    for n, badge in enumerate(badges, start=1):
      state_manager = _kwargs(badge, 'StateManager')
      expected_states = copy.deepcopy(states)
      for state in expected_states:
        if state.get('sprite') == 'Badge':
          state['sprite'] = f'Badge{n}'
      self.assertEqual(state_manager['stateConfigs'], expected_states)
      self.assertEqual(state_manager['initialState'], states[0]['state'])
      self.assertEqual(
          _kwargs(badge, 'Appearance')['spriteNames'], [f'Badge{n}', 'Static']
      )
      self.assertEqual(_kwargs(badge, 'Appearance')['spriteShapes'], ['*', '*'])
      self.assertEqual(
          _kwargs(badge, 'Appearance')['palettes'][1],
          _kwargs(original, 'Appearance')['palettes'][1],
      )
      self.assertEqual(_kwargs(badge, 'AvatarConnector')['playerIndex'], n)
      self.assertEqual(
          _kwargs(badge, 'Transform'), _kwargs(original, 'Transform')
      )

  def test_game_object_builder_keeps_badge_and_avatar_states_consistent(self):
    badge = _prefab(_STATE_CONFIGS[1])
    avatar = copy.deepcopy(badge)
    avatar['name'] = 'avatar'
    avatar['components'][-1] = {'component': 'Avatar', 'kwargs': {'index': -1}}
    prefabs = {'avatar': avatar, 'avatar_badge': badge}
    original = copy.deepcopy(prefabs)
    game_objects, avatars = game_object_utils.build_game_objects(
        num_players=2,
        ascii_map='\n.',
        prefabs=prefabs,
        char_prefab_map={},
        use_badges=True,
    )
    self.assertEqual(prefabs, original)
    self.assertLen(game_objects, 2)
    self.assertLen(avatars, 2)
    for badge, avatar in zip(game_objects, avatars):
      self.assertEqual(
          _kwargs(badge, 'StateManager'), _kwargs(avatar, 'StateManager')
      )

  def test_badge_states_are_independent_of_each_other_and_the_prefab(self):
    prefab = _prefab(_STATE_CONFIGS[0])
    original = copy.deepcopy(prefab)
    first, second = game_object_utils.build_avatar_badges(
        2, prefabs={'avatar_badge': prefab}
    )
    _kwargs(first, 'StateManager')['stateConfigs'][0]['sprite'] = 'changed'
    self.assertEqual(prefab, original)
    self.assertEqual(
        _kwargs(second, 'StateManager')['stateConfigs'][0]['sprite'], 'Badge2'
    )

  @parameterized.parameters((None,), ({},))
  def test_missing_badge_prefab_still_raises(self, prefabs):
    with self.assertRaisesRegex(ValueError, 'no avatar_badge prefab'):
      game_object_utils.build_avatar_badges(1, prefabs=prefabs)


if __name__ == '__main__':
  absltest.main()
