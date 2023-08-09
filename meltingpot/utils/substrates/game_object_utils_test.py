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
"""Tests for game_object_utils."""

from absl.testing import absltest
from absl.testing import parameterized
from meltingpot.utils.substrates import game_object_utils


def get_transform(x, y, orientation):
  return game_object_utils.Transform(position=game_object_utils.Position(x, y),
                                     orientation=orientation)


class ParseMapTest(parameterized.TestCase):

  @parameterized.parameters(
      ('\nHello', 'H', 1),
      ('\nHello', 'h', 0),
      ('\nHello', 'l', 2),
      ('\nHello\nWorld', 'l', 3),
      ('\nHello\nWorld', 'o', 2),
      ('\nHello\nWorld', 'd', 1),
      ('\nHello\nWorld', 'W', 1),
      ('\nWWWW\nW AW\nWWWW', 'A', 1),
      ('\nWWWW\nW AW\nWWWW', 'W', 10),
      ('\nWWWW\nW AW\nWWWW', 'P', 0),
      )
  def test_get_positions_length(self, ascii_map, char, exp_len):
    transforms = game_object_utils.get_game_object_positions_from_map(
        ascii_map, char)
    self.assertLen(transforms, exp_len)

  def test_get_positions(self):
    # Locations of 'A' -> (2, 1)
    # Locations of ' ' -> (1, 1), (3, 1) and (4, 1)
    ascii_map = '''
WWWWWW
W A  W
WWWWWW
'''
    transforms = game_object_utils.get_game_object_positions_from_map(
        ascii_map, 'A')
    self.assertSameElements(
        [get_transform(2, 1, game_object_utils.Orientation.NORTH)], transforms)

    transforms = game_object_utils.get_game_object_positions_from_map(
        ascii_map, ' ')
    self.assertSameElements(
        [
            get_transform(1, 1, game_object_utils.Orientation.NORTH),
            get_transform(3, 1, game_object_utils.Orientation.NORTH),
            get_transform(4, 1, game_object_utils.Orientation.NORTH)
        ],
        transforms)

    transforms = game_object_utils.get_game_object_positions_from_map(
        ascii_map, 'W')
    self.assertSameElements(
        [
            # Top walls
            get_transform(0, 0, game_object_utils.Orientation.NORTH),
            get_transform(1, 0, game_object_utils.Orientation.NORTH),
            get_transform(2, 0, game_object_utils.Orientation.NORTH),
            get_transform(3, 0, game_object_utils.Orientation.NORTH),
            get_transform(4, 0, game_object_utils.Orientation.NORTH),
            get_transform(5, 0, game_object_utils.Orientation.NORTH),

            # Side walls
            get_transform(0, 1, game_object_utils.Orientation.NORTH),
            get_transform(5, 1, game_object_utils.Orientation.NORTH),

            # Bottom walls
            get_transform(0, 2, game_object_utils.Orientation.NORTH),
            get_transform(1, 2, game_object_utils.Orientation.NORTH),
            get_transform(2, 2, game_object_utils.Orientation.NORTH),
            get_transform(3, 2, game_object_utils.Orientation.NORTH),
            get_transform(4, 2, game_object_utils.Orientation.NORTH),
            get_transform(5, 2, game_object_utils.Orientation.NORTH),
        ],
        transforms)

  def test_get_game_objects(self):
    ascii_map = '''
WWWWWW
W A  W
WWWWWW
'''
    wall = {
        'name': 'wall',
        'components': [
            {
                'component': 'PieceTypeManager',
                'kwargs': {
                    'initialPieceType': 'wall',
                    'pieceTypeConfigs': [{'pieceType': 'wall',}],
                },
            },
            {
                'component': 'Transform',
                'kwargs': {
                    'position': (0, 0),
                    'orientation': 'N'
                },
            },
        ]
    }
    apple = {
        'name': 'apple',
        'components': [
            {
                'component': 'PieceTypeManager',
                'kwargs': {
                    'initialPieceType': 'apple',
                    'pieceTypeConfigs': [{'pieceType': 'apple',}],
                },
            },
            {
                'component': 'Transform',
                'kwargs': {
                    'position': (0, 0),
                    'orientation': 'N'
                },
            },
        ]
    }
    prefabs = {'wall': wall, 'apple': apple}
    game_objects = game_object_utils.get_game_objects_from_map(
        ascii_map, {'W': 'wall', 'A': 'apple'}, prefabs)
    self.assertLen(game_objects, 15)
    self.assertEqual(
        1,
        sum([1 if go['name'] == 'apple' else 0 for go in game_objects]))
    self.assertEqual(
        14,
        sum([1 if go['name'] == 'wall' else 0 for go in game_objects]))

    positions = []
    for go in game_objects:
      if go['name'] == 'wall':
        positions.append(game_object_utils.get_first_named_component(
            go, 'Transform')['kwargs']['position'])

    self.assertSameElements(
        [
            (0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0),  # Top walls
            (0, 1), (5, 1),  # Side walls
            (0, 2), (1, 2), (2, 2), (3, 2), (4, 2), (5, 2),  # Bottom walls
        ],
        positions)


AVATAR = {
    'name': 'avatar',
    'components': [
        {
            'component': 'StateManager',
            'kwargs': {
                'initialState': 'player',
                'stateConfigs': [
                    {'state': 'player',
                     'layer': 'upperPhysical',
                     'sprite': 'Avatar',},  # Will be overridden

                    {'state': 'playerWait',},
                ]
            }
        },
        {
            'component': 'Transform',
            'kwargs': {
                'position': (0, 0),
                'orientation': 'N'
            }
        },
        {
            'component': 'Appearance',
            'kwargs': {
                'renderMode': 'ascii_shape',
                'spriteNames': ['Avatar'],  # Will be overridden
                'spriteShapes': ["""*"""],
                'palettes': [(0, 0, 255, 255)],  # Will be overridden
                'noRotates': [True]
            }
        },
        {
            'component': 'Avatar',
            'kwargs': {
                'index': -1,  # Will be overridden
                'spawnGroup': 'spawnPoints',
                'aliveState': 'player',
                'waitState': 'playerWait',
                'actionOrder': ['move'],
                'actionSpec': {
                    'move': {'default': 0, 'min': 0, 'max': 4},
                },
            }
        },
    ],
}


BADGE = {
    'name': 'avatar_badge',
    'components': [
        {
            'component': 'StateManager',
            'kwargs': {
                'initialState': 'badgeWait',
                'stateConfigs': [
                    {'state': 'badge',
                     'layer': 'overlay',
                     'sprite': 'Badge',
                     'groups': ['badges']},

                    {'state': 'badgeWait',
                     'groups': ['badgeWaits']},
                ]
            }
        },
        {
            'component': 'Transform',
            'kwargs': {
                'position': (0, 0),
                'orientation': 'N'
            }
        },
        {
            'component': 'Appearance',
            'kwargs': {
                'renderMode': 'ascii_shape',
                'spriteNames': ['Badge'],
                'spriteShapes': ['*'],
                'palettes': [(0, 0, 255, 255)],
                'noRotates': [False]
            }
        },
        {
            'component': 'AvatarConnector',
            'kwargs': {
                'playerIndex': -1,  # player index to be overwritten.
                'aliveState': 'badge',
                'waitState': 'badgeWait'
            }
        },
    ]
}


class BuildAvatarObjectsTest(parameterized.TestCase):

  @parameterized.parameters(
      [1], [2], [3], [4], [5]
      )
  def test_simple_build(self, num_players):
    prefabs = {'avatar': AVATAR}
    avatars = game_object_utils.build_avatar_objects(
        num_players=num_players,
        prefabs=prefabs,
        player_palettes=None,
        )
    self.assertLen(avatars, num_players)

  def test_with_palette_build(self):
    palettes = [(255, 0, 0, 255), (0, 255, 0, 255)]
    prefabs = {'avatar': AVATAR}
    avatars = game_object_utils.build_avatar_objects(
        num_players=2,
        prefabs=prefabs,
        player_palettes=palettes,
        )
    self.assertLen(avatars, 2)
    self.assertEqual(
        game_object_utils.get_first_named_component(
            avatars[0], 'Appearance')['kwargs']['palettes'][0],
        palettes[0])
    self.assertEqual(
        game_object_utils.get_first_named_component(
            avatars[1], 'Appearance')['kwargs']['palettes'][0],
        palettes[1])


class BuildAvatarBadgesTest(parameterized.TestCase):

  @parameterized.parameters(
      [1], [2], [3], [4], [5]
      )
  def test_simple_build(self, num_players):
    prefabs = {'avatar_badge': BADGE}
    badges = game_object_utils.build_avatar_badges(
        num_players=num_players,
        prefabs=prefabs,
        badge_palettes=None,
        )
    self.assertLen(badges, num_players)

  def test_with_palette_build(self):
    palettes = [(255, 0, 0, 255), (0, 255, 0, 255)]
    prefabs = {'avatar_badge': BADGE}
    badges = game_object_utils.build_avatar_badges(
        num_players=2,
        prefabs=prefabs,
        badge_palettes=palettes,
        )
    self.assertLen(badges, 2)
    self.assertEqual(
        game_object_utils.get_first_named_component(
            badges[0], 'Appearance')['kwargs']['palettes'][0],
        palettes[0])
    self.assertEqual(
        game_object_utils.get_first_named_component(
            badges[1], 'Appearance')['kwargs']['palettes'][0],
        palettes[1])


if __name__ == '__main__':
  absltest.main()
