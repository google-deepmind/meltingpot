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
"""Tests for human-player level utilities."""

from unittest import mock

from absl.testing import absltest
from meltingpot.human_players import level_playing_utils
from ml_collections import config_dict


class RunEpisodePlayerPrefixesTest(absltest.TestCase):

  def test_rejects_prefix_count_mismatch_before_building_environment(self):
    full_config = config_dict.ConfigDict({
        'lab2d_settings': {
            'numPlayers': 2,
        },
    })
    env_builder = mock.Mock()

    with self.assertRaisesRegex(ValueError, 'same length'):
      level_playing_utils.run_episode(
          render_observation='WORLD.RGB',
          config_overrides={},
          action_map={},
          full_config=full_config,
          interactive=level_playing_utils.RenderType.NONE,
          env_builder=env_builder,
          player_prefixes=('red',),
      )

    env_builder.assert_not_called()


if __name__ == '__main__':
  absltest.main()
