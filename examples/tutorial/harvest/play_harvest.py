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
"""A simple human player for playing the `Harvest` level interactively.

Use `WASD` keys to move the character around. `Q` and `E` to turn.
"""
from absl import app
from absl import flags
from meltingpot.human_players import level_playing_utils

from .configs.environment import harvest as game

FLAGS = flags.FLAGS

flags.DEFINE_integer('screen_width', 800,
                     'Width, in pixels, of the game screen')
flags.DEFINE_integer('screen_height', 600,
                     'Height, in pixels, of the game screen')
flags.DEFINE_integer('frames_per_second', 8, 'Frames per second of the game')
flags.DEFINE_string('observation', 'RGB', 'Name of the observation to render')
flags.DEFINE_bool('verbose', False, 'Whether we want verbose output')
flags.DEFINE_bool('display_text', False,
                  'Whether we to display a debug text message')
flags.DEFINE_string('text_message', 'This page intentionally left blank',
                    'Text to display if `display_text` is `True`')


_ACTION_MAP = {
    'move': level_playing_utils.get_direction_pressed,
    'turn': level_playing_utils.get_turn_pressed,
}


def verbose_fn(unused_timestep, unused_player_index: int) -> None:
  pass


def text_display_fn(unused_timestep, unused_player_index: int) -> str:
  return FLAGS.text_message


def main(argv):
  del argv  # Unused.
  level_playing_utils.run_episode(
      FLAGS.observation,
      {},  # Settings overrides
      _ACTION_MAP,
      game.get_config(),
      level_playing_utils.RenderType.PYGAME,
      FLAGS.screen_width, FLAGS.screen_height, FLAGS.frames_per_second,
      verbose_fn if FLAGS.verbose else None,
      text_display_fn if FLAGS.display_text else None)


if __name__ == '__main__':
  app.run(main)
