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

"""Run matrix-game regressions against the native Lua components."""

from pathlib import Path

from absl.testing import absltest
import dmlab2d
from dmlab2d import runfiles_helper


class MatrixInteractionTasteRecipientTest(absltest.TestCase):

  def test_native_lua_regressions(self):
    root = Path(__file__).resolve().parents[3]
    lab = dmlab2d.Lab2d(
        runfiles_helper.find(),
        {
            'levelDirectory': str(root),
            'levelName': (
                'meltingpot/lua/levels/the_matrix/interaction_taste_recipient_test'
            ),
        },
    )
    self.assertEmpty(lab.observation_names())


if __name__ == '__main__':
  absltest.main()
