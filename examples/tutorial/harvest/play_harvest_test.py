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
"""Regression tests for the Harvest tutorial player."""

import pathlib
import runpy
import sys
from unittest import mock

from absl.testing import absltest


class PlayHarvestTest(absltest.TestCase):

  def test_standalone_import_and_verbose_callback(self):
    script = pathlib.Path(__file__).with_name('play_harvest.py')
    script_dir = str(script.parent)

    with mock.patch.object(sys, 'path', [script_dir, *sys.path]):
      namespace = runpy.run_path(str(script), run_name='harvest_tutorial_test')

    namespace['verbose_fn'](None, 0, 0)


if __name__ == '__main__':
  absltest.main()
