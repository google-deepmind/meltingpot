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
"""Regression tests for video writer initialization failures."""

import tempfile
from unittest import mock

from absl.testing import absltest
import dm_env
from meltingpot.utils.evaluation import video_subject
import numpy as np


class VideoSubjectWriterTest(absltest.TestCase):

  def test_failed_writer_open_raises_and_cleans_up(self):
    writer = mock.Mock()
    writer.isOpened.return_value = False
    subject = video_subject.VideoSubject(tempfile.mkdtemp())
    frame = np.zeros((8, 16, 3), dtype=np.uint8)
    timestep = dm_env.restart(observation=[{'WORLD.RGB': frame}])

    with mock.patch.object(
        video_subject.cv2, 'VideoWriter', return_value=writer
    ):
      with self.assertRaisesRegex(RuntimeError, 'open video writer'):
        subject.on_next(timestep)

    writer.release.assert_called_once_with()
    writer.write.assert_not_called()
    self.assertIsNone(subject._writer)
    self.assertIsNone(subject._path)


if __name__ == '__main__':
  absltest.main()
