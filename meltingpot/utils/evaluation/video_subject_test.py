# Copyright 2024 DeepMind Technologies Limited.
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

import tempfile

from absl.testing import absltest
import cv2
import dm_env
from meltingpot.utils.evaluation import video_subject
import numpy as np


def _as_timesteps(frames):
  first, *mids, last = frames
  yield dm_env.restart(observation=[{'WORLD.RGB': first}])
  for frame in mids:
    yield dm_env.transition(observation=[{'WORLD.RGB': frame}], reward=0)
  yield dm_env.termination(observation=[{'WORLD.RGB': last}], reward=0)


def _get_frames(path):
  capture = cv2.VideoCapture(path)
  while capture.isOpened():
    ret, bgr_frame = capture.read()
    if not ret:
      break
    rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    yield rgb_frame
  capture.release()


def _write_frames_to_subject(subject, frames):
  results = []
  subject.subscribe(on_next=results.append)

  timesteps = _as_timesteps(frames)
  for n, timestep in enumerate(timesteps):
    subject.on_next(timestep)
    if results:
      return n, results.pop()
  return None, None


FRAME_SHAPE = (8, 16)
ONES = np.zeros(FRAME_SHAPE, np.uint8)
ZERO = np.zeros(FRAME_SHAPE, np.uint8)
EYE = np.eye(*FRAME_SHAPE, dtype=np.uint8) * 255
RED_EYE = np.stack([EYE, ZERO, ZERO], axis=-1)
GREEN_EYE = np.stack([ZERO, EYE, ZERO], axis=-1)
BLUE_EYE = np.stack([ZERO, ZERO, EYE], axis=-1)
TEST_FRAMES = np.stack([RED_EYE, GREEN_EYE, BLUE_EYE], axis=0)


class VideoSubjectTest(absltest.TestCase):

  def test_lossless_writes_correct_frames(self):
    # Use lossless compression for equality test.
    subject = video_subject.VideoSubject(
        root=tempfile.mkdtemp(), extension='avi', codec='png '
    )
    step_written, video_path = _write_frames_to_subject(subject, TEST_FRAMES)
    frames_written = np.stack(list(_get_frames(video_path)), axis=0)

    with self.subTest('written_on_final_step'):
      self.assertEqual(step_written, TEST_FRAMES.shape[0] - 1)

    with self.subTest('contents'):
      np.testing.assert_equal(frames_written, TEST_FRAMES)

  def test_default_writes_correct_shape(self):
    subject = video_subject.VideoSubject(tempfile.mkdtemp())
    step_written, video_path = _write_frames_to_subject(subject, TEST_FRAMES)
    frames_written = np.stack(list(_get_frames(video_path)), axis=0)

    with self.subTest('written_on_final_step'):
      self.assertEqual(step_written, TEST_FRAMES.shape[0] - 1)

    with self.subTest('shape'):
      self.assertEqual(frames_written.shape, TEST_FRAMES.shape)

if __name__ == '__main__':
  absltest.main()
