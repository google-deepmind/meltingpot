# Copyright 2023 DeepMind Technologies Limited.
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

import os
import tempfile

from absl.testing import absltest
import cv2
import dm_env
from meltingpot.utils.evaluation import evaluation
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


FRAME_SHAPE = (4, 8)
ZERO = np.zeros(FRAME_SHAPE, np.uint8)
EYE = np.eye(*FRAME_SHAPE, dtype=np.uint8) * 255
RED_EYE = np.stack([EYE, ZERO, ZERO], axis=-1)
GREEN_EYE = np.stack([ZERO, EYE, ZERO], axis=-1)
BLUE_EYE = np.stack([ZERO, ZERO, EYE], axis=-1)


class EvaluationTest(absltest.TestCase):

  def test_video_subject(self):
    video_path = None
    step_written = None

    def save_path(path):
      nonlocal video_path
      video_path = path

    tempdir = tempfile.mkdtemp()
    assert os.path.exists(tempdir)
    # Use lossless compression for test.
    subject = evaluation.VideoSubject(tempdir, extension='avi', codec='png ')
    subject.subscribe(on_next=save_path)

    frames = [RED_EYE, GREEN_EYE, BLUE_EYE]
    for n, timestep in enumerate(_as_timesteps(frames)):
      subject.on_next(timestep)
      if step_written is None and video_path is not None:
        step_written = n

    with self.subTest('video_exists'):
      self.assertTrue(video_path and os.path.exists(video_path))

    with self.subTest('written_on_final_step'):
      self.assertEqual(step_written, 2)

    with self.subTest('contents'):
      written = list(_get_frames(video_path))
      np.testing.assert_equal(written, frames)

  def test_return_subject(self):
    episode_return = None
    step_written = None

    def save_return(ret):
      nonlocal episode_return
      episode_return = ret

    subject = evaluation.ReturnSubject()
    subject.subscribe(on_next=save_return)

    timesteps = [
        dm_env.restart(observation=[{}])._replace(reward=[0, 0]),
        dm_env.transition(observation=[{}], reward=[2, 4]),
        dm_env.termination(observation=[{}], reward=[1, 3]),
    ]
    for n, timestep in enumerate(timesteps):
      subject.on_next(timestep)
      if step_written is None and episode_return is not None:
        step_written = n

    with self.subTest('written_on_final_step'):
      self.assertEqual(step_written, 2)

    with self.subTest('contents'):
      np.testing.assert_equal(episode_return, [3, 7])

if __name__ == '__main__':
  absltest.main()
