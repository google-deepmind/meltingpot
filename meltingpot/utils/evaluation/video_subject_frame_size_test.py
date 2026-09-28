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
"""Video frame-size checks with real lossless encoding and decoding."""

from pathlib import Path
import tempfile

from absl.testing import absltest
from absl.testing import parameterized
import cv2
import dm_env
from meltingpot.utils.evaluation import video_subject
import numpy as np


def _frame(height=8, width=16, shift=0):
  values = np.arange(height * width * 3).reshape(height, width, 3)
  return ((values + shift) % 256).astype(np.uint8)


def _timestep(step_type, frame):
  return dm_env.TimeStep(
      step_type=step_type,
      reward=0.0,
      discount=1.0,
      observation=[{'WORLD.RGB': frame}],
  )


def _read_frames(path):
  capture = cv2.VideoCapture(path)
  try:
    frames = []
    while True:
      success, frame = capture.read()
      if not success:
        return frames
      frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
  finally:
    capture.release()


class VideoFrameSizeTest(parameterized.TestCase):

  def make_recorder(self):
    temporary = tempfile.TemporaryDirectory()
    self.addCleanup(temporary.cleanup)
    recorder = video_subject.VideoSubject(
        temporary.name, extension='avi', codec='png '
    )
    self.addCleanup(recorder.dispose)
    paths = []
    recorder.subscribe(paths.append)
    return recorder, paths, Path(temporary.name)

  @parameterized.product(
      shape=((10, 16), (8, 18), (16, 8)),
      step_type=(dm_env.StepType.MID, dm_env.StepType.LAST),
  )
  def test_mismatched_frames_raise_without_emitting_or_losing_the_recording(
      self, shape, step_type
  ):
    recorder, paths, directory = self.make_recorder()
    first, middle, last = _frame(), _frame(shift=23), _frame(shift=51)
    recorder.on_next(_timestep(dm_env.StepType.FIRST, first))
    bad_frame = _frame(*shape)
    original = bad_frame.copy()
    with self.assertRaisesRegex(ValueError, 'WORLD.RGB.*size'):
      recorder.on_next(_timestep(step_type, bad_frame))
    self.assertEmpty(paths)
    self.assertLen(list(directory.iterdir()), 1)
    np.testing.assert_array_equal(bad_frame, original)
    recorder.on_next(_timestep(dm_env.StepType.MID, middle))
    recorder.on_next(_timestep(dm_env.StepType.LAST, last))
    self.assertLen(paths, 1)
    actual = _read_frames(paths[0])
    self.assertLen(actual, 3)
    np.testing.assert_array_equal(actual, [first, middle, last])

  def test_frame_size_can_change_between_complete_episodes(self):
    recorder, paths, directory = self.make_recorder()
    for height, width in ((8, 16), (12, 20), (16, 8)):
      expected = [_frame(height, width, shift) for shift in (0, 11, 99)]
      for step_type, frame in zip(dm_env.StepType, expected):
        recorder.on_next(_timestep(step_type, frame))
      np.testing.assert_array_equal(_read_frames(paths[-1]), expected)
    self.assertLen(paths, 3)
    self.assertLen(set(paths), 3)
    self.assertEqual({Path(path) for path in paths}, set(directory.iterdir()))

  @parameterized.parameters('strided', 'fortran')
  def test_valid_noncontiguous_frames_keep_their_pixels(self, layout):
    recorder, paths, _ = self.make_recorder()
    if layout == 'strided':
      frame = _frame(8, 32)[:, ::2]
    else:
      frame = np.asfortranarray(_frame())
    self.assertFalse(frame.flags.c_contiguous)
    original = frame.copy()
    recorder.on_next(_timestep(dm_env.StepType.FIRST, frame))
    recorder.on_next(_timestep(dm_env.StepType.LAST, frame))
    np.testing.assert_array_equal(_read_frames(paths[0]), [original, original])
    np.testing.assert_array_equal(frame, original)

  @parameterized.parameters(dm_env.StepType.MID, dm_env.StepType.LAST)
  def test_frames_before_first_keep_the_existing_error(self, step_type):
    recorder, paths, directory = self.make_recorder()
    with self.assertRaisesRegex(ValueError, 'First timestep'):
      recorder.on_next(_timestep(step_type, _frame()))
    self.assertEmpty(paths)
    self.assertEmpty(list(directory.iterdir()))

  @parameterized.parameters('channels', 'dtype')
  def test_existing_frame_validation_remains_active(self, invalid):
    recorder, paths, _ = self.make_recorder()
    frame = _frame()
    bad_frame = (
        frame[:, :, :2] if invalid == 'channels' else frame.astype(float)
    )
    with self.assertRaises(ValueError):
      recorder.on_next(_timestep(dm_env.StepType.FIRST, bad_frame))
    self.assertEmpty(paths)
    recorder.on_next(_timestep(dm_env.StepType.FIRST, frame))
    recorder.on_next(_timestep(dm_env.StepType.LAST, frame))
    np.testing.assert_array_equal(_read_frames(paths[0]), [frame, frame])


if __name__ == '__main__':
  absltest.main()
