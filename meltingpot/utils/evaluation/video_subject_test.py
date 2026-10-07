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

import pathlib
import tempfile
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
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
    return recorder, paths, pathlib.Path(temporary.name)

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
    self.assertEqual(
        {pathlib.Path(path) for path in paths}, set(directory.iterdir())
    )

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
