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
"""Episode state must be detached before synchronous observers are called."""

from pathlib import Path
import tempfile

from absl.testing import absltest
from absl.testing import parameterized
import cv2
import dm_env
from meltingpot.utils.evaluation import return_subject
from meltingpot.utils.evaluation import video_subject
import numpy as np


def _video_timestep(step_type, value, shape=(8, 16, 3)):
  return dm_env.TimeStep(
      step_type=step_type,
      reward=(0.0,),
      discount=1.0,
      observation=({'WORLD.RGB': np.full(shape, value, np.uint8)},),
  )


def _read_video(path):
  capture = cv2.VideoCapture(path)
  frames = []
  try:
    while True:
      ok, frame = capture.read()
      if not ok:
        return frames
      frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
  finally:
    capture.release()


class ReturnCallbackStateTest(parameterized.TestCase):

  @parameterized.parameters((None,), ((10.0, 20.0),))
  def test_callback_can_start_the_next_episode(self, restart_reward):
    recorder = return_subject.ReturnSubject()
    self.addCleanup(recorder.dispose)
    completed = []

    def on_return(value):
      completed.append(value)
      if len(completed) == 1:
        first = dm_env.restart(())
        recorder.on_next(first._replace(reward=restart_reward))

    recorder.subscribe(on_return)
    recorder.on_next(dm_env.restart(()))
    recorder.on_next(dm_env.termination((1.0, 2.0), ()))
    recorder.on_next(dm_env.transition((3.0, 4.0), ()))
    recorder.on_next(dm_env.termination((5.0, 6.0), ()))
    np.testing.assert_array_equal(completed[0], [1.0, 2.0])
    initial = [0.0, 0.0] if restart_reward is None else restart_reward
    np.testing.assert_array_equal(
        completed[1], np.asarray(initial) + [8.0, 10.0]
    )

  def test_failing_observer_does_not_keep_the_completed_episode_active(self):
    recorder = return_subject.ReturnSubject()
    self.addCleanup(recorder.dispose)

    def fail(value):
      del value
      raise RuntimeError('observer failed')

    subscription = recorder.subscribe(fail)
    recorder.on_next(dm_env.restart(()))
    with self.assertRaisesRegex(RuntimeError, 'observer failed'):
      recorder.on_next(dm_env.termination((2.0,), ()))
    subscription.dispose()
    with self.assertRaisesRegex(ValueError, 'FIRST'):
      recorder.on_next(dm_env.transition((10.0,), ()))
    completed = []
    recorder.subscribe(completed.append)
    recorder.on_next(dm_env.restart(()))
    recorder.on_next(dm_env.termination((3.0,), ()))
    np.testing.assert_array_equal(completed, [[3.0]])

  def test_complete_nested_episode_and_empty_returns_keep_their_values(self):
    recorder = return_subject.ReturnSubject()
    self.addCleanup(recorder.dispose)
    completed = []

    def on_return(value):
      completed.append(value)
      if len(completed) == 1:
        recorder.on_next(dm_env.restart(()))
        recorder.on_next(dm_env.termination((), ()))

    recorder.subscribe(on_return)
    recorder.on_next(dm_env.restart(()))
    recorder.on_next(dm_env.termination((1.5,), ()))
    np.testing.assert_array_equal(completed[0], [1.5])
    self.assertEqual(completed[1].shape, (0,))
    with self.assertRaisesRegex(ValueError, 'FIRST'):
      recorder.on_next(dm_env.transition((1.0,), ()))


class VideoCallbackStateTest(absltest.TestCase):

  def make_recorder(self):
    directory = tempfile.TemporaryDirectory()
    self.addCleanup(directory.cleanup)
    recorder = video_subject.VideoSubject(
        directory.name, extension='avi', codec='png '
    )
    self.addCleanup(recorder.dispose)
    return recorder, Path(directory.name)

  def test_callback_can_open_a_differently_sized_next_recording(self):
    recorder, directory = self.make_recorder()
    completed = []
    next_shape = (12, 20, 3)

    def on_video(path):
      completed.append(path)
      np.testing.assert_array_equal(
          _read_video(path)[-1],
          np.full((8, 16, 3), 30, np.uint8)
          if len(completed) == 1
          else np.full(next_shape, 90, np.uint8),
      )
      if len(completed) == 1:
        recorder.on_next(_video_timestep(dm_env.StepType.FIRST, 50, next_shape))

    recorder.subscribe(on_video)
    recorder.on_next(_video_timestep(dm_env.StepType.FIRST, 10))
    recorder.on_next(_video_timestep(dm_env.StepType.LAST, 30))
    recorder.on_next(_video_timestep(dm_env.StepType.MID, 70, next_shape))
    recorder.on_next(_video_timestep(dm_env.StepType.LAST, 90, next_shape))
    self.assertLen(completed, 2)
    self.assertLen(set(completed), 2)
    self.assertEqual(
        {Path(path) for path in completed}, set(directory.iterdir())
    )
    for path, values, shape in (
        (completed[0], (10, 30), (8, 16, 3)),
        (completed[1], (50, 70, 90), next_shape),
    ):
      expected = [np.full(shape, value, np.uint8) for value in values]
      np.testing.assert_array_equal(_read_video(path), expected)

  def test_failing_observer_leaves_no_released_writer_as_an_active_episode(
      self,
  ):
    recorder, _ = self.make_recorder()
    completed = []

    def fail(path):
      completed.append(path)
      raise RuntimeError('video callback failed')

    subscription = recorder.subscribe(fail)
    recorder.on_next(_video_timestep(dm_env.StepType.FIRST, 15))
    with self.assertRaisesRegex(RuntimeError, 'video callback failed'):
      recorder.on_next(_video_timestep(dm_env.StepType.LAST, 25))
    subscription.dispose()
    self.assertLen(_read_video(completed[0]), 2)
    with self.assertRaisesRegex(ValueError, 'FIRST'):
      recorder.on_next(_video_timestep(dm_env.StepType.MID, 35))
    recorder.subscribe(completed.append)
    recorder.on_next(_video_timestep(dm_env.StepType.FIRST, 45))
    recorder.on_next(_video_timestep(dm_env.StepType.LAST, 55))
    np.testing.assert_array_equal(
        _read_video(completed[1]),
        [np.full((8, 16, 3), x, np.uint8) for x in (45, 55)],
    )


if __name__ == '__main__':
  absltest.main()
