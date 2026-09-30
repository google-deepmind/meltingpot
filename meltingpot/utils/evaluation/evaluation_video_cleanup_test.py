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

"""Verify video cleanup through the real evaluation loop and OpenCV writer."""

import pathlib
import tempfile
from types import SimpleNamespace
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import cv2
import dm_env
from meltingpot.utils.evaluation import evaluation
from meltingpot.utils.evaluation import video_subject
from meltingpot.utils.policies import policy
from meltingpot.utils.scenarios import population
import numpy as np
from reactivex import subject


class CountingPolicy(policy.Policy):

  def __init__(self, error=None):
    self.error = error

  def initial_state(self):
    return 0

  def step(self, timestep, prev_state):
    del timestep
    if self.error is not None:
      raise self.error
    return 0, prev_state + 1

  def close(self):
    pass


class EvaluationVideoCleanupTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    directory = tempfile.TemporaryDirectory()
    self.addCleanup(directory.cleanup)
    self.root = pathlib.Path(directory.name)
    self.frames = [
        np.full((8, 16, 3), value, np.uint8) for value in (25, 80, 150)
    ]
    self.source = subject.Subject()
    self.recorders, self.writers = [], []
    make_writer = cv2.VideoWriter
    make_subject = video_subject.VideoSubject

    def track_writer(*args, **kwargs):
      writer = make_writer(*args, **kwargs)
      self.writers.append(writer)
      self.addCleanup(writer.release)
      return writer

    def track_subject(root):
      recorder = make_subject(root, extension='avi', codec='png ')
      original_dispose = recorder.dispose

      def dispose():
        # Source subscriptions must be detached before disposing their observer.
        self.assertEmpty(self.source.observers)
        original_dispose()

      recorder.dispose = mock.Mock(side_effect=dispose)
      self.recorders.append(recorder)
      self.addCleanup(original_dispose)
      return recorder

    self.enter_context(
        mock.patch.object(cv2, 'VideoWriter', side_effect=track_writer)
    )
    self.enter_context(
        mock.patch.object(
            video_subject, 'VideoSubject', side_effect=track_subject
        )
    )

  def make_environment(self, fail_after=None, invalid_frame=False):
    error = RuntimeError('synthetic substrate failure')
    env = mock.Mock()
    env.observables.return_value = SimpleNamespace(timestep=self.source)
    self.step_count = 0

    def emit(step_type, index):
      frame = self.frames[index]
      if invalid_frame and index == 1:
        frame = frame.astype(np.float32)
      timestep = dm_env.TimeStep(
          step_type=step_type,
          reward=(float(index),),
          discount=0.0 if step_type.last() else 1.0,
          observation=({'WORLD.RGB': frame},),
      )
      self.source.on_next(timestep)
      return timestep

    def reset():
      self.step_count = 0
      return emit(dm_env.StepType.FIRST, 0)

    def step(actions):
      self.assertEqual(actions, (0,))
      if self.step_count == fail_after:
        raise error
      self.step_count += 1
      return emit(
          dm_env.StepType.LAST if self.step_count == 2 else dm_env.StepType.MID,
          self.step_count,
      )

    env.reset.side_effect, env.step.side_effect = reset, step
    return env, error

  def make_population(self, error=None):
    group = population.Population(
        policies={'bot': CountingPolicy(error)},
        names_by_role={'role': ['bot']},
        roles=['role'],
    )
    self.addCleanup(group.close)
    return group

  def run_evaluation(self, env, group, count=1, record=True):
    # Keep the existing empty-background per-capita calculation unchanged.
    with np.errstate(invalid='ignore'):
      return evaluation.run_and_observe_episodes(
          group, env, count, str(self.root) if record else None
      )

  def assert_released(self):
    self.assertLen(self.recorders, 1)
    self.assertTrue(all(not writer.isOpened() for writer in self.writers))
    self.recorders[0].dispose.assert_called_once_with()
    self.assertTrue(self.recorders[0].is_disposed)
    self.assertEmpty(self.source.observers)

  @parameterized.parameters(0, 1)
  def test_substrate_failure_finalizes_an_incomplete_recording(
      self, fail_after
  ):
    env, error = self.make_environment(fail_after=fail_after)
    with self.assertRaises(RuntimeError) as caught:
      self.run_evaluation(env, self.make_population())
    self.assertIs(caught.exception, error)
    self.assertLen(self.writers, 1)
    self.assert_released()
    paths = list(self.root.glob('*.avi'))
    self.assertLen(paths, 1)
    capture = cv2.VideoCapture(str(paths[0]))
    try:
      frames = []
      while True:
        ok, frame = capture.read()
        if not ok:
          break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
      capture.release()
    np.testing.assert_array_equal(frames, self.frames[: fail_after + 1])

  def test_policy_failure_finalizes_the_recording_without_masking_error(self):
    env, _ = self.make_environment()
    error = ValueError('synthetic policy failure')
    with self.assertRaises(ValueError) as caught:
      self.run_evaluation(env, self.make_population(error))
    self.assertIs(caught.exception, error)
    self.assertLen(self.writers, 1)
    self.assert_released()

  def test_invalid_later_frame_releases_the_active_writer(self):
    env, _ = self.make_environment(invalid_frame=True)
    with self.assertRaisesRegex(ValueError, 'uint8'):
      self.run_evaluation(env, self.make_population())
    self.assertLen(self.writers, 1)
    self.assert_released()

  def test_successful_episodes_keep_returns_and_complete_video_contents(self):
    env, _ = self.make_environment()
    result = self.run_evaluation(env, self.make_population(), count=2)
    self.assertLen(result, 2)
    np.testing.assert_array_equal(
        np.stack(result.focal_player_returns), [[3.0], [3.0]]
    )
    self.assertLen(self.writers, 2)
    self.assert_released()
    for path in result.video_path:
      capture = cv2.VideoCapture(path)
      try:
        decoded = []
        while True:
          ok, frame = capture.read()
          if not ok:
            break
          decoded.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
      finally:
        capture.release()
      np.testing.assert_array_equal(decoded, self.frames)

  def test_zero_episodes_dispose_the_unused_recorder(self):
    env, _ = self.make_environment()
    result = self.run_evaluation(env, self.make_population(), count=0)
    self.assertTrue(result.empty)
    self.assertEmpty(self.writers)
    self.assert_released()
    env.reset.assert_not_called()

  def test_video_disabled_does_not_create_a_recorder(self):
    env, _ = self.make_environment()
    result = self.run_evaluation(env, self.make_population(), record=False)
    self.assertNotIn('video_path', result)
    self.assertEmpty(self.recorders)
    self.assertEmpty(self.writers)
    self.assertEmpty(list(self.root.iterdir()))


if __name__ == '__main__':
  absltest.main()
