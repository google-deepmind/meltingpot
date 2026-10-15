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
"""Convention-following policies retain historical RGB snapshots."""

from absl.testing import absltest
from absl.testing import parameterized
import dm_env
from meltingpot.utils.policies import policy
from meltingpot.utils.policies import puppet_policy
from meltingpot.utils.puppeteers import allelopathic_harvest
from meltingpot.utils.puppeteers import puppeteer
import numpy as np

_GOALS = puppeteer.puppet_goals(('initial', 'red', 'green', 'blue'))
_COLORS = ((240, 0, 0), (0, 150, 0), (0, 150, 0), (0, 150, 0))


def _follower(window=3):
  return allelopathic_harvest.ConventionFollower(
      initial_goal=_GOALS['initial'],
      preference_goals=tuple(_GOALS[name] for name in ('red', 'green', 'blue')),
      color_threshold=100,
      recency_window=window,
  )


def _timestep(frame, step_type=dm_env.StepType.MID):
  return dm_env.TimeStep(step_type, 0.0, 1.0, {'RGB': frame})


class _GoalPolicy(policy.Policy[int]):
  """Deterministic policy exposing which goal reached the puppet."""

  def initial_state(self):
    return 0

  def step(self, timestep, prev_state):
    return int(np.argmax(timestep.observation['GOAL'])), prev_state + 1

  def close(self):
    pass


class ConventionFrameSnapshotTest(parameterized.TestCase):

  @parameterized.parameters('contiguous', 'strided', 'fortran', 'readonly')
  def test_reusing_rgb_storage_does_not_change_the_recency_window(self, layout):
    storage = np.zeros((2, 4, 3), dtype=np.uint8)
    if layout == 'strided':
      frame = storage[:, ::2, :]
    elif layout == 'fortran':
      storage = np.asfortranarray(storage)
      frame = storage
    elif layout == 'readonly':
      frame = storage.view()
      frame.flags.writeable = False
    else:
      frame = storage
    follower = _follower()
    state = follower.initial_state()
    snapshots = []
    for index, color in enumerate(_COLORS):
      storage[:] = color
      before = frame.copy()
      incoming = _timestep(
          frame, dm_env.StepType.FIRST if index == 0 else dm_env.StepType.MID
      )
      outgoing, state = follower.step(incoming, state)
      snapshots.insert(0, before)
      self.assertLen(state.recent_frames, min(index + 1, 3))
      for remembered, expected in zip(state.recent_frames, snapshots[:3]):
        np.testing.assert_array_equal(remembered, expected)
        self.assertFalse(np.shares_memory(remembered, storage))
      self.assertIs(outgoing.observation['RGB'], frame)
      self.assertEqual(outgoing[:3], incoming[:3])
      np.testing.assert_array_equal(frame, before)
      expected_goal = 'red' if index < 3 else 'green'
      np.testing.assert_array_equal(
          outgoing.observation['GOAL'], _GOALS[expected_goal]
      )

  @parameterized.parameters(1, 3)
  def test_saved_states_remain_usable_after_the_input_frame_changes(
      self, window
  ):
    follower = _follower(window)
    frame = np.full((2, 2, 3), _COLORS[0], dtype=np.uint8)
    _, saved = follower.step(
        _timestep(frame, dm_env.StepType.FIRST), follower.initial_state()
    )
    original = frame.copy()
    frame[:] = _COLORS[1]
    _, next_state = follower.step(_timestep(frame), saved)
    np.testing.assert_array_equal(saved.recent_frames[0], original)
    self.assertEqual(saved.step_count, 1)
    self.assertEqual(next_state.step_count, 2)
    replay, _ = follower.step(_timestep(frame.copy()), saved)
    expected = 'green' if window == 1 else 'red'
    np.testing.assert_array_equal(replay.observation['GOAL'], _GOALS[expected])

  def test_first_timestep_discards_previous_episode_and_accepts_tuple_frames(
      self,
  ):
    follower = _follower()
    state = follower.initial_state()
    for _ in range(5):
      _, state = follower.step(
          _timestep(np.full((1, 1, 3), _COLORS[0], np.uint8)), state
      )
    green = (((0, 150, 0),),)
    outgoing, state = follower.step(
        _timestep(green, dm_env.StepType.FIRST), state
    )
    self.assertEqual(state.step_count, 1)
    self.assertLen(state.recent_frames, 1)
    np.testing.assert_array_equal(state.recent_frames[0], green)
    np.testing.assert_array_equal(outgoing.observation['GOAL'], _GOALS['green'])
    self.assertIs(outgoing.observation['RGB'], green)

  def test_puppet_policy_actions_match_independent_frames_with_a_recycled_buffer(
      self,
  ):
    recycled = puppet_policy.PuppetPolicy(_follower(), _GoalPolicy())
    reference = puppet_policy.PuppetPolicy(_follower(), _GoalPolicy())
    self.addCleanup(recycled.close)
    self.addCleanup(reference.close)
    recycled_state = recycled.initial_state()
    reference_state = reference.initial_state()
    buffer = np.zeros((2, 2, 3), np.uint8)
    actions = []
    for index, color in enumerate(_COLORS):
      buffer[:] = color
      step_type = dm_env.StepType.FIRST if index == 0 else dm_env.StepType.MID
      action, recycled_state = recycled.step(
          _timestep(buffer, step_type), recycled_state
      )
      expected, reference_state = reference.step(
          _timestep(buffer.copy(), step_type), reference_state
      )
      self.assertEqual(action, expected)
      self.assertEqual(recycled_state[1], reference_state[1])
      actions.append(action)
    self.assertEqual(actions, [1, 1, 1, 2])


if __name__ == '__main__':
  absltest.main()
