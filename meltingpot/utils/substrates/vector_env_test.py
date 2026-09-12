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
"""Tests for subprocess vectorized environments."""

import dataclasses

from absl.testing import absltest
import dm_env
from meltingpot.utils.substrates import vector_env
import numpy as np


_ACTION_SPEC = dm_env.specs.DiscreteArray(
    num_values=10, dtype=np.int64, name="action"
)
_OBSERVATION_SPEC = {
    "VALUE": dm_env.specs.Array(shape=(), dtype=np.int64, name="VALUE")
}
_REWARD_SPEC = dm_env.specs.Array(shape=(), dtype=np.float64, name="reward")
_DISCOUNT_SPEC = dm_env.specs.BoundedArray(
    shape=(), dtype=np.float64, minimum=0, maximum=1, name="discount"
)


class _FakeEnv:
  """Small picklable environment used to exercise process boundaries."""

  def __init__(self, initial_value: int) -> None:
    self._initial_value = initial_value
    self._value = initial_value

  def _observation(self):
    return (
        {"VALUE": np.asarray(self._value, dtype=np.int64)},
        {"VALUE": np.asarray(self._value + 1, dtype=np.int64)},
    )

  def reset(self) -> dm_env.TimeStep:
    self._value = self._initial_value
    return dm_env.TimeStep(
        step_type=dm_env.StepType.FIRST,
        reward=(0.0, 0.0),
        discount=1.0,
        observation=self._observation(),
    )

  def step(self, action) -> dm_env.TimeStep:
    if any(value < 0 for value in action):
      raise ValueError("negative actions are invalid")
    self._value += sum(action)
    return dm_env.TimeStep(
        step_type=dm_env.StepType.MID,
        reward=(float(self._value), float(self._value)),
        discount=1.0,
        observation=self._observation(),
    )

  def observation(self):
    return self._observation()

  def events(self):
    return (("value", self._value),)

  def action_spec(self):
    return (_ACTION_SPEC, _ACTION_SPEC)

  def observation_spec(self):
    return (_OBSERVATION_SPEC, _OBSERVATION_SPEC)

  def reward_spec(self):
    return (_REWARD_SPEC, _REWARD_SPEC)

  def discount_spec(self):
    return _DISCOUNT_SPEC

  def close(self) -> None:
    pass


@dataclasses.dataclass(frozen=True)
class _FakeBuilder:
  initial_value: int

  def __call__(self):
    return _FakeEnv(self.initial_value)


@dataclasses.dataclass(frozen=True)
class _FailingBuilder:

  def __call__(self):
    raise RuntimeError("builder failed")


def _value(timestep: dm_env.TimeStep) -> int:
  return int(timestep.observation[0]["VALUE"])


class SubprocessVectorEnvTest(absltest.TestCase):

  def test_requires_at_least_one_builder(self):
    with self.assertRaisesRegex(ValueError, "At least one"):
      vector_env.SubprocessVectorEnv(())

  def test_runs_workers_and_resets_independently(self):
    with vector_env.SubprocessVectorEnv(
        (_FakeBuilder(10), _FakeBuilder(20)), start_method="spawn"
    ) as env:
      self.assertEqual(env.num_envs, 2)
      self.assertEqual(env.start_method, "spawn")

      reset = env.reset()
      self.assertEqual([_value(timestep) for timestep in reset], [10, 20])

      stepped = env.step(((1, 2), (3, 4)))
      self.assertEqual([_value(timestep) for timestep in stepped], [13, 27])

      reset_second = env.reset(indices=(1,))
      self.assertLen(reset_second, 1)
      self.assertEqual(_value(reset_second[0]), 20)

      observations = env.observation()
      self.assertEqual(int(observations[0][0]["VALUE"]), 13)
      self.assertEqual(int(observations[1][0]["VALUE"]), 20)
      self.assertEqual(env.events(), ((('value', 13),), (('value', 20),)))

  def test_exposes_member_environment_specs(self):
    with vector_env.SubprocessVectorEnv(
        (_FakeBuilder(0), _FakeBuilder(1)), start_method="spawn"
    ) as env:
      action_spec = env.action_spec()
      self.assertLen(action_spec, 2)
      self.assertEqual(action_spec[0].num_values, 10)
      self.assertEqual(action_spec[0].dtype, np.dtype(np.int64))

      observation_spec = env.observation_spec()
      self.assertLen(observation_spec, 2)
      self.assertEqual(observation_spec[0]["VALUE"].shape, ())
      self.assertEqual(
          observation_spec[0]["VALUE"].dtype, np.dtype(np.int64)
      )

      reward_spec = env.reward_spec()
      self.assertLen(reward_spec, 2)
      self.assertEqual(reward_spec[0].dtype, np.dtype(np.float64))

      discount_spec = env.discount_spec()
      self.assertEqual(discount_spec.minimum, 0)
      self.assertEqual(discount_spec.maximum, 1)

  def test_worker_error_reports_index_and_keeps_protocol_synchronized(self):
    with vector_env.SubprocessVectorEnv(
        (_FakeBuilder(0), _FakeBuilder(100)), start_method="spawn"
    ) as env:
      env.reset()
      with self.assertRaisesRegex(
          vector_env.VectorEnvWorkerError,
          "Worker 1 raised ValueError: negative actions are invalid",
      ):
        env.step(((1, 1), (-1, 0)))

      # The successful worker response from the failed batch was drained, so a
      # subsequent request still receives the correct response from each worker.
      stepped = env.step(((1, 0), (2, 0)))
      self.assertEqual([_value(timestep) for timestep in stepped], [3, 102])

  def test_rejects_wrong_action_batch_size_before_sending(self):
    with vector_env.SubprocessVectorEnv(
        (_FakeBuilder(0), _FakeBuilder(0)), start_method="spawn"
    ) as env:
      with self.assertRaisesRegex(ValueError, "Expected actions for 2"):
        env.step(((1, 2),))

  def test_rejects_invalid_reset_indices(self):
    with vector_env.SubprocessVectorEnv(
        (_FakeBuilder(0), _FakeBuilder(0)), start_method="spawn"
    ) as env:
      with self.assertRaisesRegex(ValueError, "must not contain duplicates"):
        env.reset(indices=(0, 0))
      with self.assertRaisesRegex(IndexError, "outside"):
        env.reset_at(2)

  def test_reports_worker_startup_failures(self):
    with self.assertRaisesRegex(
        vector_env.VectorEnvWorkerError,
        "Worker 1 raised RuntimeError: builder failed",
    ):
      vector_env.SubprocessVectorEnv(
          (_FakeBuilder(0), _FailingBuilder()), start_method="spawn"
      )

  def test_operations_fail_after_close(self):
    env = vector_env.SubprocessVectorEnv(
        (_FakeBuilder(0),), start_method="spawn"
    )
    env.close()
    env.close()
    with self.assertRaisesRegex(RuntimeError, "closed"):
      env.reset()


if __name__ == "__main__":
  absltest.main()
