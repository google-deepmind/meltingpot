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
"""Scalar NumPy observations follow the same precision rules as arrays."""

import inspect
import tempfile

from absl.testing import absltest
from absl.testing import parameterized
import dm_env
from meltingpot.utils.policies import saved_model_policy
import numpy as np
import tensorflow as tf

_KEY_SPEC = tf.TensorSpec((2,), tf.uint32)
_STATE_SPEC = tf.TensorSpec((), tf.int32)
_TIMESTEP_SPEC = dm_env.TimeStep(
    step_type=tf.TensorSpec((), tf.int32),
    reward=tf.TensorSpec((), tf.float32),
    discount=tf.TensorSpec((), tf.float32),
    observation={
        'value': tf.TensorSpec((), tf.float32),
        'count': tf.TensorSpec((), tf.int32),
    },
)


class _ScalarModel(tf.Module):
  """Small serialized policy implementing the real permissive-model protocol."""

  @tf.function(input_signature=[])
  def function_signatures(self):
    kind = tf.constant(int(inspect.Parameter.POSITIONAL_OR_KEYWORD))
    return {
        'initial_state': ((tf.constant('key'), kind),),
        'step': (
            (tf.constant('key'), kind),
            (tf.constant('timestep'), kind),
            (tf.constant('prev_state'), kind),
        ),
    }

  @tf.function(input_signature=[])
  def function_tables(self):
    return {}

  @tf.function(input_signature=[_KEY_SPEC])
  def initial_state(self, key):
    return key, tf.constant(0, tf.int32)

  @tf.function(input_signature=[_KEY_SPEC, _TIMESTEP_SPEC, _STATE_SPEC])
  def step(self, key, timestep, prev_state):
    action = tf.cast(timestep.observation['value'], tf.int32)
    action += timestep.observation['count'] + prev_state
    return key, ((action, tf.constant(0.0)), prev_state + 1)


class ScalarObservationTest(parameterized.TestCase):

  @parameterized.parameters(np.float64, np.int64)
  def test_numpy_scalar_and_array_have_identical_downcast_shapes_and_values(
      self, dtype
  ):
    scalar = dtype(7)
    actual = saved_model_policy._downcast(scalar)
    expected = saved_model_policy._downcast(np.asarray(scalar))
    self.assertEqual(actual.dtype, expected.dtype)
    self.assertEqual(actual.shape, ())
    np.testing.assert_array_equal(actual, expected)
    self.assertEqual(scalar.dtype, np.dtype(dtype))

  @parameterized.parameters(
      np.float32(2.5),
      np.int32(3),
      np.uint64(7),
      np.uint8(5),
      np.bool_(True),
      3,
      2.5,
      None,
  )
  def test_other_scalar_types_are_passed_through_unchanged(self, value):
    self.assertIs(saved_model_policy._downcast(value), value)

  @parameterized.parameters(np.float64, np.int64)
  def test_array_views_still_convert_without_modifying_the_source(self, dtype):
    source = np.arange(12, dtype=dtype).reshape(3, 4)
    values = source[:, ::2]
    values.setflags(write=False)
    before = source.copy()
    actual = saved_model_policy._downcast(values)
    self.assertEqual(
        actual.dtype, np.dtype(np.float32 if dtype == np.float64 else np.int32)
    )
    np.testing.assert_array_equal(actual, values)
    np.testing.assert_array_equal(source, before)

  @parameterized.product(
      graph_mode=(False, True),
      scalar_values=(False, True),
  )
  def test_real_saved_model_inference_accepts_scalar_and_zero_dimensional_observations(
      self, graph_mode, scalar_values
  ):
    with tempfile.TemporaryDirectory() as directory:
      tf.saved_model.save(_ScalarModel(), directory)
      policy_type = (
          saved_model_policy.TF1SavedModelPolicy
          if graph_mode
          else saved_model_policy.TF2SavedModelPolicy
      )
      with policy_type(directory) as policy:
        state = policy.initial_state()
        for index in range(3):
          value = np.float64(2.5 + index)
          count = np.int64(3)
          if not scalar_values:
            value = np.asarray(value)
            count = np.asarray(count)
          timestep = dm_env.TimeStep(
              step_type=dm_env.StepType.FIRST
              if index == 0
              else dm_env.StepType.MID,
              reward=0.0,
              discount=1.0,
              observation={'value': value, 'count': count},
          )
          action, state = policy.step(timestep, state)
          self.assertEqual(action, 5 + 2 * index)
          self.assertEqual(int(state[1]), index + 1)
          self.assertEqual(
              timestep.observation['value'].dtype, np.dtype(np.float64)
          )
          self.assertEqual(
              timestep.observation['count'].dtype, np.dtype(np.int64)
          )


if __name__ == '__main__':
  absltest.main()
