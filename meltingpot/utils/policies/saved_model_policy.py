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
"""Policy from a Saved Model."""

import contextlib
import random

import dm_env
from meltingpot.utils.policies import permissive_model
from meltingpot.utils.policies import policy
import numpy as np
import tensorflow as tf
import tree


def _numpy_to_placeholder(
    template: tree.Structure[np.ndarray], prefix: str
) -> tree.Structure[tf.Tensor]:
  """Returns placeholders that matches a given template.

  Args:
    template: template numpy arrays.
    prefix: a prefix to add to the placeholder names.

  Returns:
    A tree of placeholders matching the template arrays' specs.
  """
  def fn(path, x):
    name = '.'.join(str(x) for x in path)
    return tf.compat.v1.placeholder(shape=x.shape, dtype=x.dtype,
                                    name=f'{prefix}.{name}')
  return tree.map_structure_with_path(fn, template)


def _downcast(x):
  """Downcasts input to 32-bit precision."""
  if not isinstance(x, np.ndarray):
    return x
  elif x.dtype == np.float64:
    return np.asarray(x, dtype=np.float32)
  elif x.dtype == np.int64:
    return np.asarray(x, dtype=np.int32)
  else:
    return x


class TF2SavedModelPolicy(policy.Policy[tree.Structure[tf.Tensor]]):
  """Policy wrapping a saved model for TF2 inference.

  Note: the model should have methods:
  1. `initial_state(random_key)`
  2. `step(key, timestep, prev_state)`
  that accept unbatched inputs.
  """

  def __init__(self, model_path: str, device_name: str = 'cpu') -> None:
    """Initialize a policy instance.

    Args:
      model_path: Path to the SavedModel.
      device_name: Device to load SavedModel onto. Defaults to a cpu device.
        See tf.device for supported device names.
    """
    self._strategy = tf.distribute.OneDeviceStrategy(device_name)
    with self._strategy.scope():
      model = tf.saved_model.load(model_path)
      self._model = permissive_model.PermissiveModel(model)

  def step(
      self,
      timestep: dm_env.TimeStep,
      prev_state: tree.Structure[tf.Tensor],
  ) -> tuple[int, tree.Structure[tf.Tensor]]:
    """See base class."""
    prev_key, prev_state = prev_state
    timestep = timestep._replace(
        step_type=int(timestep.step_type),
        observation=tree.map_structure(_downcast, timestep.observation),
    )
    next_key, outputs = self._strategy.run(
        self._model.step, [prev_key, timestep, prev_state])
    (action, _), next_state = outputs
    return int(action.numpy()), (next_key, next_state)

  def initial_state(self) -> tree.Structure[tf.Tensor]:
    """See base class."""
    random_seed = random.getrandbits(32)
    seed_key = np.array([0, random_seed], dtype=np.uint32)
    key, state = self._strategy.run(self._model.initial_state, [seed_key])
    return key, state

  def close(self) -> None:
    """See base class."""


class TF1SavedModelPolicy(policy.Policy[tree.Structure[np.ndarray]]):
  """Policy wrapping a saved model for TF1 inference.

  Note: the model should have methods:
  1. `initial_state(batch_size, trainable)`
  2. `step(step_type, reward, discount, observation, prev_state)`
  that accept batched inputs and produce batched outputs.
  """

  def __init__(self, model_path: str, device_name: str = 'cpu') -> None:
    """Initialize a policy instance.

    Args:
      model_path: Path to the SavedModel.
      device_name: Device to load SavedModel onto. Defaults to a cpu device.
        See tf.device for supported device names.
    """
    self._device_name = device_name
    self._graph = tf.compat.v1.Graph()
    self._session = tf.compat.v1.Session(graph=self._graph)

    with self._build_context():
      model = tf.compat.v1.saved_model.load_v2(model_path)
      self._model = permissive_model.PermissiveModel(model)

    self._initial_state_outputs = None
    self._step_inputs = None
    self._step_outputs = None

  @contextlib.contextmanager
  def _build_context(self):
    with self._graph.as_default():  # pylint: disable=not-context-manager
      with tf.compat.v1.device(self._device_name):
        yield

  def _build_initial_state_graph(self) -> None:
    """Builds the TF1 subgraph for the initial_state operation."""
    with self._build_context():
      key_in = tf.compat.v1.placeholder(shape=[2], dtype=np.uint32)
      self._initial_state_outputs = self._model.initial_state(key_in)
      self._initial_state_input = key_in

  def _build_step_graph(self, timestep, prev_state) -> None:
    """Builds the TF1 subgraph for the step operation.

    Args:
      timestep: an example timestep.
      prev_state: an example previous state.
    """
    if not self._initial_state_outputs:
      self._build_initial_state_graph()

    with self._build_context():
      step_type_in = tf.compat.v1.placeholder(
          shape=[], dtype=np.int32, name='step_type')
      reward_in = tf.compat.v1.placeholder(
          shape=[], dtype=np.float32, name='reward')
      discount_in = tf.compat.v1.placeholder(
          shape=[], dtype=np.float32, name='discount')
      observation_in = _numpy_to_placeholder(
          timestep.observation, prefix='observation')
      timestep_in = dm_env.TimeStep(
          step_type=step_type_in,
          reward=reward_in,
          discount=discount_in,
          observation=observation_in)
      prev_key_in, prev_state_in = _numpy_to_placeholder(
          prev_state, prefix='prev_state')
      next_key, outputs = self._model.step(prev_key_in, timestep_in,
                                           prev_state_in)
      (action, _), next_state = outputs
      input_values = tree.flatten_with_path({
          'timestep': timestep_in,
          'prev_state': (prev_key_in, prev_state_in),
      })
      self._step_inputs = dict(input_values)
      self._step_outputs = (action, (next_key, next_state))

    self._graph.finalize()

  def step(
      self, timestep: dm_env.TimeStep, prev_state: tree.Structure[np.ndarray]
  ) -> tuple[int, tree.Structure[np.ndarray]]:
    """See base class."""
    timestep = timestep._replace(
        step_type=int(timestep.step_type),
        observation=tree.map_structure(_downcast, timestep.observation),
    )
    if not self._step_inputs:
      self._build_step_graph(timestep, prev_state)
    input_values = tree.flatten_with_path({
        'timestep': timestep,
        'prev_state': prev_state,
    })
    feed_dict = {
        self._step_inputs[path]: value for path, value in input_values
        if path in self._step_inputs
    }
    action, next_state = self._session.run(self._step_outputs, feed_dict)
    return int(action), next_state

  def initial_state(self) -> tree.Structure[np.ndarray]:
    """See base class."""
    if not self._initial_state_outputs:
      self._build_initial_state_graph()
    random_seed = random.getrandbits(32)
    seed_key = np.array([0, random_seed], dtype=np.uint32)
    feed_dict = {self._initial_state_input: seed_key}
    return self._session.run(self._initial_state_outputs, feed_dict)

  def close(self) -> None:
    """See base class."""
    self._session.close()


if tf.executing_eagerly():
  SavedModelPolicy = TF2SavedModelPolicy
else:
  SavedModelPolicy = TF1SavedModelPolicy
