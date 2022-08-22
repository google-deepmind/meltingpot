# Copyright 2020 DeepMind Technologies Limited.
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
"""Bot policy implementations."""

import abc
import contextlib
from typing import Callable, Generic, Mapping, Tuple, TypeVar

import dm_env
import immutabledict
import numpy as np
import tensorflow as tf
import tree

from meltingpot.python.utils.bots import permissive_model
from meltingpot.python.utils.bots import puppeteer_functions

State = TypeVar('State')


class Policy(Generic[State], metaclass=abc.ABCMeta):
  """Abstract base class for a policy.

  Must not possess any mutable state not in `initial_state`.
  """

  @abc.abstractmethod
  def initial_state(self) -> State:
    """Returns the initial state of the agent.

    Must not have any side effects.
    """
    raise NotImplementedError()

  @abc.abstractmethod
  def step(self, timestep: dm_env.TimeStep,
           prev_state: State) -> Tuple[int, State]:
    """Steps the agent.

    Must not have any side effects.

    Args:
      timestep: information from the environment
      prev_state: the previous state of the agent.

    Returns:
      action: the action to send to the environment.
      next_state: the state for the next step call.
    """
    raise NotImplementedError()

  @abc.abstractmethod
  def close(self) -> None:
    """Closes the policy."""
    raise NotImplementedError()

  def __enter__(self):
    return self

  def __exit__(self, *args, **kwargs):
    del args, kwargs
    self.close()


class TF2SavedModelPolicy(Policy[tree.Structure[tf.Tensor]]):
  """Policy wrapping a saved model for TF2 inference.

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
    self._strategy = tf.distribute.OneDeviceStrategy(device_name)
    with self._strategy.scope():
      model = tf.saved_model.load(model_path)
      self._model = permissive_model.PermissiveModel(model)

  def step(
      self,
      timestep: dm_env.TimeStep,
      prev_state: tree.Structure[tf.Tensor],
  ) -> Tuple[int, tree.Structure[tf.Tensor]]:
    """See base class."""
    step_type = np.array(timestep.step_type, dtype=np.int64)[None]
    reward = np.asarray(timestep.reward, dtype=np.float32)[None]
    discount = np.asarray(timestep.discount, dtype=np.float32)[None]
    observation = tree.map_structure(lambda x: x[None], timestep.observation)
    output, next_state = self._strategy.run(
        fn=self._model.step,
        kwargs=dict(
            step_type=step_type,
            reward=reward,
            discount=discount,
            observation=observation,
            prev_state=prev_state,
        ),
    )
    if isinstance(output.action, Mapping):
      # Legacy bots trained with older action spec.
      action = output.action['environment_action']
    else:
      action = output.action
    action = int(action.numpy()[0])
    return action, next_state

  def initial_state(self) -> tree.Structure[tf.Tensor]:
    """See base class."""
    return self._strategy.run(
        fn=self._model.initial_state, kwargs=dict(batch_size=1, trainable=None))

  def close(self) -> None:
    """See base class."""


def _numpy_to_placeholder(
    template: tree.Structure[np.ndarray]) -> tree.Structure[tf.Tensor]:
  """Returns placeholders that matches a given template.

  Args:
    template: template numpy arrays.

  Returns:
    A tree of placeholders matching the template arrays' specs.
  """
  fn = lambda x: tf.compat.v1.placeholder(shape=x.shape, dtype=x.dtype)
  return tree.map_structure(fn, template)


class TF1SavedModelPolicy(Policy[tree.Structure[np.ndarray]]):
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
    with self._graph.as_default():
      with tf.compat.v1.device(self._device_name):
        yield

  def _build_initial_state_graph(self) -> None:
    """Builds the TF1 subgraph for the initial_state operation."""
    with self._build_context():
      self._initial_state_outputs = self._model.initial_state(
          batch_size=1, trainable=None)

  def _build_step_graph(self, timestep, prev_state) -> None:
    """Builds the TF1 subgraph for the step operation.

    Args:
      timestep: an example timestep.
      prev_state: an example previous state.
    """
    if not self._initial_state_outputs:
      self._build_initial_state_graph()

    with self._build_context():
      step_type_in = tf.compat.v1.placeholder(shape=[], dtype=np.int64)
      reward_in = tf.compat.v1.placeholder(shape=[], dtype=np.float32)
      discount_in = tf.compat.v1.placeholder(shape=[], dtype=np.float32)
      observation_in = _numpy_to_placeholder(timestep.observation)
      prev_state_in = _numpy_to_placeholder(prev_state)
      output, next_state = self._model.step(
          step_type=step_type_in[None],
          reward=reward_in[None],
          discount=discount_in[None],
          observation=tree.map_structure(lambda x: x[None], observation_in),
          prev_state=prev_state_in)
      if isinstance(output.action, Mapping):
        # Legacy bots trained with older action spec.
        action = output.action['environment_action'][0]
      else:
        action = output.action[0]

    timestep_in = dm_env.TimeStep(
        step_type=step_type_in,
        reward=reward_in,
        discount=discount_in,
        observation=observation_in)
    self._step_inputs = tree.flatten([timestep_in, prev_state_in])
    self._step_outputs = (action, next_state)

    self._graph.finalize()

  def step(
      self, timestep: dm_env.TimeStep, prev_state: tree.Structure[np.ndarray]
  ) -> Tuple[int, tree.Structure[np.ndarray]]:
    """See base class."""
    if not self._step_inputs:
      self._build_step_graph(timestep, prev_state)
    input_values = tree.flatten([timestep, prev_state])
    feed_dict = dict(zip(self._step_inputs, input_values))
    action, next_state = self._session.run(self._step_outputs, feed_dict)
    return int(action), next_state

  def initial_state(self) -> tree.Structure[np.ndarray]:
    """See base class."""
    if not self._initial_state_outputs:
      self._build_initial_state_graph()
    return self._session.run(self._initial_state_outputs)

  def close(self) -> None:
    """See base class."""
    self._session.close()


if tf.executing_eagerly():
  SavedModelPolicy = TF2SavedModelPolicy
else:
  SavedModelPolicy = TF1SavedModelPolicy


_GOAL_OBS_NAME = 'GOAL'


class _Puppeteer(Generic[State]):
  """A puppeteer that controls the timestep forwarded to the puppet."""

  def __init__(
      self,
      puppeteer_fn_builder: Callable[[], puppeteer_functions.PuppeteerFn],
  ) -> None:
    """Initializes the instance.

    Args:
      puppeteer_fn_builder: Builds the puppeteer function that will be called at
        every step to obtain the goal of that step for the underlying puppet.
    """
    self._puppeteer_fn_builder = puppeteer_fn_builder

  def initial_state(self) -> State:
    """Returns the initial state of the puppeteer."""
    step_count = 0
    puppeteer_fn = self._puppeteer_fn_builder()
    return (step_count, puppeteer_fn)

  def step(self, timestep: dm_env.TimeStep,
           prev_state: State) -> Tuple[dm_env.TimeStep, State]:
    """Steps the puppeteer.

    Args:
      timestep: information from the environment.
      prev_state: the previous state of the puppeteer.

    Returns:
      timestep: the timestep to forward to the puppet.
      next_state: the state for the next step call.
    """
    step_count, puppeteer_fn = prev_state
    goal = puppeteer_fn(step_count, timestep.observation)
    if timestep.step_type == dm_env.StepType.LAST:
      next_state = self.initial_state()
    else:
      next_state = (step_count + 1, puppeteer_fn)

    puppet_observation = immutabledict.immutabledict(
        timestep.observation, **{_GOAL_OBS_NAME: goal})
    puppet_timestep = timestep._replace(observation=puppet_observation)
    return puppet_timestep, next_state


class PuppetPolicy(Policy[State], Generic[State]):
  """A puppet policy controlled by a puppeteer function."""

  def __init__(
      self,
      puppeteer_fn_builder: Callable[[], puppeteer_functions.PuppeteerFn],
      puppet_policy: Policy) -> None:
    """Creates a new PuppetBot.

    Args:
      puppeteer_fn_builder: Builds the puppeteer function that will be called at
        every step to obtain the goal of that step for the underlying puppet.
      puppet_policy: The puppet policy. Will be closed with this wrapper.
    """
    self._puppeteer = _Puppeteer(puppeteer_fn_builder)
    self._puppet = puppet_policy

  def step(self, timestep: dm_env.TimeStep,
           prev_state: State) -> Tuple[int, State]:
    """See base class."""
    puppeteer_state, puppet_state = prev_state
    puppet_timestep, puppeteer_state = self._puppeteer.step(
        timestep, puppeteer_state)
    action, puppet_state = self._puppet.step(puppet_timestep, puppet_state)
    next_state = (puppeteer_state, puppet_state)
    return action, next_state

  def initial_state(self) -> State:
    """See base class."""
    return (self._puppeteer.initial_state(), self._puppet.initial_state())

  def close(self) -> None:
    """See base class."""
    self._puppet.close()
