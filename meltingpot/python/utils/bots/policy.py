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
from typing import Mapping, Tuple

import dm_env
import numpy as np
import tensorflow as tf
import tree

from meltingpot.python.utils.bots import permissive_model
from meltingpot.python.utils.bots import puppeteer_functions

State = tree.Structure[np.ndarray]


class Policy(metaclass=abc.ABCMeta):
  """Abstract base class for a policy."""

  @abc.abstractmethod
  def initial_state(self) -> State:
    """Returns the initial state of the agent."""
    raise NotImplementedError()

  @abc.abstractmethod
  def step(self, timestep: dm_env.TimeStep,
           prev_state: State) -> Tuple[int, State]:
    """Steps the agent.

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


def _tensor_to_numpy(
    tensors: tree.Structure[tf.Tensor]) -> tree.Structure[np.ndarray]:
  """Converts tensors to numpy arrays.

  Args:
    tensors: input tensors.

  Returns:
    The values of the tensors.
  """
  if tf.executing_eagerly():
    return tree.map_structure(lambda x: x.numpy(), tensors)
  else:
    with tf.compat.v1.Session() as sess:
      return sess.run(tensors)


class SavedModelPolicy(Policy):
  """Policy wrapping a saved model for inference.

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

  def step(self, timestep: dm_env.TimeStep,
           prev_state: State) -> Tuple[int, State]:
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
    action = int(_tensor_to_numpy(action)[0])
    next_state = _tensor_to_numpy(next_state)
    return action, next_state

  def initial_state(self) -> State:
    """See base class."""
    state = self._strategy.run(
        fn=self._model.initial_state,
        kwargs=dict(batch_size=1, trainable=None))
    return _tensor_to_numpy(state)

  def close(self) -> None:
    """See base class."""


_GOAL_OBS_NAME = 'GOAL'


class PuppetPolicy(Policy):
  """A puppet policy controlled by a puppeteer function."""

  def __init__(self, puppeteer_fn: puppeteer_functions.PuppeteerFn,
               puppet_policy: Policy) -> None:
    """Creates a new PuppetBot.

    Args:
      puppeteer_fn: The puppeteer function. This will be called at every step to
        obtain the goal of that step for the underlying puppet.
      puppet_policy: The puppet policy. Will be closed with this wrapper.
    """
    self._puppeteer_fn = puppeteer_fn
    self._puppet = puppet_policy

  def _puppeteer_initial_state(self) -> int:
    return 0

  def _puppeteer_step(self, timestep: dm_env.TimeStep,
                      prev_state: int) -> Tuple[dm_env.TimeStep, int]:
    """Returns the transformed observation for the puppet step."""
    goal = self._puppeteer_fn(prev_state, timestep.observation)
    next_state = prev_state + 1
    puppet_observation = timestep.observation.copy()
    puppet_observation[_GOAL_OBS_NAME] = goal
    puppet_timestep = timestep._replace(observation=puppet_observation)
    return puppet_timestep, next_state

  def step(self, timestep: dm_env.TimeStep,
           prev_state: State) -> Tuple[int, State]:
    """See base class."""
    puppet_timestep, puppeteer_state = self._puppeteer_step(
        timestep, prev_state['puppeteer'])
    action, puppet_state = self._puppet.step(puppet_timestep,
                                             prev_state['puppet'])
    next_state = {
        'puppeteer': puppeteer_state,
        'puppet': puppet_state,
    }
    return action, next_state

  def initial_state(self) -> State:
    """See base class."""
    return {
        'puppeteer': 0,
        'puppet': self._puppet.initial_state(),
    }

  def close(self) -> None:
    """See base class."""
    self._puppet.close()
