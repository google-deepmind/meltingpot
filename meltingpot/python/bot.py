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
"""Bot factory."""

import abc
import os
import re
from typing import Tuple

import dm_env
from ml_collections import config_dict
import numpy as np
import tensorflow as tf
import tree

from meltingpot.python.configs import bots as bot_config
from meltingpot.python.utils.bots import permissive_model
from meltingpot.python.utils.bots import puppeteer_functions

_MODELS_ROOT = re.sub('meltingpot/python/.*', 'meltingpot/assets/saved_models',
                      __file__)

AVAILABLE_BOTS = frozenset(bot_config.BOTS)

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


class SavedModelPolicy(Policy):
  """Policy wrapping a saved model for inference.

  Note: the model should have methods:
  1. `initial_state(batch_size, trainable)`
  2. `step(step_type, reward, discount, observation, prev_state)`
  that accept batched inputs and produce batched outputs.
  """

  def __init__(self, model_path: str) -> None:
    """Initialize a policy instance.

    Args:
      model_path: Path to the SavedModel.
    """
    model = tf.saved_model.load(model_path)
    self._model = permissive_model.PermissiveModel(model)

  def step(self, timestep: dm_env.TimeStep,
           prev_state: State) -> Tuple[int, State]:
    """See base class."""
    step_type = np.array(timestep.step_type, dtype=np.int64)[None]
    reward = np.asarray(timestep.reward, dtype=np.float32)[None]
    discount = np.asarray(timestep.discount, dtype=np.float32)[None]
    observation = tree.map_structure(lambda x: x[None], timestep.observation)
    output, next_state = self._model.step(
        step_type=step_type,
        reward=reward,
        discount=discount,
        observation=observation,
        prev_state=prev_state)
    action = int(output.action['environment_action'][0])
    return action, next_state

  def initial_state(self) -> State:
    """See base class."""
    return self._model.initial_state(batch_size=1, trainable=None)

  def close(self) -> None:
    """See base class."""
    pass


_GOAL_OBS_NAME = 'GOAL'


class PuppetPolicy(Policy):
  """Wraps a puppet deepfunc as a python bot."""

  def __init__(self, puppeteer_fn: puppeteer_functions.PuppeteerFn,
               puppet_path: str) -> None:
    """Creates a new PuppetBot.

    Args:
      puppeteer_fn: The puppeteer function. This will be called at every step to
        obtain the goal of that step for the underlying puppet.
      puppet_path: Path to the puppet's saved model.
    """
    self._puppeteer_fn = puppeteer_fn
    self._puppet = SavedModelPolicy(puppet_path)

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


def get_config(bot_name: str) -> config_dict.ConfigDict:
  """Returns a config for the specified bot.

  Args:
    bot_name: name of the bot. Must be in AVAILABLE_BOTS.
  """
  if bot_name not in AVAILABLE_BOTS:
    raise ValueError(f'Unknown bot {bot_name!r}.')
  bot = bot_config.BOTS[bot_name]
  config = config_dict.create(
      bot_name=bot_name,
      substrate=bot.substrate,
      puppeteer_fn=bot.puppeteer_fn,
      saved_model_path=os.path.join(_MODELS_ROOT, bot.substrate, bot.model),
  )
  return config.lock()


def build(config: config_dict.ConfigDict) -> Policy:
  """Builds a bot policy for the given config.

  Args:
    config: bot config resulting from `get_config`.

  Returns:
    The bot policy.
  """
  if config.puppeteer_fn:
    return PuppetPolicy(config.puppeteer_fn, config.saved_model_path)
  else:
    return SavedModelPolicy(config.saved_model_path)
