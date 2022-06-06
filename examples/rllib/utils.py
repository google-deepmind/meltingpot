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
"""MeltingPotEnv as a MultiAgentEnv wrapper to interface with RLLib."""

from typing import Tuple

import dm_env
import dmlab2d
from gym import spaces
from ml_collections import config_dict
import numpy as np
from ray.rllib.agents import trainer
from ray.rllib.env import multi_agent_env
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID
import tree

from meltingpot.python import substrate
from meltingpot.python.utils.bots import policy

PLAYER_STR_FORMAT = 'player_{index}'


def _timestep_to_observations(timestep: dm_env.TimeStep):
  gym_observations = {}
  for index, observation in enumerate(timestep.observation):
    gym_observations[PLAYER_STR_FORMAT.format(index=index)] = {
        key: value for key, value in observation.items() if 'WORLD' not in key
    }
  return gym_observations


def _remove_world_observations_from_space(
    observation: spaces.Dict) -> spaces.Dict:
  return spaces.Dict(
      {key: observation[key] for key in observation if 'WORLD' not in key})


def _spec_to_space(spec: tree.Structure[dm_env.specs.Array]) -> spaces.Space:
  """Converts a dm_env nested structure of specs to a Gym Space.

  BoundedArray is converted to Box Gym spaces. DiscreteArray is converted to
  Discrete Gym spaces. Using Tuple and Dict spaces recursively as needed.

  Args:
    spec: The nested structure of specs

  Returns:
    The Gym space corresponding to the given spec.
  """
  if isinstance(spec, dm_env.specs.DiscreteArray):
    return spaces.Discrete(spec.num_values)
  elif isinstance(spec, dm_env.specs.BoundedArray):
    return spaces.Box(spec.minimum, spec.maximum, spec.shape, spec.dtype)
  elif isinstance(spec, dm_env.specs.Array):
    if np.issubdtype(spec.dtype, np.floating):
      return spaces.Box(-np.inf, np.inf, spec.shape, spec.dtype)
    elif np.issubdtype(spec.dtype, np.integer):
      info = np.iinfo(spec.dtype)
      return spaces.Box(info.min, info.max, spec.shape, spec.dtype)
    else:
      raise NotImplementedError(f'Unsupported dtype {spec.dtype}')
  elif isinstance(spec, (list, tuple)):
    return spaces.Tuple([_spec_to_space(s) for s in spec])
  elif isinstance(spec, dict):
    return spaces.Dict({key: _spec_to_space(s) for key, s in spec.items()})
  else:
    raise ValueError('Unexpected spec: {}'.format(spec))


class MeltingPotEnv(multi_agent_env.MultiAgentEnv):
  """An adapter between the Melting Pot substrates and RLLib MultiAgentEnv."""

  def __init__(self, env: dmlab2d.Environment):
    self._env = env
    self._num_players = len(self._env.observation_spec())
    self._ordered_agent_ids = [
        PLAYER_STR_FORMAT.format(index=index)
        for index in range(self._num_players)
    ]
    self._agent_ids = set(self._ordered_agent_ids)
    super().__init__()

  def reset(self):
    """See base class."""
    timestep = self._env.reset()
    return _timestep_to_observations(timestep)

  def step(self, action):
    """See base class."""
    actions = [action[agent_id] for agent_id in self._ordered_agent_ids]
    timestep = self._env.step(actions)
    rewards = {
        agent_id: timestep.reward[index]
        for index, agent_id in enumerate(self._ordered_agent_ids)
    }
    done = {'__all__': True if timestep.last() else False}
    info = {}

    observations = _timestep_to_observations(timestep)
    return observations, rewards, done, info

  def close(self):
    """See base class."""
    self._env.close()

  def get_dmlab2d_env(self):
    """Returns the underlying DM Lab2D environment."""
    return self._env

  def single_player_observation_space(self) -> spaces.Space:
    """The observation space for a single player in this environment."""
    return _remove_world_observations_from_space(
        _spec_to_space(self._env.observation_spec()[0]))

  def single_player_action_space(self):
    """The action space for a single player in this environment."""
    return _spec_to_space(self._env.action_spec()[0])


def env_creator(env_config):
  """Outputs an environment for registering."""
  env = substrate.build(config_dict.ConfigDict(env_config))
  env = MeltingPotEnv(env)
  return env


class RayModelPolicy(policy.Policy):
  """Policy wrapping an rllib model for inference.

  Note: Currently only supports a single input, batching is not enabled
  """

  def __init__(self,
               model: trainer.Trainer,
               policy_id: str = DEFAULT_POLICY_ID) -> None:
    """Initialize a policy instance.

    Args:
      model: An rllib.trainer.Trainer checkpoint.
      policy_id: Which policy to use (if trained in multi_agent mode)
    """
    self._model = model
    self._prev_action = 0
    self._policy_id = policy_id

  def step(self, timestep: dm_env.TimeStep,
           prev_state: policy.State) -> Tuple[int, policy.State]:
    """See base class."""
    observations = {
        key: value
        for key, value in timestep.observation.items()
        if 'WORLD' not in key
    }

    action, state, _ = self._model.compute_single_action(
        observations,
        prev_state,
        policy_id=self._policy_id,
        prev_action=self._prev_action,
        prev_reward=timestep.reward)

    self._prev_action = action
    return action, state

  def initial_state(self) -> policy.State:
    """See base class."""
    self._prev_action = 0
    return self._model.get_policy(self._policy_id).get_initial_state()

  def close(self) -> None:
    """See base class."""
