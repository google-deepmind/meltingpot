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
"""PettingZoo interface to meltingpot environments."""

import functools

from gymnasium.utils import EzPickle
import matplotlib.pyplot as plt
from ml_collections import config_dict
from pettingzoo.utils import parallel_to_aec
from pettingzoo.utils.env import ParallelEnv
from pettingzoo.utils import wrappers

from examples import utils
from meltingpot.python import substrate

PLAYER_STR_FORMAT = 'player_{index}'
MAX_CYCLES = 1000


def parallel_env(render_mode, env_config, max_cycles=MAX_CYCLES):
  env = ParallelMeltingPotEnv(render_mode, env_config, max_cycles)
  # env = ParallelMeltingPotPettingZooEnv(env)
  return env


def raw_env(render_mode, env_config, max_cycles=MAX_CYCLES):
  return parallel_to_aec(
    parallel_env(render_mode, env_config, max_cycles))


def env(render_mode, env_config, max_cycles=MAX_CYCLES):
  aec_env = raw_env(render_mode, env_config, max_cycles)
  aec_env = wrappers.AssertOutOfBoundsWrapper(aec_env)
  aec_env = wrappers.OrderEnforcingWrapper(aec_env)
  return aec_env


class ParallelMeltingPotEnv(ParallelEnv, EzPickle):
  """An adapter between Melting Pot substrates and PettingZoo's ParallelEnv."""
  metadata = {'render_modes': ['human', 'rgb_array'], 'name': 'MeltingPotPettingZooEnv-v0'}

  def __init__(self, render_mode, env_config, max_cycles):
    EzPickle.__init__(self, render_mode, env_config, max_cycles)
    self.env_config = config_dict.ConfigDict(env_config)
    self.max_cycles = max_cycles
    self.render_mode = render_mode
    self._env = substrate.build(env_config['substrate'], roles=env_config['roles'])
    self._num_players = len(self._env.observation_spec())
    self.possible_agents = [
        PLAYER_STR_FORMAT.format(index=index)
        for index in range(self._num_players)
    ]
    self.agents = [agent for agent in self.possible_agents]
    observation_space = utils.remove_world_observations_from_space(
        utils.spec_to_space(self._env.observation_spec()[0]))
    self.observation_space = functools.lru_cache(
        maxsize=None)(lambda agent_id: observation_space)
    action_space = utils.spec_to_space(self._env.action_spec()[0])
    self.action_space = functools.lru_cache(maxsize=None)(
        lambda agent_id: action_space)
    self.state_space = utils.spec_to_space(
        self._env.observation_spec()[0]['WORLD.RGB'])

  def state(self):
    return self._env.observation()

  def reset(self, seed=None, **kwargs):
    """See base class."""
    timestep = self._env.reset()
    self.agents = self.possible_agents[:]
    self.num_cycles = 0
    return utils.timestep_to_observations(timestep)

  def step(self, action):
    """See base class."""
    actions = [action[agent] for agent in self.agents]
    timestep = self._env.step(actions)
    rewards = {
        agent: timestep.reward[index] for index, agent in enumerate(self.agents)
    }
    self.num_cycles += 1
    termination = timestep.last()
    terminations = {agent: termination for agent in self.agents}
    truncation = self.num_cycles >= self.max_cycles
    truncations = {agent: truncation for agent in self.agents}
    infos = {agent: {} for agent in self.agents}
    if termination:
      self.agents = []

    observations = utils.timestep_to_observations(timestep)
    return observations, rewards, terminations, truncations, infos

  def close(self):
    """See base class."""
    self._env.close()

  def render(self):
    rgb_arr = self.state()['WORLD.RGB']
    if self.render_mode == 'human':
      plt.cla()
      plt.imshow(rgb_arr, interpolation='nearest')
      plt.show(block=False)
      return None
    return rgb_arr
