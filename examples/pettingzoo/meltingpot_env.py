""" PettingZoo interface to meltingpot environments"""

import dm_env
from meltingpot.python import substrate
from ml_collections import config_dict
from gym import spaces
import numpy as np
from functools import lru_cache

from gym.utils import EzPickle
from pettingzoo.utils import wrappers
from pettingzoo.utils.conversions import parallel_to_aec_wrapper
from pettingzoo.utils.env import ParallelEnv
import tree
import matplotlib.pyplot as plt

PLAYER_STR_FORMAT = 'player_{index}'
MAX_CYCLES = 1000

def _timestep_to_observations(timestep: dm_env.TimeStep):
  gym_observations = {}
  for index, observation in enumerate(timestep.observation):
    gym_observations[PLAYER_STR_FORMAT.format(index=index)] = {
        key: value for key, value in observation.items() if 'WORLD' not in key
    }
  return gym_observations


def _remove_world_observations_from_space(
    observation: spaces.Dict) -> spaces.Dict:
  return spaces.Dict({
      key: observation[key] for key in observation if 'WORLD' not in key})


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

def parallel_env(env_config, max_cycles=MAX_CYCLES):
    return _parallel_env(env_config, max_cycles)

def raw_env(env_config, max_cycles=MAX_CYCLES):
    return parallel_to_aec_wrapper(parallel_env(env_config, max_cycles))

def env(env_config, max_cycles=MAX_CYCLES):
    aec_env = raw_env(env_config, max_cycles)
    aec_env = wrappers.AssertOutOfBoundsWrapper(aec_env)
    aec_env = wrappers.OrderEnforcingWrapper(aec_env)
    return aec_env

class MeltingPotPettingZooEnv(ParallelEnv):
  """An adapter between the Melting Pot substrates and PettingZoo's ParallelEnv"""

  def __init__(self, env_config, max_cycles):
    self.env_config = config_dict.ConfigDict(env_config)
    self.max_cycles = max_cycles
    self._env = substrate.build(self.env_config)
    self._num_players = len(self._env.observation_spec())
    self.possible_agents = [
        PLAYER_STR_FORMAT.format(index=index)
        for index in range(self._num_players)
    ]
    observation_space = _remove_world_observations_from_space(
        _spec_to_space(self._env.observation_spec()[0]))
    self.observation_space = lru_cache(maxsize=None)(lambda agent_id: observation_space)
    action_space = _spec_to_space(self._env.action_spec()[0])
    self.action_space = lru_cache(maxsize=None)(lambda agent_id: action_space)
    self.state_space = _spec_to_space(self._env.observation_spec()[0]['WORLD.RGB'])

  def state(self):
    return self._env.observation()

  def reset(self):
    """See base class."""
    timestep = self._env.reset()
    self.agents = self.possible_agents[:]
    self.num_cycles = 0
    return _timestep_to_observations(timestep)

  def step(self, action):
    """See base class."""
    actions = [action[agent] for agent in self.agents]
    timestep = self._env.step(actions)
    rewards = {
        agent: timestep.reward[index]
        for index, agent in enumerate(self.agents)
    }
    self.num_cycles += 1
    done = timestep.last() or self.num_cycles >= self.max_cycles 
    dones = {agent: done for agent in self.agents}
    infos = {agent: {} for agent in self.agents}
    if done:
        self.agents = []

    observations = _timestep_to_observations(timestep)
    return observations, rewards, dones, infos

  def close(self):
    """See base class."""
    self._env.close()
  
  def seed(self, seed=None):
    raise NotImplementedError
  
  def render(self, mode="human", filename=None):
    rgb_arr = self.state()['WORLD.RGB']
    if mode == "human":
        plt.cla()
        plt.imshow(rgb_arr, interpolation="nearest")
        if filename is None:
            plt.show(block=False)
        else:
            plt.savefig(filename)
        return None
    return rgb_arr

class _parallel_env(MeltingPotPettingZooEnv, EzPickle):
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, env_config, max_cycles):
        EzPickle.__init__(self, env_config, max_cycles)
        super().__init__(env_config, max_cycles)
