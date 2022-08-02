# Copyright 2020 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from multiprocessing import cpu_count
from os.path import dirname
from os.path import join
from os.path import realpath
import random

from gym.spaces import dict as gym_dict
from gym.spaces import discrete
from ray import init
from ray.rllib.evaluation import episode
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.utils import typing as rllib_typing
from ray.tune import run
from ray.tune.registry import register_env
from tensorflow import config as tensorflow_config
from tensorflow.keras.optimizers import RMSprop

from examples.rllib import utils
from meltingpot.python import substrate

SUBSTRATE_NAME = "commons_harvest_open"
EXPERIMENT_NAME = "commons_harvest_open_a3c"


def create_model_config() -> rllib_typing.ModelConfigDict:
  """Create the model config
      
    "The agent's network consists of a convNet with two layers with 16, 32
    output channels, kernel shapes 8, 4, and strides 8, 1 respectively. It is
    followed by an MLP with two layers with 64 neurons each. All activation
    functions are ReLU. It is followed by an LSTM with 128 units. Policy and
    baseline (for the critic) are produced by linear layers connected to the
    output of LSTM. We used an auxiliary loss (Jaderberg et al., 2016) for
    shaping the representation using contrastive predictive coding (Oord et al.,
    2018). CPC was discriminating between nearby time points via LSTM state
    representations (a standard augmentation in recent works with A3C)." -
    Scalable Evaluation of Multi-Agent Reinforcement Learning with Melting Pot

    Note we have not included contrastive predictive coding, but this could be
    done using a custom PyTorch model along the lines of the original paper
    code at https://github.com/jefflai108/Contrastive-Predictive-Coding-PyTorch.
    
    https://docs.ray.io/en/latest/rllib/rllib-models.html

    Returns:
      rllib_typing.ModelConfigDict: RlLib model config
    """
  return {
      # Conv filters last layer added as RlLib suggests this as a hack to
      # flatten, because their default flatten approach is broken. This means we
      # have correspondingly also removed one of the two 64x MLP layers as it
      # acts in a similar way to this last filter.
      # https://docs.ray.io/en/latest/rllib/rllib-models.html#built-in-models
      # https://github.com/ray-project/ray/issues/21736
      # [out_channels, kernel, stride]
      "conv_filters": [[16, 8, 8], [32, 4, 1], [64, 11, 1]],
      "conv_activation": "relu",
      # "no_final_linear": False,  # Flatten (doesn't work - see conv_filters)
      "post_fcnet_hiddens": [64],
      "post_fcnet_activation": "relu",
      "use_lstm": True,
      "lstm_cell_size": 128,
  }


def create_optimizer() -> RMSprop:
  """Create the RMSprop optimizer
    
    "We used RMSProp optimizer with learning rate of 4e-4, epsilon = 10e-5 and
    momentum set to zero and decay of 0.99." -
    Scalable Evaluation of Multi-Agent Reinforcement Learning with Melting Pot

    https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/RMSprop

    Returns:
      RMSprop: TensorFlow RMSProp optimizer
    """
  return RMSprop(
      learning_rate=4e-4,
      epsilon=10e-5,
      momentum=0.0,
      rho=0.99,  # decay
  )


class IndependentAgents:
  """Independent Agents

  Creates a set of independent agents, with no shared parameters or
  observations. This is the approach the paper took (it is not mentioned
  explicitly).

  For each player within an episode, `policy_mapping_fn` can then be called to
  assign an agent's policy to a random player (so that the agent will end up
  learning from all the different player starting points).

  Attributes:
    policies: A dict containing RlLib policies with keys `agent_x` (one per
    agent).
  """

  def __init__(self, observation_space: gym_dict.Dict,
               action_space: discrete.Discrete, num_players: int):
    """Create the set of independent agents
    
    Args:
      observation_space (gym_dict.Dict): Gym observation space
      action_space (discrete.Discrete): Gym action space
      num_players (int): Number of players in the substrate

    https://docs.ray.io/en/latest/rllib/rllib-env.html#multi-agent-and-hierarchical
    """
    self.policies: rllib_typing.MultiAgentPolicyConfigDict = {}

    self.__remaining_agents: list[rllib_typing.PolicyID] = []

    # Create an independent policy for each agent. The policies are named
    # `agent_x` (where x is the policy number).
    for player_id in range(num_players):
      self.policies[f"agent_{player_id}"] = PolicySpec(
          observation_space=observation_space, action_space=action_space)

  def policy_mapping_fn(self,
                        agent_id: rllib_typing.AgentID,
                        episode: episode.Episode = None,
                        **kwargs) -> rllib_typing.PolicyID:
    """Policy mapping function

    "Every agent participated in every training episode, with each agent playing
    as exactly one player (selected randomly on each episode)." -
    Scalable Evaluation of Multi-Agent Reinforcement Learning with Melting Pot

    Return a random policy ID from the set of those available, so that each
    player receives a random policy for each episode.

    Args:
      agent_id (str): Agent ID (of the form `player_x`)

    Returns:
      str: Policy ID
    """
    # If this is the first player, reset __remaining_agents to include all
    # policies. These are shuffled so that a random one can be chosen for each
    # player.
    if len(self.__remaining_agents) == 0:
      self.__remaining_agents = list(self.policies.keys())
      random.shuffle(self.__remaining_agents)

    # Get a random policy ID, from those that have not yet been used for this
    # episode
    return self.__remaining_agents.pop()


def count_available_gpus() -> int:
  """Count the number of GPUs available on the host machine

  Returns:
    int: Number of GPUs that are available to TensorFlow
  """
  gpus = tensorflow_config.list_physical_devices('GPU')
  return len(gpus)


def create_config(observation_space: gym_dict.Dict,
                  action_space: discrete.Discrete,
                  env_config: dict) -> rllib_typing.TrainerConfigDict:
  """Create the trainer config
    
  "Baseline cost 0.5, entropy regularizer for policy at 0.003. All agents
  used discount rate gamma = 0.99." -
  Scalable Evaluation of Multi-Agent Reinforcement Learning with Melting Pot
  
  https://docs.ray.io/en/latest/rllib/rllib-training.html#basic-python-api
  https://docs.ray.io/en/latest/rllib/rllib-algorithms.html#a3c

  Args:
    observation_space (gym_dict.Dict): Gym observation space
    action_space (discrete.Discrete): Gym action space
    env_config (dict): Meltingpot substrate config

  Returns:
    rllib_typing.TrainerConfigDict: RlLib A3C config
  """
  num_players = env_config["num_players"]

  agents = IndependentAgents(observation_space, action_space, num_players)

  horizon = env_config.lab2d_settings["maxEpisodeLengthFrames"]

  video_dir = join(dirname(realpath(__file__)), ".videos", EXPERIMENT_NAME)

  general_config = {
      # Debugging
      "log_level": "ERROR",
      # Environment specific
      "env": "meltingpot",
      "env_config": env_config,
      # Evaluation settings (record videos as we go)
      "evaluation_num_workers": 0,  # Don't dedicate a full time core
      "create_env_on_driver": True,  # Run as blocking, on the main thread
      "evaluation_interval": 100,  # Run every x iterations
      "evaluation_config": {
          "record_env": video_dir,
      },
      # Multi Agent
      "multiagent": {
          "policies": agents.policies,
          "policy_mapping_fn": agents.policy_mapping_fn
      },
      # Resources
      "num_gpus": count_available_gpus(),
      "num_workers": cpu_count() - 1,  # 1 core is reserved for the main thread
      # Rollouts
      "horizon": horizon,
      # Training
      "gamma": 0.99,
      "model": create_model_config(),
      "optimizer": create_optimizer(),
  }

  a3c_specific_config = {"entropy_coeff": 0.003, "vf_loss_coeff": 0.5}

  return {**general_config, **a3c_specific_config}


def train() -> None:
  """Train a multi-agent model
  
  Uses the hyperparameters from the paper behind Melting Pot - Scalable
  Evaluation of Multi-Agent Reinforcement Learning with Melting
  Pot - https://arxiv.org/abs/2107.06857 .

  You can view the progress of training on TensorBoard, using:
  
  `tensorboard --logdir=examples/rllib/.results/<<EXPERIMENT_NAME>>`
  """

  # Initialize Ray
  init(configure_logging=True)

  # Setup the training environment (MeltingPot substrate)
  env_config = substrate.get_config(SUBSTRATE_NAME)
  register_env("meltingpot", utils.env_creator)
  environment = utils.env_creator(env_config)

  # Get trainer config
  config = create_config(environment.observation_space["player_0"],
                         environment.action_space["player_0"], env_config)

  # Train
  run(
      "A3C",
      checkpoint_at_end=True,
      checkpoint_freq=50,
      config=config,
      local_dir=join(dirname(realpath(__file__)), ".results"),
      log_to_file=True,
      metric="episode_reward_mean",
      mode="max",
      name=EXPERIMENT_NAME,
      # Set resume to `True` if you want to resume a training run that you've
      # stopped, and it will automatically use your last checkpoint from
      # the `local_dir` set above. However please note that when you do this, it
      # will also resume with your config from there, and will therefore not pick
      # up any changes you've subsequently made here.
      resume=False,
      stop={
          "timesteps_total": 1e9  # From the paper (p9)
      },
  )


if __name__ == "__main__":
  train()
