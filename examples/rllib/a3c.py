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

from ray import init
from ray.rllib.policy.policy import PolicySpec
from ray.tune import run
from ray.tune.registry import register_env
from tensorflow.keras.optimizers import RMSprop

from examples.rllib import utils
from meltingpot.python import substrate

SUBSTRATE_NAME = "commons_harvest_open"
EXPERIMENT_NAME = "commons_harvest_open_a3c"


def create_model_config() -> dict:
  """Create the model config
      
    ## Melting Pot paper
    
    > The agent's network consists of a convNet with two layers with 16, 32
    output channels, kernel shapes 8, 4, and strides 8, 1 respectively. It is
    followed by an MLP with two layers with 64 neurons each. All activation
    functions are ReLU. It is followed by an LSTM with 128 units. Policy and
    baseline (for the critic) are produced by linear layers connected to the
    output of LSTM. We used an auxiliary loss (Jaderberg et al., 2016) for
    shaping the representation using contrastive predictive coding (Oord et al.,
    2018). CPC was discriminating between nearby time points via LSTM state
    representations (a standard augmentation in recent works with A3C). 

    Note we have not included contrastive predictive coding, but this could be
    done using a custom PyTorch model along the lines of the [original paper
    code](https://github.com/jefflai108/Contrastive-Predictive-Coding-PyTorch).
    
    ## Ray RlLib defaults
    
    https://docs.ray.io/en/latest/rllib/rllib-models.html
    """
  return {
      # The last conv filter is added as a hack to flatten, because their
      # default approach to this doesn't work. This means we have
      # correspondingly also removed one of the two 64x MLP layers as it acts in
      # a similar way to this last filter.
      # https://docs.ray.io/en/latest/rllib/rllib-models.html#built-in-models
      # https://github.com/ray-project/ray/issues/21736
      # [out_channels, kernel, stride]
      "conv_filters": [[16, 8, 8], [32, 4, 1], [64, 11, 1]],
      "conv_activation": "relu",
      "no_final_linear": False,
      "post_fcnet_hiddens": [64],
      "post_fcnet_activation": "relu",
      "use_lstm": True,
      "lstm_cell_size": 128,
  }


def create_optimizer() -> RMSprop:
  """Create the RMSprop optimizer
    
    ## Melting Pot paper
    
    >We used RMSProp optimizer with learning rate of 4e-4, epsilon = 10e-5 and
    momentum set to zero and decay of 0.99. 
    """
  return RMSprop(
      learning_rate=4e-4,
      epsilon=10e-5,
      momentum=0.0,
      rho=0.99,  # decay
  )


def create_config(observation_space, action_space, env_config) -> dict:
  """Create the trainer config
    
    ## Melting Pot paper
    
    >  Baseline cost 0.5, entropy regularizer for policy at 0.003. All agents used discount rate gamma = 0.99.
    
    # RlLib Documentation

    https://docs.ray.io/en/latest/rllib/rllib-training.html#basic-python-api
    https://docs.ray.io/en/latest/rllib/rllib-algorithms.html#a3c

    """
  # We use the same policy for all agents
  policies = {
      "a3c":
          PolicySpec(
              observation_space=observation_space, action_space=action_space)
  }

  horizon = env_config.lab2d_settings["maxEpisodeLengthFrames"]

  video_dir = join(dirname(realpath(__file__)), ".videos", EXPERIMENT_NAME)

  general_config = {
      # Debugging
      "log_level": "ERROR",
      # Enviroment specific
      "env": "meltingpot",
      "env_config": env_config,
      # Evaluation settings (record videos as we go)
      "evaluation_num_workers": 0,  # Don't dedicate a full time core, as this runs very rarely
      "create_env_on_driver": True,  # Run as blocking, on the main thread
      "evaluation_interval": 100,  # Run every x iterations
      "evaluation_config": {
          "record_env": video_dir,
      },
      # Framework settings
      # NOTE: A3C does not work with tf2
      # Multi Agent
      "multiagent": {
          "policies":
              policies,
          "policy_mapping_fn":
              lambda agent_id, episode, worker, **kwargs: "a3c",
      },
      # Resources
      "num_gpus": 1,
      "num_workers":
          cpu_count() - 1,  # 1 core is reserved for the main thread
      # Rollouts
      "horizon": horizon,
      # Training
      "gamma": 0.99,
      "model": create_model_config(),
      "optimizer": create_optimizer(),
  }

  a3c_specific_config = {"entropy_coeff": 0.003, "vf_loss_coeff": 0.5}

  return {**general_config, **a3c_specific_config}


def train():
  """Train a multi-agent model
  
  Uses the hyperparameters from the paper behind Melting Pot - [Scalable
  Evaluation of Multi-Agent Reinforcement Learning with Melting
  Pot](https://arxiv.org/abs/2107.06857)"""

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
      resume="AUTO",
      stop={
          "timesteps_total": 1e9  # From the paper (p9)
      },
  )


if __name__ == "__main__":
  train()
