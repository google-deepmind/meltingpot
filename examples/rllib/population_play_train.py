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
"""Runs an example of a self-play training experiment."""

import os

import ray
from ray import air
from ray import tune
from ray.rllib.algorithms import ppo
from ray.rllib.policy.policy import PolicySpec
from ray.tune.registry import register_env
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.policy.policy import Policy



from examples.rllib import utils
from meltingpot.python import substrate

import random



def get_config(
    substrate_name: str = "stag_hunt_in_the_matrix__repeated",
    num_rollout_workers: int = 2,
    rollout_fragment_length: int = 100,
    train_batch_size: int = 1600,
    fcnet_hiddens=(64, 64),
    post_fcnet_hiddens=(256,),
    lstm_cell_size: int = 256,
    sgd_minibatch_size: int = 128,
):
  """Get the configuration for running an agent on a substrate using RLLib.
  
  We need the following 2 pieces to run the training:

  Args:
    substrate_name: The name of the MeltingPot substrate, coming from
      `substrate.AVAILABLE_SUBSTRATES`. 
    num_rollout_workers: The number of workers for playing games.
    rollout_fragment_length: Unroll time for learning.
    train_batch_size: Batch size (batch * rollout_fragment_length)
    fcnet_hiddens: Fully connected layers.
    post_fcnet_hiddens: Layer sizes after the fully connected torso.
    lstm_cell_size: Size of the LSTM.
    sgd_minibatch_size: Size of the mini-batch for learning.
  
  Returns:
    The configuration for running the experiment.
  """
  # Gets the default training configuration
  config = ppo.PPOConfig()
  # Number of arenas. 
  # This is called num_rollout_workers in 2.2.0.
  config.num_rollout_workers = num_rollout_workers
  # This is to match our unroll lengths.
  config.rollout_fragment_length = rollout_fragment_length
  # Total (time x batch) timesteps on the learning update.
  config.train_batch_size = train_batch_size
  # Mini-batch size.
  config.sgd_minibatch_size = sgd_minibatch_size
  # Use the raw observations/actions as defined by the environment.
  config.preprocessor_pref = None
  # Use TensorFlow as the tensor framework.
  config = config.framework("tf")
  # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
  config.num_gpus = int(os.environ.get("RLLIB_NUM_GPUS", "0"))
  config.log_level = "DEBUG"

  # 2. Set environment config. This will be passed to
  # the env_creator function via the register env lambda below.
  player_roles = substrate.get_config(substrate_name).default_player_roles
  config.env_config = {"substrate": substrate_name, "roles": player_roles}

  config.env = "meltingpot"

  # 4. Extract space dimensions
  test_env = utils.env_creator(config.env_config)
  register_env("meltingpot", lambda env_config: utils.env_creator(config.env_config))

  # Setup PPO with policies, one per entry in default player roles.
  policies = {}
  player_to_agent = {}
  for i in range(len(player_roles)):
    rgb_shape = test_env.observation_space[f"player_{i}"]["RGB"].shape
    sprite_x = rgb_shape[0] // 8
    sprite_y = rgb_shape[1] // 8

    policies[f"agent_{i}"] = PolicySpec(
        policy_class=None,  # use default policy
        observation_space=test_env.observation_space[f"player_{i}"],
        action_space=test_env.action_space[f"player_{i}"],
        config={
            "model": {
                "conv_filters": [[16, [8, 8], 8],
                                 [128, [sprite_x, sprite_y], 1]],
            },
        })
    player_to_agent[f"player_{i}"] = f"agent_{i}"

  def policy_mapping_fn(agent_id, **kwargs):
    del kwargs
    return player_to_agent[agent_id]

  # 5. Configuration for multi-agent setup with one policy per role:
  config.multi_agent(policies=policies, policy_mapping_fn=policy_mapping_fn)

  # 6. Set the agent architecture.
  # Definition of the model architecture.
  # The strides of the first convolutional layer were chosen to perfectly line
  # up with the sprites, which are 8x8.
  # The final layer must be chosen specifically so that its output is
  # [B, 1, 1, X]. See the explanation in
  # https://docs.ray.io/en/latest/rllib-models.html#built-in-models. It is
  # because rllib is unable to flatten to a vector otherwise.
  # The acb models used as baselines in the meltingpot paper were not run using
  # rllib, so they used a different configuration for the second convolutional
  # layer. It was 32 channels, [4, 4] kernel shape, and stride = 1.
  config.model["fcnet_hiddens"] = fcnet_hiddens
  config.model["fcnet_activation"] = "relu"
  config.model["conv_activation"] = "relu"
  config.model["post_fcnet_hiddens"] = post_fcnet_hiddens
  config.model["post_fcnet_activation"] = "relu"
  config.model["use_lstm"] = True
  config.model["lstm_use_prev_action"] = True
  config.model["lstm_use_prev_reward"] = False
  config.model["lstm_cell_size"] = lstm_cell_size

  return config


def main():
  

  config = get_config()

  tune.register_env("meltingpot", utils.env_creator)

  # 6. Initialize ray, train and save
  ray.init()

  train_batch_size = 1600
  total_timesteps = 5000000
  num_iters = total_timesteps // train_batch_size


  seeds = ["Seed_0","Seed_1","Seed_2","Seed_3","Seed_4"]
  num_seeds = len(seeds)

  checkpoints_dict = {}                                 # A dictionary of lists of checkpoints for each seed/population
  for i in seeds:
    checkpoints_dict[i] = []



  for i in range(num_iters):

    for seed in range(num_seeds):

        algo_config = config.build()
        path_to_checkpoint = algo_config.save(f"checkpoints/seed_{seed}")
        checkpoints_dict[f"Seed_{seed}"].append(path_to_checkpoint)

        print("An algo_config is saved at: ", path_to_checkpoint)
      
        # Set Player 1's policy
        if (len(checkpoints_dict[f"Seed_{seed}"]) > 0):
            algo_config = algo_config.from_checkpoint(checkpoints_dict[f"Seed_{seed}"].pop(), policy_ids={"agent_0"})

        # Set Player 2's policy
        swap_seed = random.randint(0, num_seeds - 1)
        if (len(checkpoints_dict[f"Seed_{swap_seed}"]) > 0):
            algo_config = algo_config.from_checkpoint(checkpoints_dict[f"Seed_{swap_seed}"].pop(), policy_ids={"agent_1"})

        config = algo_config.get_config()
        
            
        stop = {
        "training_iteration": 2,                       # train_batch_size = 1600
        }

        results = tune.Tuner(
        "PPO",
        param_space=config.to_dict(),
        run_config=air.RunConfig(name= "PPO_expt", stop=stop, checkpoint_config = air.CheckpointConfig(checkpoint_frequency = 20),verbose=1),
        )
        results.fit()
        path_to_checkpoint = algo_config.save(f"checkpoints/seed_{seed}")
        checkpoints_dict[f"Seed_{seed}"].append(path_to_checkpoint)

if __name__ == "__main__":  
  main()


