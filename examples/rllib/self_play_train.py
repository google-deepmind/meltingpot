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

import copy
import os

from ml_collections import config_dict
import multiagent_wrapper  # pylint: disable=g-bad-import-order
import ray
from ray.rllib.agents.registry import get_trainer_class
from ray.tune.registry import register_env

from meltingpot.python import substrate


def main():
  # We need the following 3 pieces to run the training:
  # 1. The agent algorithm to use.
  agent_algorithm = "PPO"
  # 2. The name of the MeltingPot substrate, coming
  # from substrate.AVAILABLE_SUBSTRATES.
  substrate_name = "allelopathic_harvest"
  # 3. The number of CPUs to be used to run the training.
  num_cpus = 1

  # function that outputs the environment you wish to register.
  def env_creator(env_config):
    env = substrate.build(config_dict.ConfigDict(env_config))
    env = multiagent_wrapper.MeltingPotEnv(env)
    return env

  # 1. Gets default training configuration and specifies the POMgame to load.
  config = copy.deepcopy(
      get_trainer_class(agent_algorithm)._default_config)  # pylint: disable=protected-access

  # 2. Set environment config. This will be passed to
  # the env_creator function via the register env lambda below.
  config["env_config"] = substrate.get_config(substrate_name)

  # 3. Register env
  register_env("meltingpot", env_creator)

  # 4. Extract space dimensions
  test_env = env_creator(config["env_config"])
  obs_space = test_env.single_player_observation_space()
  act_space = test_env.single_player_action_space()

  # 5. Configuration for multiagent setup with policy sharing:
  config["multiagent"] = {
      "policies": {
          # the first tuple value is None -> uses default policy
          "av": (None, obs_space, act_space, {}),
      },
      "policy_mapping_fn": lambda agent_id, **kwargs: "av"
  }

  # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
  config["num_gpus"] = int(os.environ.get("RLLIB_NUM_GPUS", "0"))
  config["log_level"] = "DEBUG"
  config["num_workers"] = 1
  # Fragment length, collected at once from each worker and for each agent!
  config["rollout_fragment_length"] = 30
  # Training batch size -> Fragments are concatenated up to this point.
  config["train_batch_size"] = 200
  # After n steps, force reset simulation
  config["horizon"] = 2000
  # Default: False
  config["no_done_at_end"] = False
  # Info: If False, each agents trajectory is expected to have
  # maximum one done=True in the last step of the trajectory.
  # If no_done_at_end = True, environment is not resetted
  # when dones[__all__]= True.

  # Setup for the neural network.
  config["model"] = {}
  # The strides of the first convolutional layer were chosen to perfectly line
  # up with the sprites, which are 8x8.
  # The final layer must be chosen specifically so that its output is
  # [B, 1, 1, X]. See the explanation in
  # https://docs.ray.io/en/latest/rllib-models.html#built-in-models. It is
  # because rllib is unable to flatten to a vector otherwise.
  # The a3c models used as baselines in the meltingpot paper were not run using
  # rllib, so they used a different configuration for the second convolutional
  # layer. It was 32 channels, [4, 4] kernel shape, and stride = 1.
  config["model"]["conv_filters"] = [[16, [8, 8], 8], [128, [11, 11], 1]]
  config["model"]["conv_activation"] = "relu"
  config["model"]["post_fcnet_hiddens"] = [256]
  config["model"]["post_fcnet_activation"] = "relu"
  config["model"]["use_lstm"] = True
  config["model"]["lstm_use_prev_action"] = True
  config["model"]["lstm_use_prev_reward"] = False
  config["model"]["lstm_cell_size"] = 256

  # 6. Initialize ray and trainer object
  ray.init(num_cpus=num_cpus + 1)
  trainer = get_trainer_class(agent_algorithm)(env="meltingpot", config=config)

  # 7. Train once
  trainer.train()


if __name__ == "__main__":
  main()
