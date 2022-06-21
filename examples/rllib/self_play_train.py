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

import ray
from ray.rllib.agents.registry import get_trainer_class
from ray.rllib.policy.policy import PolicySpec
from ray.tune import tune
from ray.tune.registry import register_env

from examples.rllib import utils
from meltingpot.python import substrate


def main():
  # We need the following 2 pieces to run the training:
  # 1. The agent algorithm to use.
  agent_algorithm = "PPO"
  # 2. The name of the MeltingPot substrate, coming
  # from substrate.AVAILABLE_SUBSTRATES.
  substrate_name = "bach_or_stravinsky_in_the_matrix"

  # 1. Gets default training configuration and specifies the POMgame to load.
  config = copy.deepcopy(
      get_trainer_class(agent_algorithm).get_default_config())

  # 2. Set environment config. This will be passed to
  # the env_creator function via the register env lambda below.
  config["env_config"] = substrate.get_config(substrate_name)

  # 3. Register env
  register_env("meltingpot", utils.env_creator)
  config["env"] = "meltingpot"

  # 4. Extract space dimensions
  test_env = utils.env_creator(config["env_config"])

  # 5. Configuration for multiagent setup with policy sharing:
  config["multiagent"] = {
      "policies": {
          "av":
              PolicySpec(
                  policy_class=None,  # use default policy
                  observation_space=test_env.observation_space["player_0"],
                  action_space=test_env.action_space["player_0"],
                  config={}),
      },
      "policy_mapping_fn": lambda agent_id, episode, worker, **kwargs: "av",
      "count_steps_by": "env_steps",
  }

  # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
  config["num_gpus"] = int(os.environ.get("RLLIB_NUM_GPUS", "0"))
  config["log_level"] = "DEBUG"
  config["num_workers"] = 1
  # After n steps, force reset simulation
  config["horizon"] = config["env_config"].lab2d_settings[
      "maxEpisodeLengthFrames"]
  # each unroll happens exactly over one episode
  # ensuring that we have a full episode allows a reward to be calculated
  config["batch_mode"] = "complete_episodes"
  # Fragment length, in this case the number of episodes
  # collected at once from each worker and for each agent!
  config["rollout_fragment_length"] = 1
  config["train_batch_size"] = config["env_config"].lab2d_settings[
      "maxEpisodeLengthFrames"]
  # fix the batch size for a training pass
  # Default: 128
  config["sgd_minibatch_size"] = 128
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

  # 6. Initialize ray, train and save
  ray.init()

  tune.run(
      agent_algorithm,
      stop={"training_iteration": 1},
      config=config,
      metric="episode_reward_mean",
      mode="max",
      checkpoint_at_end=True)


if __name__ == "__main__":
  main()
