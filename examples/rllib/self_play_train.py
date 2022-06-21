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
from ray.tune.schedulers import PopulationBasedTraining
from ray.tune.registry import register_env

from examples.rllib import utils
from meltingpot.python import substrate


def main():
  # We need the following 2 pieces to run the training:
  # 1. The agent algorithm to use.
  agent_algorithm = "PPO"
  # 2. The name of the MeltingPot substrate, coming
  # from substrate.AVAILABLE_SUBSTRATES.
  substrate_name = "commons_harvest_open_simple"
  num_players = 1

  # 1. Gets default training configuration and specifies the POMgame to load.
  config = copy.deepcopy(
      get_trainer_class(agent_algorithm).get_default_config())

  # 2. Set environment config. This will be passed to
  # the env_creator function via the register env lambda below.
  config["env_config"] = substrate.get_config(substrate_name)
  assert (config["env_config"]["num_players"] == num_players)

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
  config["log_level"] = "WARN"
  config["num_workers"] = 6
  config["num_envs_per_worker"] = 1
  config["num_cpus_per_worker"] = 1
  # After n steps, force reset simulation
  config["horizon"] = config["env_config"].lab2d_settings[
      "maxEpisodeLengthFrames"]
  # complete_episodes: Each unroll happens exactly over one episode, from
  #   beginning to end. Data collection will not stop unless the episode
  #   terminates or a configured horizon (hard or soft) is hit.
  config["batch_mode"] = "complete_episodes"
  # Divide episodes into fragments of this many steps each during rollouts.
  # Sample batches of this size are collected from rollout workers and
  # combined into a larger batch of `train_batch_size` for learning.
  config["rollout_fragment_length"] = config["horizon"]
  # Training batch size, if applicable. Should be >= rollout_fragment_length.
  # Samples batches will be concatenated together to a batch of this size,
  # which is then passed to SGD.
  config["train_batch_size"] = 2 * config["num_workers"] * config[
          "horizon"] * config["num_envs_per_worker"]
  # fix the batch size for a training pass
  # TODO: this is measured in complete fragments too?
  # Default: 128
  config["sgd_minibatch_size"] = 128
  # Default: 30
  config["num_sgd_iter"] = 30

  config["clip_rewards"] = False
  config["normalize_actions"] = True


  # learning rate
  config["lr"] = 1e-4
  # discount rate
  config["gamma"] = 0.99

  # wtf
  config["normalize_actions"] = True
  # config["exploration_config"] = {"type": "StochasticSampling"}

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
  config["model"]["conv_filters"] = [[16, [8, 8], 8], [128, [7, 7], 1]]
  config["model"]["conv_activation"] = "relu"
  config["model"]["post_fcnet_hiddens"] = [128]
  config["model"]["post_fcnet_activation"] = "relu"
  config["model"]["use_lstm"] = True
  config["model"]["lstm_use_prev_action"] = True
  config["model"]["lstm_use_prev_reward"] = False
  config["model"]["lstm_cell_size"] = 128

  # trainer_config["metrics_num_episodes_for_smoothing"] = 1
  # trainer_config["always_attach_evaluation_results"] = True
  # trainer_config["evaluation_interval"] = 1
  # trainer_config["evaluation_duration"] = 1

  # trainer = get_trainer_class(agent_algorithm)(config=trainer_config)
  # trainer.restore("./save/1p_harvest_simple/checkpoint_000150/checkpoint-150")

  # result = trainer.train()
  # print(pretty_print(result))

  # 6. Initialize ray, train and save
  ray.init(logging_level="info")

  scheduler = PopulationBasedTraining(
    # time_attr="time_total_s",
    time_attr="training_iteration",
    metric="episode_reward_mean",
    mode="max",
    perturbation_interval=5,
    burn_in_period=10,
    hyperparam_mutations={
      "lr": [1e-3, 3e-4, 1e-4, 3e-5, 1e-5],
    },
    quantile_fraction=0.25,
    resample_probability=0.25,
    log_config=True
)

  # TODO: use torch or tf2
  tune.run(
      agent_algorithm,
      stop={"training_iteration": 1000},
      scheduler=scheduler,
      config=config,
      log_to_file=True,
      checkpoint_freq=25,
      checkpoint_at_end=True,
      num_samples=4,  # how many trials to run concurrently
  )


if __name__ == "__main__":
  main()
