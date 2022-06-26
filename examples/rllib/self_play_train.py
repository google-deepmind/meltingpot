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

import os
import ray
from ray.rllib.policy.policy import PolicySpec
from ray.tune import tune
from ray.tune.schedulers import PopulationBasedTraining
from ray.tune.registry import register_env
from ray.rllib.agents.ppo import PPOConfig
from examples.rllib import utils
from meltingpot.python import substrate

# Setup for the neural network.
# The strides of the first convolutional layer were chosen to perfectly line
# up with the sprites, which are 8x8.
# The final layer must be chosen specifically so that its output is
# [B, 1, 1, X]. See the explanation in
# https://docs.ray.io/en/latest/rllib-models.html#built-in-models. It is
# because rllib is unable to flatten to a vector otherwise.
# The a3c models used as baselines in the meltingpot paper were not run using
# rllib, so they used a different configuration for the second convolutional
# layer. It was 32 channels, [4, 4] kernel shape, and stride = 1.
custom_model = {
  "conv_filters": [[16, [8, 8], 8], [128, [7, 7], 1]],
  "conv_activation": "relu",
  "post_fcnet_hiddens": [128],
  "post_fcnet_activation": "relu",
  "use_lstm": True,
  "lstm_use_prev_action": True,
  "lstm_use_prev_reward": False,
  "lstm_cell_size": 128
}


def main():
  """
  Example of a self-play training experiment with RLLib
  """

  # Setup the training environment (MeltingPot subscrate)
  substrate_name = "commons_harvest_open_simple"
  env_config = substrate.get_config(substrate_name)
  assert env_config["num_players"] == 1
  register_env("meltingpot", utils.env_creator)

  # Configure the horizon
  # After n steps, force reset simulation
  # Each unroll happens exactly over one episode, from
  # beginning to end. Data collection will not stop unless the episode
  # terminates or a configured horizon (hard or soft) is hit.
  horizon = env_config.lab2d_settings["maxEpisodeLengthFrames"]

  # Extract space dimensions
  test_env = utils.env_creator(env_config)

  # Configure the PPO Alogrithm
  config = PPOConfig().training(
    model=custom_model,
    gamma=0.99,
    lr=1e-4
    # Previously was 2 * config["num_workers"] * horizon *
    # config["num_envs_per_worker"]
    # train_batch_size=4000, # Default=4000
    # num_sgd_iter=30, # Default=30
    # sgd_minibatch_size = 128 # Default=128
  ).resources(
    # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
    num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0")),
    num_cpus_per_worker=1
  ).rollouts(
    num_rollout_workers=0,
    horizon=horizon,
    batch_mode="complete_episodes",
    rollout_fragment_length=horizon
  ).environment(
    env="meltingpot",
    clip_rewards=False,
    normalize_actions=True,
    env_config=env_config
  ).multi_agent(
    policies={
      "av":
      PolicySpec(
        policy_class=None,  # use default policy
        observation_space=test_env.observation_space["player_0"],
        action_space=test_env.action_space["player_0"],
        config={}),
    },
    policy_mapping_fn=lambda agent_id, episode, worker, **kwargs: "av",
    count_steps_by="env_steps"
  )

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
    log_config=True)

  # TODO: use torch or tf2
  tune.run(
    "PPO",
    stop={"training_iteration": 1000},
    scheduler=scheduler,
    config=config.to_dict(),
    log_to_file=True,
    checkpoint_freq=25,
    checkpoint_at_end=True,
    num_samples=4,  # how many trials to run concurrentl
  )


if __name__ == "__main__":
  main()
