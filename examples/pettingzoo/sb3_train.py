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
"""Binary to run Stable Baselines 3 agents on meltingpot substrates."""

import gymnasium as gym
from meltingpot import substrate
import stable_baselines3
from stable_baselines3.common import callbacks
from stable_baselines3.common import torch_layers
from stable_baselines3.common import vec_env
import supersuit as ss
import torch
from torch import nn
import torch.nn.functional as F

from . import utils

device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
    "cpu")


# Use this with lambda wrapper returning observations only
class CustomCNN(torch_layers.BaseFeaturesExtractor):
  """Class describing a custom feature extractor."""

  def __init__(
      self,
      observation_space: gym.spaces.Box,
      features_dim=128,
      num_frames=6,
      fcnet_hiddens=(1024, 128),
  ):
    """Construct a custom CNN feature extractor.

    Args:
      observation_space: the observation space as a gym.Space
      features_dim: Number of features extracted. This corresponds to the number
        of unit for the last layer.
      num_frames: The number of (consecutive) frames to feed into the network.
      fcnet_hiddens: Sizes of hidden layers.
    """
    super().__init__(observation_space, features_dim)
    # We assume CxHxW images (channels first)
    # Re-ordering will be done by pre-preprocessing or wrapper

    self.conv = nn.Sequential(
        nn.Conv2d(
            num_frames * 3, num_frames * 3, kernel_size=8, stride=4, padding=0),
        nn.ReLU(),  # 18 * 21 * 21
        nn.Conv2d(
            num_frames * 3, num_frames * 6, kernel_size=5, stride=2, padding=0),
        nn.ReLU(),  # 36 * 9 * 9
        nn.Conv2d(
            num_frames * 6, num_frames * 6, kernel_size=3, stride=1, padding=0),
        nn.ReLU(),  # 36 * 7 * 7
        nn.Flatten(),
    )
    flat_out = num_frames * 6 * 7 * 7
    self.fc1 = nn.Linear(in_features=flat_out, out_features=fcnet_hiddens[0])
    self.fc2 = nn.Linear(
        in_features=fcnet_hiddens[0], out_features=fcnet_hiddens[1])

  def forward(self, observations) -> torch.Tensor:
    # Convert to tensor, rescale to [0, 1], and convert from
    #   B x H x W x C to B x C x H x W
    observations = observations.permute(0, 3, 1, 2)
    features = self.conv(observations)
    features = F.relu(self.fc1(features))
    features = F.relu(self.fc2(features))
    return features


def main():
  # Config
  env_name = "commons_harvest__open"
  env_config = substrate.get_config(env_name)
  env = utils.parallel_env(env_config)
  rollout_len = 1000
  total_timesteps = 2000000
  num_agents = env.max_num_agents

  # Training
  num_cpus = 1  # number of cpus
  num_envs = 1  # number of parallel multi-agent environments
  # number of frames to stack together; use >4 to avoid automatic
  # VecTransposeImage
  num_frames = 4
  # output layer of cnn extractor AND shared layer for policy and value
  # functions
  features_dim = 128
  fcnet_hiddens = [1024, 128]  # Two hidden layers for cnn extractor
  ent_coef = 0.001  # entropy coefficient in loss
  batch_size = (rollout_len * num_envs // 2
               )  # This is from the rllib baseline implementation
  lr = 0.0001
  n_epochs = 30
  gae_lambda = 1.0
  gamma = 0.99
  target_kl = 0.01
  grad_clip = 40
  verbose = 3
  model_path = None  # Replace this with a saved model

  env = utils.parallel_env(
      max_cycles=rollout_len,
      env_config=env_config,
  )
  env = ss.observation_lambda_v0(env, lambda x, _: x["RGB"], lambda s: s["RGB"])
  env = ss.frame_stack_v1(env, num_frames)
  env = ss.pettingzoo_env_to_vec_env_v1(env)
  env = ss.concat_vec_envs_v1(
      env,
      num_vec_envs=num_envs,
      num_cpus=num_cpus,
      base_class="stable_baselines3")
  env = vec_env.VecMonitor(env)
  env = vec_env.VecTransposeImage(env, True)

  eval_env = utils.parallel_env(
      max_cycles=rollout_len,
      env_config=env_config,
  )
  eval_env = ss.observation_lambda_v0(eval_env, lambda x, _: x["RGB"],
                                      lambda s: s["RGB"])
  eval_env = ss.frame_stack_v1(eval_env, num_frames)
  eval_env = ss.pettingzoo_env_to_vec_env_v1(eval_env)
  eval_env = ss.concat_vec_envs_v1(
      eval_env, num_vec_envs=1, num_cpus=1, base_class="stable_baselines3")
  eval_env = vec_env.VecMonitor(eval_env)
  eval_env = vec_env.VecTransposeImage(eval_env, True)
  eval_freq = 100000 // (num_envs * num_agents)

  policy_kwargs = dict(
      features_extractor_class=CustomCNN,
      features_extractor_kwargs=dict(
          features_dim=features_dim,
          num_frames=num_frames,
          fcnet_hiddens=fcnet_hiddens,
      ),
      net_arch=[features_dim],
  )

  tensorboard_log = "./results/sb3/harvest_open_ppo_paramsharing"

  model = stable_baselines3.PPO(
      "CnnPolicy",
      env=env,
      learning_rate=lr,
      n_steps=rollout_len,
      batch_size=batch_size,
      n_epochs=n_epochs,
      gamma=gamma,
      gae_lambda=gae_lambda,
      ent_coef=ent_coef,
      max_grad_norm=grad_clip,
      target_kl=target_kl,
      policy_kwargs=policy_kwargs,
      tensorboard_log=tensorboard_log,
      verbose=verbose,
  )
  if model_path is not None:
    model = stable_baselines3.PPO.load(model_path, env=env)
  eval_callback = callbacks.EvalCallback(
      eval_env, eval_freq=eval_freq, best_model_save_path=tensorboard_log)
  model.learn(total_timesteps=total_timesteps, callback=eval_callback)

  logdir = model.logger.dir
  model.save(logdir + "/model")
  del model
  stable_baselines3.PPO.load(logdir + "/model")


if __name__ == "__main__":
  main()
