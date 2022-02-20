import argparse

import gym
import supersuit as ss
import torch
import torch.nn.functional as F
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env.vec_monitor import VecMonitor
from stable_baselines3.common.vec_env.vec_transpose import VecTransposeImage
from torch import nn

from examples.pettingzoo.meltingpot_env import parallel_env
from meltingpot.python import substrate

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


# Use this with lambda wrapper returning observations only
class CustomCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        features_dim=128,
        num_frames=6,
        fcnet_hiddens=[1024, 128],
    ):
        super(CustomCNN, self).__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper

        self.conv = nn.Sequential(
            nn.Conv2d(num_frames * 3, num_frames * 3, kernel_size=8, stride=4, padding=0),
            nn.ReLU(), # 18 * 21 * 21
            nn.Conv2d(num_frames * 3, num_frames * 6, kernel_size=5, stride=2, padding=0),
            nn.ReLU(), # 36 * 9 * 9
            nn.Conv2d(num_frames * 6, num_frames * 6, kernel_size=3, stride=1, padding=0),
            nn.ReLU(), # 36 * 7 * 7
            nn.Flatten()
        )
        flat_out = num_frames * 6 * 7 * 7
        self.fc1 = nn.Linear(in_features=flat_out, out_features=fcnet_hiddens[0])
        self.fc2 = nn.Linear(in_features=fcnet_hiddens[0], out_features=fcnet_hiddens[1])

    def forward(self, observations) -> torch.Tensor:
        # Convert to tensor, rescale to [0, 1], and convert from B x H x W x C to B x C x H x W
        observations = observations.permute(0, 3, 1, 2)
        features = self.conv(observations)
        features = F.relu(self.fc1(features))
        features = F.relu(self.fc2(features))
        return features


def main():
    # Config
    env_name = "commons_harvest_open"
    env_config = substrate.get_config(env_name)
    env = parallel_env(env_config)
    rollout_len = 1000
    total_timesteps = 16000000

    # Training
    num_cpus = 1  # number of cpus
    num_envs = 1  # number of parallel multi-agent environments
    num_frames = 6  # number of frames to stack together; use >4 to avoid automatic VecTransposeImage
    features_dim = (
        128  # output layer of cnn extractor AND shared layer for policy and value functions
    )
    fcnet_hiddens = [1024, 128]  # Two hidden layers for cnn extractor
    ent_coef = 0.001  # entropy coefficient in loss
    batch_size = rollout_len * num_envs // 2  # This is from the rllib baseline implementation
    lr = 0.0001
    n_epochs = 30
    gae_lambda = 1.0
    gamma = 0.99
    target_kl = 0.01
    grad_clip = 40
    verbose = 3

    env = parallel_env(
        max_cycles=rollout_len,
        env_config=env_config,
    )
    env = ss.observation_lambda_v0(env, lambda x, _: x["RGB"], lambda s: s["RGB"])
    env = ss.frame_stack_v1(env, num_frames)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(
        env, num_vec_envs=num_envs, num_cpus=num_cpus, base_class="stable_baselines3"
    )
    env = VecMonitor(env)
    env = VecTransposeImage(env, True)

    policy_kwargs = dict(
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(
            features_dim=features_dim, num_frames=num_frames, fcnet_hiddens=fcnet_hiddens
        ),
        net_arch=[features_dim],
    )

    tensorboard_log = "./results/sb3/harvest_open_ppo_paramsharing"

    model = PPO(
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
    model.learn(total_timesteps=total_timesteps)

    logdir = model.logger.dir
    model.save(logdir + "/model")
    del model
    model = PPO.load(logdir + "/model")  # noqa: F841


if __name__ == "__main__":
    main()
