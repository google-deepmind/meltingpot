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
"""Runs the bots trained in self_play_train.py and renders in pygame.

You must provide experiment_state, expected to be
~/ray_results/PPO/experiment_state_YOUR_RUN_ID.json
"""

import argparse

import dm_env
from dmlab2d.ui_renderer import pygame
import numpy as np
from ray.rllib.algorithms.registry import get_trainer_class
from ray.tune.analysis.experiment_analysis import ExperimentAnalysis
from ray.tune.registry import register_env

from . import utils


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
      "--experiment_state",
      type=str,
      default="~/ray_results/PPO",
      help="ray.tune experiment_state to load. The default setting will load"
      " the last training run created by self_play_train.py. If you want to use"
      " a specific run, provide a path, expected to be of the format "
      " ~/ray_results/PPO/experiment_state-DATETIME.json")

  args = parser.parse_args()

  agent_algorithm = "PPO"

  register_env("meltingpot", utils.env_creator)

  experiment = ExperimentAnalysis(
      args.experiment_state,
      default_metric="episode_reward_mean",
      default_mode="max")

  config = experiment.best_config
  checkpoint_path = experiment.best_checkpoint

  trainer = get_trainer_class(agent_algorithm)(config=config)
  trainer.restore(checkpoint_path)

  # Create a new environment to visualise
  env = utils.env_creator(config["env_config"]).get_dmlab2d_env()

  bots = [
      utils.RayModelPolicy(trainer, f"agent_{i}")
      for i in range(len(config["env_config"]["default_player_roles"]))
  ]

  timestep = env.reset()
  states = [bot.initial_state() for bot in bots]
  actions = [0] * len(bots)

  # Configure the pygame display
  scale = 4
  fps = 5

  pygame.init()
  clock = pygame.time.Clock()
  pygame.display.set_caption("DM Lab2d")
  obs_spec = env.observation_spec()
  shape = obs_spec[0]["WORLD.RGB"].shape
  game_display = pygame.display.set_mode(
      (int(shape[1] * scale), int(shape[0] * scale)))

  for _ in range(config["horizon"]):
    obs = timestep.observation[0]["WORLD.RGB"]
    obs = np.transpose(obs, (1, 0, 2))
    surface = pygame.surfarray.make_surface(obs)
    rect = surface.get_rect()
    surf = pygame.transform.scale(surface,
                                  (int(rect[2] * scale), int(rect[3] * scale)))

    game_display.blit(surf, dest=(0, 0))
    pygame.display.update()
    clock.tick(fps)

    for i, bot in enumerate(bots):
      timestep_bot = dm_env.TimeStep(
          step_type=timestep.step_type,
          reward=timestep.reward[i],
          discount=timestep.discount,
          observation=timestep.observation[i])

      actions[i], states[i] = bot.step(timestep_bot, states[i])

    timestep = env.step(actions)


if __name__ == "__main__":
  main()
