# Copyright 2026 DeepMind Technologies Limited.
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
"""Headless Harvest example for Melting Pot.

This example demonstrates how to interact with the Harvest substrate
programmatically without a graphical display.

Notes:
- Designed for headless environments (servers, CI, Codespaces)
- Rewards may remain zero for many steps with naive policies
- TensorFlow/XLA warnings may appear and can be safely ignored
"""

import os
import random

# This will suppress most of the Tensorflow logging (some early warnings are
# unavoidable).
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from meltingpot import substrate  # pylint: disable=g-import-not-at-top


def main():
  substrate_name = "commons_harvest__open"
  config = substrate.get_config(substrate_name)
  env = substrate.build(substrate_name, roles=config.default_player_roles)

  timestep = env.reset()

  num_actions = env.action_spec()[0].num_values

  print("Number of agents:", len(timestep.observation))
  print("Number of actions per agent:", num_actions)
  print("Observation keys:", timestep.observation[0].keys())

  for step in range(20):
    actions = [
        random.randint(0, num_actions - 1)
        for _ in range(len(timestep.observation))
    ]

    timestep = env.step(actions)

    print(
        f"Step {step + 1}",
        "Rewards:",
        timestep.reward,
    )


if __name__ == "__main__":
  main()
