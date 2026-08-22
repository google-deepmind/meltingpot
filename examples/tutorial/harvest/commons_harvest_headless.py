# NOTE:
# TensorFlow/XLA may emit initialization warnings in headless environments.
# These are harmless and can be safely ignored.

"""
Headless Harvest example for Melting Pot.

This example demonstrates how to interact with the Harvest substrate
programmatically without a graphical display.

Notes:
- Designed for headless environments (servers, CI, Codespaces)
- Rewards may remain zero for many steps with naive policies
- TensorFlow/XLA warnings may appear and can be safely ignored
"""

import os
import random

# This will suppress most of the Tensorflow logging (some early warnings are unavoidable)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from meltingpot import substrate


def main():
    roles = ["default"] * 7
    env = substrate.build("commons_harvest__open", roles=roles)

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
            "Rewards:", timestep.reward,
        )


if __name__ == "__main__":
    main()
