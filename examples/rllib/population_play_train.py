import os
import random

import ray
from ray import air
from ray import tune
from ray.rllib.algorithms import ppo
from ray.rllib.policy.policy import PolicySpec

from examples.rllib import utils
from meltingpot.python import substrate


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
    config = ppo.PPOConfig()
    config.num_workers = num_rollout_workers
    config.rollout_fragment_length = rollout_fragment_length
    config.train_batch_size = train_batch_size
    config.sgd_minibatch_size = sgd_minibatch_size
    config.preprocessor_pref = None
    config = config.framework("tf")
    config.num_gpus = int(os.environ.get("RLLIB_NUM_GPUS", "0"))
    config.log_level = "DEBUG"

    player_roles = substrate.get_config(substrate_name).default_player_roles
    config.env_config = {"substrate": substrate_name, "roles": player_roles}

    config.env = "meltingpot"

    test_env = utils.env_creator(config.env_config)

    policies = {}
    player_to_agent = {}
    agent2_weights = {}  # Store weights for each seed's Agent 2

    for i in range(len(player_roles)):
        rgb_shape = test_env.observation_space[f"player_{i}"]["RGB"].shape
        sprite_x = rgb_shape[0] // 8
        sprite_y = rgb_shape[1] // 8

        if i == 0:
            agent_index = i  # Agent 1 (ego agent)
        else:
            seed = i  # Seed for Agent 2
            if seed in agent2_weights:
                agent_index = agent2_weights[seed]
            else:
                available_seeds = [s for s in agent2_weights if s != seed]
                if available_seeds:
                    swap_seed = random.choice(available_seeds)
                    agent_index = agent2_weights[swap_seed]
                else:
                    agent_index = len(agent2_weights) + 1
                    agent2_weights[seed] = agent_index

        policies[f"agent_{i}"] = PolicySpec(
            policy_class=None,
            observation_space=test_env.observation_space[f"player_{i}"],
            action_space=test_env.action_space[f"player_{i}"],
            config={
                "model": {
                    "conv_filters": [
                        [16, [8, 8], 8],
                        [128, [sprite_x, sprite_y], 1]
                    ],
                },
            }
        )
        player_to_agent[f"player_{i}"] = f"agent_{agent_index}"

    def policy_mapping_fn(agent_id, **kwargs):
        del kwargs
        return player_to_agent[agent_id]

    config.multi_agent(policies=policies, policy_mapping_fn=policy_mapping_fn)

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

    ray.init()

    stop = {
        "training_iteration": 1,
    }

    results = tune.Tuner(
        "PPO",
        param_space=config.to_dict(),
        run_config=air.RunConfig(stop=stop, checkpoint_config=air.CheckpointConfig(checkpoint_frequency=20), verbose=1),
    )
    results.fit()


if __name__ == "__main__":
    main()
