import os
import random
import shutil
import ray
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.models import ModelCatalog
from ray.rllib.policy.policy import PolicySpec
# from ray.tune.schedulers import PopulationBasedTraining

from examples.rllib import utils
from meltingpot.python import substrate


def get_config(
    substrate_name: str = "stag_hunt_in_the_matrix__repeated",
    fcnet_hiddens=(64, 64),
    post_fcnet_hiddens=(256,),
    lstm_cell_size: int = 256,
    sgd_minibatch_size: int = 128,
    seed: int = 0,
):
    """Get the configuration for running an agent on a substrate using RLLib."""
    # Set random seed
    random.seed(seed)

    # Create a default configuration dictionary
    config = {
        "framework": "tf",
        "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
        "log_level": "DEBUG",
    }

    # Set environment config
    player_roles = substrate.get_config(substrate_name).default_player_roles
    config["env_config"] = {"substrate": substrate_name, "roles": player_roles}
    config["env"] = "meltingpot"

    # Extract space dimensions
    test_env = utils.env_creator(config["env_config"])

    # Setup PPO with policies, one per entry in default player roles.
    policies = {}
    player_to_agent = {}
    for i in range(len(player_roles)):
        rgb_shape = test_env.observation_space[f"player_{i}"]["RGB"].shape
        sprite_x = rgb_shape[0] // 8
        sprite_y = rgb_shape[1] // 8

        policies[f"agent_{i}"] = PolicySpec(
            policy_class=None,  # use default policy
            observation_space=test_env.observation_space[f"player_{i}"],
            action_space=test_env.action_space[f"player_{i}"],
            config={
                "model": {
                    "conv_filters": [[16, [8, 8], 8], [128, [sprite_x, sprite_y], 1]],
                },
            },
        )
        player_to_agent[f"player_{i}"] = f"agent_{i}"

    def policy_mapping_fn(agent_id, **kwargs):
        del kwargs
        return player_to_agent[agent_id]

    # Configuration for multi-agent setup with one policy per role
    config["multiagent"] = {
        "policies": policies,
        "policy_mapping_fn": policy_mapping_fn,
    }

    # Set the agent architecture
    config["model"] = {
        "fcnet_hiddens": fcnet_hiddens,
        "fcnet_activation": "relu",
        "conv_activation": "relu",
        "post_fcnet_hiddens": post_fcnet_hiddens,
        "post_fcnet_activation": "relu",
        "use_lstm": True,
        "lstm_use_prev_action": True,
        "lstm_use_prev_reward": False,
        "lstm_cell_size": lstm_cell_size,
    }

    return config


def main(num_seeds: int = 3):
    seeds = [random.randint(1, 1000) for _ in range(num_seeds)]
    checkpoint_dirs = []

    for seed in seeds:
        config = get_config(seed=seed)
        tune.register_env("meltingpot", utils.env_creator)

        # Initialize ray, train, and save
        ray.init()

        checkpoint_dir = f"checkpoints/seed_{seed}"
        checkpoint_dirs.append(checkpoint_dir)
        os.makedirs(checkpoint_dir, exist_ok=True)

        stop = {"training_iteration": 5000000}

        results = tune.run(
            PPOTrainer,
            config=config,
            stop=stop,
            checkpoint_at_end=True,
            verbose=1,
        )

        checkpoint_path = results.get_last_checkpoint()
        print(checkpoint_path)
        if checkpoint_path:
            shutil.copy(checkpoint_path, os.path.join(checkpoint_dir, "checkpoint"))

        ray.shutdown()

    # Randomly pick the latest checkpoint from any seed for population play
    population_checkpoint_dir = random.choice(checkpoint_dirs)

    # Load the best policy from the latest checkpoint
    best_policy_weights = get_best_policy_weights(population_checkpoint_dir)

    # Print the best policy weights
    print("Best Policy Weights:")
    print(best_policy_weights)


def get_best_policy_weights(checkpoint_dir: str):
    """Load the best policy weights from a checkpoint directory."""
    checkpoint_path = os.path.join(checkpoint_dir, "checkpoint")
    config = get_config()
    trainer = PPOTrainer(config=config)
    trainer.restore(checkpoint_path)
    best_policy = trainer.get_policy()
    best_policy_weights = best_policy.get_weights()
    return best_policy_weights


if __name__ == "__main__":
    main(num_seeds=3)
