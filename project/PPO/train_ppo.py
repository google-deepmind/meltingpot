#from meltingpot_wrapper import MeltingPotWrapper
import random

from meltingpot import substrate
import numpy as np
from ppo import PPO
import torch


def process_obs(obs_i):
  if isinstance(obs_i, dict):
    rgb = obs_i["RGB"][::4, ::4, :]
    gray = rgb.mean(axis=-1, keepdims=True)  # collapse color channels
    rgb = gray.flatten()
    other = np.array([
        obs_i.get("READY_TO_SHOOT", 0),
        obs_i.get("NUM_OTHERS_WHO_CLEANED_THIS_STEP", 0),
    ])
    return np.concatenate([rgb, other])
  else:
    return np.array(obs_i).flatten()

def train_ppo(
    substrate_name="pure_coordination_in_the_matrix__repeated",
    total_steps=1000000,
    rollout_length=10000,
    seed=0,
):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)

  # Build environment
  env_config = substrate.get_config(substrate_name)
  roles = env_config.default_player_roles
  env = substrate.build(substrate_name, roles=roles)
  num_agents = len(env.observation_spec())

  # Initialize
  timestep = env.reset()
  obs = timestep.observation

  # Infer dimensions
  sample_obs = process_obs(obs[0])
  obs_dim = sample_obs.shape[0]

  action_spec = env.action_spec()[0]
  action_dim = action_spec.num_values  # discrete

  print(f"Obs dim: {obs_dim}, Action dim: {action_dim}, Agents: {num_agents}")

  # Create PPO
  ppo = PPO(obs_dim, action_dim)

  # Memory buffer
  memory = {
      "obs": [],
      "actions": [],
      "log_probs": [],
      "rewards": [],
      "dones": [],
      "values": [],
  }

  episode_reward = 0
  step_count = 0
  train_rewards = []

  while step_count < total_steps:

    for _ in range(rollout_length):

      actions = []
      obs_processed = []

      for i in range(num_agents):
        obs_i = process_obs(obs[i])
        obs_processed.append(obs_i)

        action, log_prob, value = ppo.select_action(obs_i)

        actions.append(action)

        memory["obs"].append(obs_i)
        memory["actions"].append(action)
        memory["log_probs"].append(log_prob)
        memory["values"].append(value.item())

      timestep = env.step(actions)
      next_obs = timestep.observation
      rewards = timestep.reward
      done = timestep.last()

      for i in range(num_agents):
        memory["rewards"].append(rewards[i])
        memory["dones"].append(done)
        episode_reward += rewards[i]

      obs = next_obs
      step_count += 1

      if done:
        train_rewards.append(episode_reward)
        episode_reward = 0

        timestep = env.reset()
        obs = timestep.observation

    # Get last observation values for bootstrapping
    last_values = []
    for i in range(num_agents):
        obs_i = process_obs(obs[i])
        obs_tensor = torch.tensor(obs_i, dtype=torch.float32).to(ppo.device)
        with torch.no_grad():
            _, v = ppo.model(obs_tensor)
        last_values.append(v.item())

    advantages, returns = ppo.compute_gae(
        memory["rewards"],
        memory["values"],
        memory["dones"],
        last_values,
        num_agents,
    )

    memory["advantages"] = advantages
    memory["returns"] = returns

    ppo.update(memory)

    #print(f"Steps: {step_count}, Episode Reward: {episode_reward:.2f}")
    print(f"Steps: {step_count}, Avg Train Reward: {np.mean(train_rewards[-10:]) if train_rewards else 0:.2f}")

    # Reset memory
    for key in memory:
      memory[key] = []

  env.close()
  return ppo, train_rewards

def test(
    substrate_name="pure_coordination_in_the_matrix__repeated",
    model=None,
    num_episodes=3,
    max_steps=1000000,
):

  env_config = substrate.get_config(substrate_name)
  roles = env_config.default_player_roles
  env = substrate.build(substrate_name, roles=roles)
  num_agents = len(env.observation_spec())
  test_rewards = []

  for ep in range(num_episodes):
    timestep = env.reset()
    obs = timestep.observation

    total_reward = 0

    for step in range(max_steps):

      actions = []

      for i in range(num_agents):
        obs_i = process_obs(obs[i])

        # only actor is used
        action = model.act_deterministic(obs_i)
        actions.append(action)

      timestep = env.step(actions)
      obs = timestep.observation
      rewards = timestep.reward
      done = timestep.last()

      total_reward += sum(rewards)

      if done:
        break

    test_rewards.append(total_reward)
    print(f"Episode {ep+1}: Total Reward = {total_reward:.2f}")

  env.close()
  avg_reward = np.mean(test_rewards)
  return avg_reward

if __name__ == "__main__":
  seeds = [0]

  substrate_name = "commons_harvest__open"

  all_results = []
  all_test_rewards = []
  all_train_rewards = []

  for seed in seeds:
    print(f"\n=== Training with seed {seed} ===")

    model, train_reward = train_ppo(substrate_name, seed=seed)

    model.model.eval()

    test_reward = test(
        substrate_name=substrate_name,
        model=model,
        num_episodes=5
    )

    all_test_rewards.append(test_reward)
    all_train_rewards.append(train_reward)

  print("\nFinal Test Results:")
  print(f"Mean: {np.mean(all_test_rewards):.2f}")
  print(f"Std: {np.std(all_test_rewards):.2f}")
