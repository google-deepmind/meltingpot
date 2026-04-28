# from meltingpot_wrapper import MeltingPotWrapper
import os
import random

from meltingpot import substrate
import numpy as np
import pandas as pd
from ppo import PPO
import torch


def process_obs(obs_i):
  if isinstance(obs_i, dict):
    rgb = obs_i["RGB"][::4, ::4, :].flatten()
    extras = []
    if "READY_TO_SHOOT" in obs_i:
      extras.append(obs_i["READY_TO_SHOOT"])
    if "NUM_OTHERS_WHO_CLEANED_THIS_STEP" in obs_i:
      extras.append(obs_i["NUM_OTHERS_WHO_CLEANED_THIS_STEP"])
    if extras:
      return np.concatenate([rgb, np.array(extras)])
    return rgb
  else:
    return np.array(obs_i).flatten()


def train_ppo(
    substrate_name="stag_hunt_in_the_matrix__repeated",
    total_steps=1000000, # 1M for commons and 5M for the others
    rollout_length=1000,
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

  # Create PPOs for each agent
  ppos = [PPO(obs_dim, action_dim) for _ in range(num_agents)]

  episode_reward = 0
  step_count = 0
  episode_count = 0
  best_reward = -np.inf
  completed_rewards = []
  logs = []

  # Memory buffer
  memories = [{
      "obs": [],
      "actions": [],
      "log_probs": [],
      "rewards": [],
      "dones": [],
      "values": [],
  } for _ in range(num_agents)]

  while step_count < total_steps:

    for _ in range(rollout_length):

      actions = []

      for i in range(num_agents):
        obs_i = process_obs(obs[i])

        action, log_prob, value = ppos[i].select_action(obs_i)
        actions.append(action)

        memories[i]["obs"].append(obs_i)
        memories[i]["actions"].append(action)
        memories[i]["log_probs"].append(log_prob)
        memories[i]["values"].append(value.item())

      timestep = env.step(actions)
      next_obs = timestep.observation
      rewards = timestep.reward
      done = timestep.last()

      for i in range(num_agents):
        memories[i]["rewards"].append(rewards[i])
        memories[i]["dones"].append(done)
        episode_reward += rewards[i]

      obs = next_obs

      step_count += 1

      if done:
        completed_rewards.append(episode_reward)
        episode_count += 1
        episode_reward = 0
        timestep = env.reset()
        obs = timestep.observation

    for i in range(num_agents):
      obs_i = process_obs(obs[i])
      obs_tensor = torch.tensor(obs_i, dtype=torch.float32).to(ppos[i].device)
      with torch.no_grad():
        _, v = ppos[i].model(obs_tensor)

      advantages, returns = ppos[i].compute_gae_single(
          memories[i]["rewards"],
          memories[i]["values"],
          memories[i]["dones"],
          v.item(),
      )

      memories[i]["advantages"] = advantages
      memories[i]["returns"] = returns

    all_stats = []
    for i in range(num_agents):
      stats = ppos[i].update(memories[i])
      all_stats.append(stats)

    avg_reward = np.mean(completed_rewards[-10:]) if completed_rewards else 0.0

    if avg_reward > best_reward:
      best_reward = avg_reward
      for i in range(num_agents):
        torch.save(
            ppos[i].model.state_dict(), f"logs/ippo/best_model_{substrate_name}_seed{seed}_agent{i}.pt"
        )


    stats_averaged = { metric: np.mean([stat[metric] for stat in all_stats])
                  for metric in all_stats[0] }

    logs.append({
        "step": step_count,
        "avg_reward": avg_reward,
        "best_reward": best_reward,
        "episodes_completed": episode_count,
        "policy_loss": stats_averaged["policy_loss"],
        "value_loss": stats_averaged["value_loss"],
        "entropy": stats_averaged["entropy"],
        "kl": stats_averaged["kl"],
    })

    print(
        f"Step {step_count} | "
        f"R: {avg_reward:.2f} | "
        f"Best: {best_reward:.2f} | "
        f"Ep: {episode_count} | "
        f"PL: {stats_averaged['policy_loss']:.3f} | "
        f"VL: {stats_averaged['value_loss']:.3f} | "
        f"H: {stats_averaged['entropy']:.3f} | "
        f"KL: {stats_averaged['kl']:.5f}"
    )

    # Reset memory
    for i in range(num_agents):
      for key in memories[i]:
        memories[i][key] = []

  env.close()
  return ppos, logs


def test(
    substrate_name="stag_hunt_in_the_matrix__repeated",
    model=None,
    num_episodes=5,
    max_steps=5000,
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

    for _ in range(max_steps):

      actions = []

      for i in range(num_agents):
        obs_i = process_obs(obs[i])

        # only actor is used
        action = model[i].act_deterministic(obs_i)
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
  std_reward = np.std(test_rewards)
  print(f"Avg Test Reward: {avg_reward:.2f} ± {std_reward:.2f}")
  return test_rewards


def save_logs(logs, test_rewards, save_dir="logs/ippo", seed=0, substrate="env"):
  os.makedirs(save_dir, exist_ok=True)
  base = f"{save_dir}/{substrate}_seed{seed}"

  df = pd.DataFrame(logs)

  for i, r in enumerate(test_rewards):
    df[f"test_reward_{i}"] = np.nan
    df.loc[df.index[-1], f"test_reward_{i}"] = r

  df["test_avg_reward"] = np.nan
  df["test_std_reward"] = np.nan
  df.loc[df.index[-1], "test_avg_reward"] = np.mean(test_rewards)
  df.loc[df.index[-1], "test_std_reward"] = np.std(test_rewards)

  df.to_csv(f"{base}_logs.csv", index=False)
  print(f"Logs saved to {base}_logs.csv")


if __name__ == "__main__":
  seeds = [0, 1, 2]

  substrate_name = "commons_harvest__open"
  # substrate_name = "stag_hunt_in_the_matrix__repeated" # test episode length of 2500
  # substrate_name = "chicken_in_the_matrix__repeated"

  all_test_rewards = []

  for seed in seeds:
    print(f"\n=== Training with seed {seed} ===")

    models, logs = train_ppo(substrate_name, seed=seed)
    for i in range(len(models)):
      models[i].model.load_state_dict(
          torch.load(f"logs/ippo/best_model_{substrate_name}_seed{seed}_agent{i}.pt")
      )
      models[i].model.eval()

    test_reward = test(
        substrate_name=substrate_name,
        model=models,
        num_episodes=5
    )

    all_test_rewards.append(test_reward)
    save_logs(logs, test_reward, seed=seed, substrate=substrate_name)

  print("\nFinal Test Results:")
  print(f"Mean: {np.mean(all_test_rewards):.2f}")
  print(f"Std: {np.std(all_test_rewards):.2f}")
