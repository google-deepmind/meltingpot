import numpy as np
import torch
from torch.distributions import Categorical
import torch.nn as nn
import torch.optim as optim

class Actor(nn.Module):
  def __init__(self, obs_dim, action_dim):
    super().__init__()
    self.net = nn.Sequential(
        nn.Linear(obs_dim, 128),
        nn.ReLU(),
        nn.Linear(128, 128),
        nn.ReLU(),
        nn.Linear(128, action_dim),
    )

  def forward(self, x):
    return self.net(x)


class CentralizedCritic(nn.Module):
  def __init__(self, obs_dim, num_agents):
    super().__init__()
    self.net = nn.Sequential(
        nn.Linear(obs_dim * num_agents, 256),
        nn.ReLU(),
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Linear(256, 1),
    )

  def forward(self, global_obs):
    # global_obs: (batch, obs_dim * num_agents)
    return self.net(global_obs)


class MAPPO:
  def __init__(self, obs_dim, action_dim, num_agents, lr=1e-4,
                gamma=0.99, gae_lambda=0.95, epsilon=0.2,
                epochs=4, batch_size=256, value_loss_coeff=0.5,
                entr_coeff=0.05, device="cpu"):
    self.device = device
    self.gamma = gamma
    self.lamda = gae_lambda
    self.epsilon = epsilon
    self.epochs = epochs
    self.batch_size = batch_size
    self.value_loss_coeff = value_loss_coeff
    self.entr_coeff = entr_coeff
    self.num_agents = num_agents

    # one shared actor for all agents (same as your current setup)
    self.actor = Actor(obs_dim, action_dim).to(device)

    # one centralized critic
    self.critic = CentralizedCritic(obs_dim, num_agents).to(device)

    self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
    self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=5e-5)

    # self.scheduler_actor = optim.lr_scheduler.LinearLR(
    #     self.actor_optimizer, start_factor=1.0, end_factor=0.1, total_iters=300
    # )
    # self.scheduler_critic = optim.lr_scheduler.LinearLR(
    #     self.critic_optimizer, start_factor=1.0, end_factor=0.1, total_iters=300
    # )

  def select_action(self, obs_i):
    obs_tensor = torch.tensor(obs_i, dtype=torch.float32).to(self.device)
    with torch.no_grad():
      logits = self.actor(obs_tensor)
    dist = Categorical(logits=logits)
    action = dist.sample()
    log_prob = dist.log_prob(action)
    return action.item(), log_prob.detach()

  def get_value(self, all_obs):
    # all_obs: (num_agents, obs_dim) numpy array
    global_obs = torch.tensor(
        all_obs.flatten(), dtype=torch.float32
    ).unsqueeze(0).to(self.device)
    with torch.no_grad():
      value = self.critic(global_obs)
    return value.item()

  def act_deterministic(self, obs_i):
    obs_tensor = torch.tensor(obs_i, dtype=torch.float32).to(self.device)
    with torch.no_grad():
      logits = self.actor(obs_tensor)
    return torch.argmax(logits).item()

  def compute_gae(self, rewards, values, dones, last_values):
    """
    MAPPO GAE: one value estimate per timestep (from centralized critic),
    so no need to split by agent — values are already per-timestep.
    rewards: (rollout_length * num_agents,) interleaved
    values:  (rollout_length,) one per timestep
    """
    # sum rewards across agents per timestep
    n_steps = len(values)
    timestep_rewards = [
        sum(rewards[t * self.num_agents:(t + 1) * self.num_agents])
        for t in range(n_steps)
    ]

    bootstrapped_values = values + [last_values]
    gae = 0
    advantages = []

    for t in reversed(range(n_steps)):
      delta = (
          timestep_rewards[t]
          + self.gamma * bootstrapped_values[t + 1] * (1 - dones[t])
          - bootstrapped_values[t]
      )
      gae = delta + self.gamma * self.lamda * (1 - dones[t]) * gae
      advantages.insert(0, gae)

    returns = [adv + val for adv, val in zip(advantages, values)]

    # expand back to per-agent for actor update
    advantages_expanded = [adv for adv in advantages for _ in range(self.num_agents)]
    returns_expanded = [ret for ret in returns for _ in range(self.num_agents)]

    return advantages_expanded, returns_expanded

  def update(self, memory):
    obs = torch.tensor(
        np.array(memory["obs"]), dtype=torch.float32
    ).to(self.device)
    actions = torch.tensor(memory["actions"]).to(self.device)
    old_log_probs = torch.stack(memory["log_probs"]).to(self.device)
    returns = torch.tensor(
        memory["returns"], dtype=torch.float32
    ).to(self.device)
    advantages = torch.tensor(
        memory["advantages"], dtype=torch.float32
    ).to(self.device)
    # global_obs: one per timestep, shape (rollout_length, obs_dim * num_agents)
    global_obs = torch.tensor(
        np.array(memory["global_obs"]), dtype=torch.float32
    ).to(self.device)

    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    n = len(obs)
    n_steps = len(global_obs)

    policy_losses, value_losses, entropies, kls = [], [], [], []

    for _ in range(self.epochs):
        # --- actor update (per agent transitions) ---
      actor_indices = np.random.permutation(n)
      for start in range(0, n, self.batch_size):
        batch_idx = actor_indices[start:start + self.batch_size]

        logits = self.actor(obs[batch_idx])
        dist = Categorical(logits=logits)
        new_log_probs = dist.log_prob(actions[batch_idx])
        entropy = dist.entropy().mean()

        ratio = (new_log_probs - old_log_probs[batch_idx]).exp()
        surr1 = ratio * advantages[batch_idx]
        surr2 = torch.clamp(
            ratio, 1 - self.epsilon, 1 + self.epsilon
        ) * advantages[batch_idx]
        policy_loss = -torch.min(surr1, surr2).mean()
        approx_kl = (old_log_probs[batch_idx] - new_log_probs).mean()

        self.actor_optimizer.zero_grad()
        (policy_loss - self.entr_coeff * entropy).backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        self.actor_optimizer.step()

        policy_losses.append(policy_loss.item())
        entropies.append(entropy.item())
        kls.append(approx_kl.item())

      # --- critic update (per timestep) ---
      critic_indices = np.random.permutation(n_steps)
      for start in range(0, n_steps, self.batch_size):
        batch_idx = critic_indices[start:start + self.batch_size]

        values = self.critic(global_obs[batch_idx]).squeeze()
        # use per-timestep returns (not expanded)
        timestep_returns = returns[::self.num_agents][batch_idx]
        value_loss = (timestep_returns - values).pow(2).mean()

        self.critic_optimizer.zero_grad()
        (self.value_loss_coeff * value_loss).backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.critic_optimizer.step()

        value_losses.append(value_loss.item())

    # self.scheduler_actor.step()
    # self.scheduler_critic.step()

    return {
      "policy_loss": np.mean(policy_losses),
      "value_loss": np.mean(value_losses),
      "entropy": np.mean(entropies),
      "kl": np.mean(kls),
    }
