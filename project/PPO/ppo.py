import numpy as np
import torch
from torch.distributions import Categorical
import torch.nn as nn
import torch.optim as optim


class ActorCritic(nn.Module):
  def __init__(self, obs_dim, action_dim):
    super().__init__()

    self.shared = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )

    self.actor = nn.Linear(128, action_dim)
    self.critic = nn.Linear(128, 1)

  def forward(self, x):
    x = self.shared(x)
    logits = self.actor(x)
    value = self.critic(x)
    return logits, value


class PPO:
  def __init__(self, obs_dim, action_dim, lr=1e-4,
               gamma=0.99, gae_lambda=0.95,
               epsilon=0.2, epochs=4,
               batch_size=256, value_loss_coeff=0.5,
               entr_coeff=0.05, device="cpu"):
    self.device = device
    self.gamma = gamma
    self.lamda = gae_lambda
    self.epsilon = epsilon
    self.epochs = epochs
    self.batch_size = batch_size
    self.value_loss_coeff = value_loss_coeff
    self.entr_coeff = entr_coeff
    self.model = ActorCritic(obs_dim, action_dim).to(device)
    self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
    self.scheduler = optim.lr_scheduler.LinearLR(
        self.optimizer,
        start_factor=1.0,
        end_factor=0.1,
        total_iters=300  # number of update calls
    )

  def select_action(self, obs):
    obs = torch.tensor(obs, dtype=torch.float32).to(self.device)

    logits, value = self.model(obs)
    dist = Categorical(logits=logits)
    action = dist.sample()
    log_prob = dist.log_prob(action)

    return (action.item(), log_prob.detach(), value.detach())

  def act_deterministic(self, obs):
    obs = torch.tensor(obs, dtype=torch.float32).to(self.device)

    with torch.no_grad():
      logits, _ = self.model(obs)

    return torch.argmax(logits).item()

  # In PPO class
  def compute_gae(self, rewards, values, dones, last_values, num_agents):
    """
    rewards, values, dones: flat lists of length (rollout_length * num_agents),
                            interleaved by agent index at each timestep.
    last_values: list of length num_agents — value estimates for the final obs.
    """
    all_advantages = [None] * len(rewards)
    all_returns    = [None] * len(rewards)

    for agent_i in range(num_agents):
        # Extract this agent's slice (every num_agents-th element)
        agent_rewards = rewards[agent_i::num_agents]
        agent_values  = values[agent_i::num_agents]
        agent_dones   = dones[agent_i::num_agents]

        # Bootstrap from last obs instead of hardcoding 0
        bootstrapped_values = agent_values + [last_values[agent_i]]

        gae = 0
        agent_advantages = []

        for t in reversed(range(len(agent_rewards))):
            delta = (
                agent_rewards[t]
                + self.gamma * bootstrapped_values[t + 1] * (1 - agent_dones[t])
                - bootstrapped_values[t]
            )
            gae = delta + self.gamma * self.lamda * (1 - agent_dones[t]) * gae
            agent_advantages.insert(0, gae)

        agent_returns = [adv + val for adv, val in zip(agent_advantages, agent_values)]

        # Write back into the interleaved positions
        for idx, t in enumerate(range(agent_i, len(rewards), num_agents)):
            all_advantages[t] = agent_advantages[idx]
            all_returns[t]    = agent_returns[idx]

    return all_advantages, all_returns

  def update(self, memory):
    obs = torch.tensor(np.array(memory["obs"]), dtype=torch.float32).to(self.device)
    actions = torch.tensor(memory["actions"]).to(self.device)
    old_log_probs = torch.stack(memory["log_probs"]).to(self.device)
    returns = torch.tensor(memory["returns"], dtype=torch.float32).to(self.device)
    advantages = torch.tensor(memory["advantages"], dtype=torch.float32).to(self.device)

    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    n = len(obs)

    for _ in range(self.epochs):
      indices = np.arange(n)
      np.random.shuffle(indices)

      for start in range(0, n, self.batch_size):
        end = start + self.batch_size
        batch_idx = indices[start:end]

        logits, values = self.model(obs[batch_idx])
        dist = Categorical(logits=logits)

        new_log_probs = dist.log_prob(actions[batch_idx])
        entropy = dist.entropy().mean()

        ratio = (new_log_probs - old_log_probs[batch_idx]).exp()

        surr1 = ratio * advantages[batch_idx]
        surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages[batch_idx]

        policy_loss = -torch.min(surr1, surr2).mean()

        value_loss = (returns[batch_idx] - values.squeeze()).pow(2).mean()

        loss = (
            policy_loss
            + self.value_loss_coeff * value_loss
            - self.entr_coeff * entropy
        )

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
        self.optimizer.step()
        self.scheduler.step()
