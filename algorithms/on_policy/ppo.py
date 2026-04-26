import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from algorithms.base.base_agent import BaseAgent
from utils.utils import get_device
from config.hyperparameters import PPOConfig


class PPONet(nn.Module):
    """
    Network for PPO
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )
        self.actor = nn.Linear(hidden_dim, action_dim)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        shared = self.shared(x)
        logits = self.actor(shared)
        value = self.critic(shared).squeeze(-1)
        return logits, value


class PPO(BaseAgent):
    """
    Proximal Policy Optimization
    """

    def __init__(self, state_dim: int, action_dim: int, config: PPOConfig = None) -> None:
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config or PPOConfig()
        self.device = get_device()

        self.net = PPONet(state_dim, action_dim, self.config.hidden_dim).to(self.device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.config.lr)

    def select_action(self, state: torch.Tensor) -> tuple[int, torch.Tensor, torch.Tensor]:
        """
        Select an action using the current policy
        """
        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        logits, value = self.net(state)
        probs = torch.softmax(logits, dim=-1)
        dist = Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob.detach(), value.detach()

    def compute_gae(self, rewards: torch.Tensor, values: torch.Tensor, next_values: torch.Tensor, dones: torch.Tensor) -> torch.Tensor:
        """
        Compute Generalized Advantage Estimation
        """
        advantages = []
        advantage = 0.0

        for reward, value, next_value, done in zip(reversed(rewards), reversed(values), reversed(next_values), reversed(dones)):
            delta = reward + self.config.gamma * next_value * (1 - done) - value
            advantage = delta + self.config.gamma * self.config.gae_lambda * (1 - done) * advantage
            advantages.insert(0, advantage)

        return torch.tensor(advantages, dtype=torch.float32).to(self.device)

    def update(self, states: list, actions: list, old_log_probs: list, rewards: list, next_states: list, dones: list) -> dict:
        """
        Update policy and value function using PPO
        """
        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).to(self.device)
        old_log_probs = torch.stack(old_log_probs).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            _, values = self.net(states)
            _, next_values = self.net(torch.tensor(next_states, dtype=torch.float32).to(self.device))
            advantages = self.compute_gae(rewards, values, next_values, dones)
            returns = advantages + values

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_policy_loss = 0.0
        total_value_loss = 0.0

        # PPO epochs
        for _ in range(self.config.epochs):
            # Shuffle and batch
            indices = torch.randperm(len(states)).to(self.device)
            for start in range(0, len(states), self.config.batch_size):
                end = start + self.config.batch_size
                batch_indices = indices[start:end]

                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]

                logits, values_pred = self.net(batch_states)
                probs = torch.softmax(logits, dim=-1)
                dist = Categorical(probs)
                log_probs = dist.log_prob(batch_actions)

                # Policy loss with clipping
                ratio = torch.exp(log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.config.clip_eps, 1 + self.config.clip_eps) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = nn.MSELoss()(values_pred, batch_returns)

                total_loss = policy_loss + 0.5 * value_loss

                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()

        return {
            "policy_loss": total_policy_loss / self.config.epochs,
            "value_loss": total_value_loss / self.config.epochs
        }

    def save(self, path: str) -> None:
        """
        Save the agent's parameters
        """
        torch.save({
            "net": self.net.state_dict(),
            "optimizer": self.optimizer.state_dict()
        }, path)

    def load(self, path: str) -> None:
        """
        Load the agent's parameters
        """
        checkpoint = torch.load(path)
        self.net.load_state_dict(checkpoint["net"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
