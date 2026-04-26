import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from algorithms.base.base_agent import BaseAgent
from utils.utils import get_device
from config.hyperparameters import A2CConfig


class ActorCriticNet(nn.Module):
    """
    Combined actor-critic network
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


class A2C(BaseAgent):
    """
    Advantage Actor-Critic
    """

    def __init__(self, state_dim: int, action_dim: int, config: A2CConfig = None) -> None:
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config or A2CConfig()
        self.device = get_device()

        self.net = ActorCriticNet(state_dim, action_dim, self.config.hidden_dim).to(self.device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.config.lr_pi)

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
        return action.item(), log_prob, value

    def update(self, states: list, actions: list, log_probs: list, values: list, rewards: list, next_states: list, dones: list) -> dict:
        """
        Update policy and value function using collected transitions
        """
        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).to(self.device)
        log_probs = torch.stack(log_probs).to(self.device)
        values = torch.stack(values).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            _, next_values = self.net(torch.tensor(next_states, dtype=torch.float32).to(self.device))
            td_target = rewards + self.config.gamma * next_values * (1 - dones)
            advantages = td_target - values

        policy_loss = -(log_probs * advantages.detach()).mean()
        value_loss = nn.MSELoss()(values, td_target.detach())
        total_loss = policy_loss + 0.5 * value_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "total_loss": total_loss.item()
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
