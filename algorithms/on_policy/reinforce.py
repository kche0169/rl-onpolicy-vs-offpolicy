import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from algorithms.base.base_agent import BaseAgent
from utils.utils import get_device, compute_returns
from config.hyperparameters import REINFORCEConfig


class PolicyNet(nn.Module):
    """
    Simple policy network for discrete actions
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.net(x), dim=-1)


class ValueNet(nn.Module):
    """
    Simple value network for baseline
    """

    def __init__(self, state_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class REINFORCE(BaseAgent):
    """
    REINFORCE with baseline
    """

    def __init__(self, state_dim: int, action_dim: int, config: REINFORCEConfig = None) -> None:
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config or REINFORCEConfig()
        self.device = get_device()

        self.policy = PolicyNet(state_dim, action_dim, self.config.hidden_dim).to(self.device)
        self.value = ValueNet(state_dim, self.config.hidden_dim).to(self.device)

        self.optimizer_pi = optim.Adam(self.policy.parameters(), lr=self.config.lr_pi)
        self.optimizer_v = optim.Adam(self.value.parameters(), lr=self.config.lr_v)

    def select_action(self, state: torch.Tensor) -> tuple[int, torch.Tensor]:
        """
        Select an action using the current policy
        """
        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        probs = self.policy(state)
        dist = Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob

    def update(self, states: list, actions: list, log_probs: list, rewards: list, dones: list) -> dict:
        """
        Update policy and value function using collected episode
        """
        returns = compute_returns(rewards, self.config.gamma)

        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).to(self.device)
        log_probs = torch.stack(log_probs).to(self.device)
        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)

        values = self.value(states)
        advantages = returns - values.detach()

        policy_loss = -(log_probs * advantages).mean()
        self.optimizer_pi.zero_grad()
        policy_loss.backward()
        self.optimizer_pi.step()

        value_loss = nn.MSELoss()(values, returns)
        self.optimizer_v.zero_grad()
        value_loss.backward()
        self.optimizer_v.step()

        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "mean_return": returns.mean().item()
        }

    def save(self, path: str) -> None:
        """
        Save the agent's parameters
        """
        torch.save({
            "policy": self.policy.state_dict(),
            "value": self.value.state_dict(),
            "optimizer_pi": self.optimizer_pi.state_dict(),
            "optimizer_v": self.optimizer_v.state_dict()
        }, path)

    def load(self, path: str) -> None:
        """
        Load the agent's parameters
        """
        checkpoint = torch.load(path)
        self.policy.load_state_dict(checkpoint["policy"])
        self.value.load_state_dict(checkpoint["value"])
        self.optimizer_pi.load_state_dict(checkpoint["optimizer_pi"])
        self.optimizer_v.load_state_dict(checkpoint["optimizer_v"])
