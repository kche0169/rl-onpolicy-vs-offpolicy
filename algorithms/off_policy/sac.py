import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np

from algorithms.base.base_agent import BaseAgent
from algorithms.base.replay_buffer import ReplayBuffer
from utils.utils import get_device
from config.hyperparameters import SACConfig


LOG_STD_MAX = 2
LOG_STD_MIN = -5


class Actor(nn.Module):
    """
    Actor network for SAC (stochastic policy)
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256, action_scale: float = 1.0, action_bias: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.mu_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std_layer = nn.Linear(hidden_dim, action_dim)
        self.action_scale = torch.tensor(action_scale, dtype=torch.float32)
        self.action_bias = torch.tensor(action_bias, dtype=torch.float32)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.net(x)
        mu = self.mu_layer(x)
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        return mu, log_std

    def get_action(self, x: torch.Tensor, deterministic: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, log_std = self.forward(x)
        std = log_std.exp()

        if deterministic:
            x_t = torch.tanh(mu)
        else:
            dist = Normal(mu, std)
            x_t = dist.rsample()

        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias

        if not deterministic:
            log_prob = dist.log_prob(x_t)
            log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
            log_prob = log_prob.sum(1, keepdim=False)
        else:
            log_prob = torch.tensor(0.0, dtype=torch.float32).to(x.device)

        return action, log_prob, mu


class Critic(nn.Module):
    """
    Twin Q-networks for SAC
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x: torch.Tensor, a: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        sa = torch.cat([x, a], 1)
        q1 = self.q1(sa)
        q2 = self.q2(sa)
        return q1, q2


class SAC(BaseAgent):
    """
    Soft Actor-Critic
    """

    def __init__(self, state_dim: int, action_dim: int, max_action: float = 1.0, config: SACConfig = None) -> None:
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config or SACConfig()
        self.device = get_device()

        self.actor = Actor(state_dim, action_dim, self.config.hidden_dim, max_action).to(self.device)
        self.critic = Critic(state_dim, action_dim, self.config.hidden_dim).to(self.device)
        self.target_critic = Critic(state_dim, action_dim, self.config.hidden_dim).to(self.device)
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.config.lr_pi)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.config.lr_q)

        if self.config.auto_alpha:
            self.target_entropy = -torch.prod(torch.tensor([action_dim], dtype=torch.float32)).item()
            self.log_alpha = torch.tensor(0.0, dtype=torch.float32, requires_grad=True, device=self.device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.config.lr_alpha)
            self.alpha = self.log_alpha.exp().item()
        else:
            self.alpha = 0.2
            self.log_alpha = None
            self.alpha_optimizer = None

        self.replay_buffer = ReplayBuffer(self.config.buffer_size)

    def select_action(self, state: torch.Tensor, deterministic: bool = False) -> np.ndarray:
        """
        Select an action using the current policy
        """
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        action, _, _ = self.actor.get_action(state, deterministic)
        return action.cpu().detach().numpy()[0]

    def update(self) -> dict:
        """
        Update policy, critic, and alpha
        """
        if len(self.replay_buffer) < self.config.batch_size:
            return {}

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.config.batch_size)

        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.float32).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device).unsqueeze(1)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device).unsqueeze(1)

        with torch.no_grad():
            next_actions, next_log_probs, _ = self.actor.get_action(next_states)
            q1_targ, q2_targ = self.target_critic(next_states, next_actions)
            q_targ = torch.min(q1_targ, q2_targ) - self.alpha * next_log_probs.unsqueeze(1)
            backup = rewards + (1 - dones) * self.config.gamma * q_targ

        q1, q2 = self.critic(states, actions)
        q_loss = nn.MSELoss()(q1, backup) + nn.MSELoss()(q2, backup)

        self.critic_optimizer.zero_grad()
        q_loss.backward()
        self.critic_optimizer.step()

        pi, log_pi, _ = self.actor.get_action(states)
        q1_pi, q2_pi = self.critic(states, pi)
        q_pi = torch.min(q1_pi, q2_pi)
        actor_loss = (self.alpha * log_pi - q_pi).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        alpha_loss = 0.0
        if self.config.auto_alpha:
            with torch.no_grad():
                _, log_pi, _ = self.actor.get_action(states)
            alpha_loss = (-self.log_alpha * (log_pi + self.target_entropy)).mean()

            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

            self.alpha = self.log_alpha.exp().item()

        # Update target networks
        for param, target_param in zip(self.critic.parameters(), self.target_critic.parameters()):
            target_param.data.copy_(self.config.tau * param.data + (1 - self.config.tau) * target_param.data)

        return {
            "q_loss": q_loss.item(),
            "actor_loss": actor_loss.item(),
            "alpha_loss": alpha_loss.item() if isinstance(alpha_loss, torch.Tensor) else alpha_loss,
            "alpha": self.alpha
        }

    def save(self, path: str) -> None:
        """
        Save the agent's parameters
        """
        state_dict = {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "target_critic": self.target_critic.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "alpha": self.alpha
        }
        if self.config.auto_alpha:
            state_dict["log_alpha"] = self.log_alpha
            state_dict["alpha_optimizer"] = self.alpha_optimizer.state_dict()
        torch.save(state_dict, path)

    def load(self, path: str) -> None:
        """
        Load the agent's parameters
        """
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.target_critic.load_state_dict(checkpoint["target_critic"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        self.alpha = checkpoint["alpha"]
        if self.config.auto_alpha:
            self.log_alpha = checkpoint["log_alpha"]
            self.alpha_optimizer.load_state_dict(checkpoint["alpha_optimizer"])
