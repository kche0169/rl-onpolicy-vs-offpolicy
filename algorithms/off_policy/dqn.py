import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))

import torch
import torch.nn as nn
import torch.optim as optim
import random

from algorithms.base.base_agent import BaseAgent
from algorithms.base.replay_buffer import ReplayBuffer
from utils.utils import get_device
from config.hyperparameters import DQNConfig


class QNetwork(nn.Module):
    """
    Q-network for DQN
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DQN(BaseAgent):
    """
    Deep Q-Network
    """

    def __init__(self, state_dim: int, action_dim: int, config: DQNConfig = None) -> None:
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config or DQNConfig()
        self.device = get_device()

        self.q_net = QNetwork(state_dim, action_dim, self.config.hidden_dim).to(self.device)
        self.target_q_net = QNetwork(state_dim, action_dim, self.config.hidden_dim).to(self.device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.config.lr)
        self.replay_buffer = ReplayBuffer(self.config.buffer_size)

        self.step = 0
        self.epsilon = self.config.epsilon_start

    def select_action(self, state: torch.Tensor) -> int:
        """
        Select an action using epsilon-greedy
        """
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        else:
            state = torch.tensor(state, dtype=torch.float32).to(self.device)
            with torch.no_grad():
                q_values = self.q_net(state)
                return q_values.argmax().item()

    def update(self) -> dict:
        """
        Update Q-network using experience replay
        """
        if len(self.replay_buffer) < self.config.batch_size:
            return {}

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.config.batch_size)

        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)

        q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q_values = self.target_q_net(next_states).max(1)[0]
            target_q_values = rewards + self.config.gamma * next_q_values * (1 - dones)

        loss = nn.MSELoss()(q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.step += 1

        # Update epsilon
        self.epsilon = max(self.config.epsilon_end, self.epsilon - (self.config.epsilon_start - self.config.epsilon_end) / self.config.epsilon_decay)

        # Update target network
        if self.step % self.config.target_update_freq == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())

        return {"q_loss": loss.item(), "epsilon": self.epsilon}

    def save(self, path: str) -> None:
        """
        Save the agent's parameters
        """
        torch.save({
            "q_net": self.q_net.state_dict(),
            "target_q_net": self.target_q_net.state_dict(),
            "optimizer": self.optimizer.state_dict()
        }, path)

    def load(self, path: str) -> None:
        """
        Load the agent's parameters
        """
        checkpoint = torch.load(path)
        self.q_net.load_state_dict(checkpoint["q_net"])
        self.target_q_net.load_state_dict(checkpoint["target_q_net"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
