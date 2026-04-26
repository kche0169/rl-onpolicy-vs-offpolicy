from dataclasses import dataclass


@dataclass
class REINFORCEConfig:
    """
    Hyperparameters for REINFORCE with baseline
    """
    lr_pi: float = 1e-3
    lr_v: float = 1e-3
    gamma: float = 0.99
    hidden_dim: int = 64


@dataclass
class A2CConfig:
    """
    Hyperparameters for A2C
    """
    lr_pi: float = 1e-3
    lr_v: float = 1e-3
    gamma: float = 0.99
    hidden_dim: int = 64
    num_steps: int = 5


@dataclass
class PPOConfig:
    """
    Hyperparameters for PPO
    """
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    hidden_dim: int = 64
    num_steps: int = 2048
    batch_size: int = 64
    epochs: int = 10


@dataclass
class DQNConfig:
    """
    Hyperparameters for DQN
    """
    lr: float = 1e-3
    gamma: float = 0.99
    hidden_dim: int = 64
    buffer_size: int = 100000
    batch_size: int = 32
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 500
    target_update_freq: int = 1000


@dataclass
class SACConfig:
    """
    Hyperparameters for SAC
    """
    lr_pi: float = 3e-4
    lr_q: float = 3e-4
    lr_alpha: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005
    hidden_dim: int = 256
    buffer_size: int = 1000000
    batch_size: int = 256
    auto_alpha: bool = True
