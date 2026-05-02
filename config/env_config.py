from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass
class EnvConfig:
    name: str
    state_dim: int
    action_dim: int
    discrete: bool
    max_action: float = 1.0


def get_env_config(env_name: str) -> EnvConfig:
    """Get default config for common environments"""
    env_configs = {
        "CartPole-v1": EnvConfig("CartPole-v1", 4, 2, True),
        "LunarLander-v3": EnvConfig("LunarLander-v3", 8, 4, True),
        "Acrobot-v1": EnvConfig("Acrobot-v1", 6, 3, True),
        "Pendulum-v1": EnvConfig("Pendulum-v1", 3, 1, False, 2.0),
        "MountainCarContinuous-v0": EnvConfig("MountainCarContinuous-v0", 2, 1, False, 1.0),
    }
    if env_name in env_configs:
        return env_configs[env_name]
    else:
        raise ValueError(f"Unknown environment: {env_name}")