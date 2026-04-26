import random
import numpy as np
import torch


def set_seed(seed: int) -> None:
    """
    Set random seeds for reproducibility
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """
    Get the best available device (cuda/cpu)
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_returns(rewards: list, gamma: float = 0.99) -> list:
    """
    Compute discounted returns from a list of rewards
    """
    returns = []
    G = 0.0
    for reward in reversed(rewards):
        G = reward + gamma * G
        returns.insert(0, G)
    return returns
