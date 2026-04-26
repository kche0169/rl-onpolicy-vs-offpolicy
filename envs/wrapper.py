import gymnasium as gym
import numpy as np
from typing import Any, Tuple


class NormalizeObservation(gym.ObservationWrapper):
    """
    Normalize observations to [0, 1] or [-1, 1]
    """

    def __init__(self, env: gym.Env, normalize_to: str = "01") -> None:
        super().__init__(env)
        self.normalize_to = normalize_to

    def observation(self, observation: np.ndarray) -> np.ndarray:
        if self.normalize_to == "01":
            low = self.env.observation_space.low
            high = self.env.observation_space.high
            return (observation - low) / (high - low)
        elif self.normalize_to == "tanh":
            low = self.env.observation_space.low
            high = self.env.observation_space.high
            return 2 * (observation - low) / (high - low) - 1
        return observation
