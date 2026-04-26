import random
from collections import deque
from typing import Tuple, Any


class ReplayBuffer:
    """
    Simple replay buffer for off-policy RL
    """

    def __init__(self, capacity: int = 100000) -> None:
        self.buffer = deque(maxlen=capacity)

    def push(self, state: Any, action: Any, reward: float, next_state: Any, done: bool) -> None:
        """
        Add a transition to the buffer
        """
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple[list, list, list, list, list]:
        """
        Sample a batch of transitions from the buffer
        """
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return list(states), list(actions), list(rewards), list(next_states), list(dones)

    def __len__(self) -> int:
        """
        Return the current size of the buffer
        """
        return len(self.buffer)
