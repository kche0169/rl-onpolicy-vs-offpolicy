from abc import ABC, abstractmethod
from typing import Any, Tuple


class BaseAgent(ABC):
    """
    Base class for all RL agents
    """

    @abstractmethod
    def select_action(self, state: Any) -> Any:
        """
        Select an action given a state
        """
        pass

    @abstractmethod
    def update(self, *args: Any, **kwargs: Any) -> dict:
        """
        Update the agent's parameters using collected experience
        Returns dictionary of update metrics
        """
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """
        Save the agent's parameters to a file
        """
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        """
        Load the agent's parameters from a file
        """
        pass
