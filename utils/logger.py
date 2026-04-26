import json
from typing import Any, Dict


class SimpleLogger:
    """
    Simple logger for saving training metrics
    """

    def __init__(self, log_dir: str = "logs") -> None:
        self.log_dir = log_dir
        self.metrics = []

    def log(self, step: int, metrics: Dict[str, Any]) -> None:
        """
        Log metrics at a specific step
        """
        self.metrics.append({"step": step, **metrics})

    def save(self, path: str = "metrics.json") -> None:
        """
        Save all metrics to a JSON file
        """
        with open(path, "w") as f:
            json.dump(self.metrics, f)
