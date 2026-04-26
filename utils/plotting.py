import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Optional


def plot_returns(returns: List[float], window: int = 100, save_path: Optional[str] = None):
    """
    Plot training returns with moving average
    """
    plt.figure(figsize=(10, 6))

    # Plot raw returns
    plt.plot(returns, alpha=0.3, label='Raw Return')

    # Plot moving average
    if len(returns) >= window:
        moving_avg = np.convolve(returns, np.ones(window) / window, mode='valid')
        plt.plot(range(window - 1, len(returns)), moving_avg, label=f'Moving Avg ({window})', linewidth=2)

    plt.xlabel('Episode')
    plt.ylabel('Return')
    plt.title('Training Returns')
    plt.legend()
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

    plt.close()


def compare_algorithms(returns_dict: Dict[str, List[float]], window: int = 50, save_path: Optional[str] = None):
    """
    Plot multiple algorithms' returns for comparison
    """
    plt.figure(figsize=(12, 7))

    for algo, returns in returns_dict.items():
        if len(returns) >= window:
            moving_avg = np.convolve(returns, np.ones(window) / window, mode='valid')
            plt.plot(range(window - 1, len(returns)), moving_avg, label=algo.upper(), linewidth=2)
        else:
            plt.plot(returns, label=algo.upper(), linewidth=2, alpha=0.7)

    plt.xlabel('Episode')
    plt.ylabel('Return')
    plt.title('Algorithm Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

    plt.close()
