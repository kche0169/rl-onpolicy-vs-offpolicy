"""
Tests for utils functions
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__))))

import random
import numpy as np
import torch
from utils.utils import set_seed, get_device, compute_returns


def test_set_seed():
    """Test that set_seed produces reproducible results"""
    set_seed(42)
    rand_val1 = random.randint(0, 10000)
    np_val1 = np.random.randint(0, 10000)
    torch_val1 = torch.randint(0, 10000, (1,)).item()

    set_seed(42)
    rand_val2 = random.randint(0, 10000)
    np_val2 = np.random.randint(0, 10000)
    torch_val2 = torch.randint(0, 10000, (1,)).item()

    assert rand_val1 == rand_val2, "Random values should be same with same seed"
    assert np_val1 == np_val2, "NumPy values should be same with same seed"
    assert torch_val1 == torch_val2, "Torch values should be same with same seed"
    print("✓ test_set_seed passed")


def test_get_device():
    """Test that get_device returns valid device"""
    device = get_device()
    assert isinstance(device, torch.device), "Should return torch.device"
    assert device.type in ["cuda", "cpu"], "Device should be cuda or cpu"
    print(f"✓ test_get_device passed (device: {device})")


def test_compute_returns():
    """Test discounted return computation"""
    rewards = [1.0, 1.0, 1.0]
    gamma = 0.99

    returns = compute_returns(rewards, gamma)

    # G_3 = r_3 = 1.0
    # G_2 = r_2 + gamma * G_3 = 1.0 + 0.99 * 1.0 = 1.99
    # G_1 = r_1 + gamma * G_2 = 1.0 + 0.99 * 1.99 = 2.9701
    expected = [1.0 + 0.99 * 1.99, 1.0 + 0.99 * 1.0, 1.0]

    assert len(returns) == len(rewards), "Should have same length as rewards"
    for r, e in zip(returns, expected):
        assert abs(r - e) < 1e-6, f"Return {r} should be close to {e}"

    print("✓ test_compute_returns passed")


def test_compute_returns_zero_gamma():
    """Test compute_returns with gamma=0 (no discount)"""
    rewards = [1.0, 2.0, 3.0]
    returns = compute_returns(rewards, gamma=0.0)

    # With gamma=0, G_t = r_t
    assert returns == [1.0, 2.0, 3.0], "With gamma=0, return should equal immediate reward"
    print("✓ test_compute_returns_zero_gamma passed")


def test_compute_returns_single():
    """Test compute_returns with single reward"""
    returns = compute_returns([5.0], gamma=0.99)
    assert len(returns) == 1, "Should have length 1"
    assert abs(returns[0] - 5.0) < 1e-6, "Single return should equal reward"
    print("✓ test_compute_returns_single passed")


if __name__ == "__main__":
    print("Running utils tests...")
    test_set_seed()
    test_get_device()
    test_compute_returns()
    test_compute_returns_zero_gamma()
    test_compute_returns_single()
    print("\n✅ All utils tests passed!")
