import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.utils import set_seed, get_device, compute_returns


def test_set_seed():
    print("Testing set_seed()...")
    set_seed(42)
    import random
    import numpy as np
    import torch
    rand_val = random.randint(0, 100)
    np_val = np.random.randint(0, 100)
    torch_val = torch.randint(0, 100, (1,)).item()
    print(f"Random values (seed=42): rand={rand_val}, np={np_val}, torch={torch_val}")
    return True


def test_get_device():
    print("\nTesting get_device()...")
    device = get_device()
    print(f"Device: {device}")
    return True


def test_compute_returns():
    print("\nTesting compute_returns()...")
    rewards = [1.0, 1.0, 1.0]
    gamma = 0.9
    returns = compute_returns(rewards, gamma)
    expected = [1.0 + 0.9 * 1.0 + 0.81 * 1.0, 1.0 + 0.9 * 1.0, 1.0]
    print(f"Rewards: {rewards}")
    print(f"Returns: {returns}")
    print(f"Expected: {expected}")
    return all(abs(r - e) < 1e-6 for r, e in zip(returns, expected))


if __name__ == "__main__":
    print("=" * 50)
    print("Testing utils/utils.py")
    print("=" * 50)
    
    tests = [
        test_set_seed,
        test_get_device,
        test_compute_returns
    ]
    
    all_passed = True
    for test in tests:
        try:
            passed = test()
            if passed:
                print("✅ Passed")
            else:
                print("❌ Failed")
                all_passed = False
        except Exception as e:
            print(f"❌ Failed with error: {e}")
            all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("All tests passed! 🎉")
    else:
        print("Some tests failed! ❌")
    print("=" * 50)
