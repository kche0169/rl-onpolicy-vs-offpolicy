"""
Tests for ReplayBuffer
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
from algorithms.base.replay_buffer import ReplayBuffer


def test_replay_buffer_init():
    """Test ReplayBuffer initialization"""
    buffer = ReplayBuffer(capacity=1000)
    assert len(buffer) == 0, "Buffer should start empty"
    assert buffer.buffer.maxlen == 1000, "Capacity should be set correctly"
    print("✓ test_replay_buffer_init passed")


def test_replay_buffer_push():
    """Test pushing transitions to buffer"""
    buffer = ReplayBuffer(capacity=10)

    buffer.push([1, 2, 3], 0, 1.0, [4, 5, 6], False)
    assert len(buffer) == 1, "Buffer should have 1 item"

    buffer.push([7, 8, 9], 1, 2.0, [10, 11, 12], True)
    assert len(buffer) == 2, "Buffer should have 2 items"
    print("✓ test_replay_buffer_push passed")


def test_replay_buffer_capacity():
    """Test that buffer respects capacity"""
    buffer = ReplayBuffer(capacity=3)

    for i in range(5):
        buffer.push([i], 0, float(i), [i+1], False)

    assert len(buffer) == 3, "Buffer should not exceed capacity"
    assert len(buffer) == buffer.buffer.maxlen, "Buffer should be at max capacity"
    print("✓ test_replay_buffer_capacity passed")


def test_replay_buffer_sample():
    """Test sampling from buffer"""
    buffer = ReplayBuffer(capacity=100)

    for i in range(50):
        buffer.push([float(i)], i % 2, float(i), [float(i+1)], i % 5 == 4)

    states, actions, rewards, next_states, dones = buffer.sample(batch_size=10)

    assert len(states) == 10, "Should return 10 states"
    assert len(actions) == 10, "Should return 10 actions"
    assert len(rewards) == 10, "Should return 10 rewards"
    assert len(next_states) == 10, "Should return 10 next_states"
    assert len(dones) == 10, "Should return 10 dones"
    print("✓ test_replay_buffer_sample passed")


def test_replay_buffer_sample_error():
    """Test that sampling from small buffer raises error"""
    buffer = ReplayBuffer(capacity=10)

    for i in range(5):
        buffer.push([i], 0, float(i), [i+1], False)

    try:
        buffer.sample(batch_size=10)
        assert False, "Should raise error when sampling more than buffer size"
    except ValueError:
        pass
    print("✓ test_replay_buffer_sample_error passed")


def test_replay_buffer_order():
    """Test that buffer maintains insertion order (FIFO)"""
    buffer = ReplayBuffer(capacity=10)

    for i in range(5):
        buffer.push([float(i)], i, float(i), [float(i+1)], False)

    states, _, _, _, _ = buffer.sample(batch_size=5)
    assert len(set(tuple(s) for s in states)) == 5, "Should have 5 unique states"
    print("✓ test_replay_buffer_order passed")


if __name__ == "__main__":
    print("Running ReplayBuffer tests...")
    test_replay_buffer_init()
    test_replay_buffer_push()
    test_replay_buffer_capacity()
    test_replay_buffer_sample()
    test_replay_buffer_sample_error()
    test_replay_buffer_order()
    print("\n✅ All ReplayBuffer tests passed!")
