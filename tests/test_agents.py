"""
Tests for RL agents
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__))))

import torch
import numpy as np
from algorithms.on_policy.reinforce import REINFORCE
from algorithms.on_policy.a2c import A2C
from algorithms.on_policy.ppo import PPO
from algorithms.off_policy.dqn import DQN
from algorithms.off_policy.sac import SAC
from utils.utils import set_seed


def test_reinforce_init():
    """Test REINFORCE agent initialization"""
    set_seed(42)
    agent = REINFORCE(state_dim=4, action_dim=2)

    assert agent.state_dim == 4, "State dim should be 4"
    assert agent.action_dim == 2, "Action dim should be 2"
    assert isinstance(agent.policy, torch.nn.Module), "Should have policy network"
    assert isinstance(agent.value, torch.nn.Module), "Should have value network"
    print("✓ test_reinforce_init passed")


def test_reinforce_select_action():
    """Test REINFORCE action selection"""
    set_seed(42)
    agent = REINFORCE(state_dim=4, action_dim=2)

    state = [0.1, 0.2, 0.3, 0.4]
    action, log_prob = agent.select_action(state)

    assert isinstance(action, int), "Action should be integer"
    assert 0 <= action < 2, f"Action should be 0 or 1, got {action}"
    assert isinstance(log_prob, torch.Tensor), "Log prob should be tensor"
    print("✓ test_reinforce_select_action passed")


def test_reinforce_update():
    """Test REINFORCE update step"""
    set_seed(42)
    agent = REINFORCE(state_dim=4, action_dim=2)

    # Collect real data by simulating a rollout
    states = []
    actions = []
    log_probs = []
    rewards = []
    dones = []

    state = [0.1, 0.2, 0.3, 0.4]
    for i in range(5):
        action, log_prob = agent.select_action(state)
        states.append(state.copy())
        actions.append(action)
        log_probs.append(log_prob)
        rewards.append(1.0)
        dones.append(i == 4)

    metrics = agent.update(states, actions, log_probs, rewards, dones)

    assert "policy_loss" in metrics, "Should return policy_loss"
    assert "value_loss" in metrics, "Should return value_loss"
    assert isinstance(metrics["policy_loss"], float), "Policy loss should be float"
    assert isinstance(metrics["value_loss"], float), "Value loss should be float"
    print("✓ test_reinforce_update passed")


def test_a2c_init():
    """Test A2C agent initialization"""
    set_seed(42)
    agent = A2C(state_dim=4, action_dim=2)

    assert agent.state_dim == 4, "State dim should be 4"
    assert agent.action_dim == 2, "Action dim should be 2"
    print("✓ test_a2c_init passed")


def test_a2c_select_action():
    """Test A2C action selection"""
    set_seed(42)
    agent = A2C(state_dim=4, action_dim=2)

    state = [0.1, 0.2, 0.3, 0.4]
    action, log_prob, value = agent.select_action(state)

    assert isinstance(action, int), "Action should be integer"
    assert 0 <= action < 2, f"Action should be 0 or 1, got {action}"
    assert isinstance(log_prob, torch.Tensor), "Log prob should be tensor"
    assert isinstance(value, torch.Tensor), "Value should be tensor"
    print("✓ test_a2c_select_action passed")


def test_dqn_init():
    """Test DQN agent initialization"""
    set_seed(42)
    agent = DQN(state_dim=4, action_dim=2)

    assert agent.state_dim == 4, "State dim should be 4"
    assert agent.action_dim == 2, "Action dim should be 2"
    assert hasattr(agent, 'q_net'), "Should have Q network"
    assert hasattr(agent, 'target_q_net'), "Should have target Q network"
    assert hasattr(agent, 'replay_buffer'), "Should have replay buffer"
    print("✓ test_dqn_init passed")


def test_dqn_select_action():
    """Test DQN action selection (epsilon-greedy)"""
    set_seed(42)
    agent = DQN(state_dim=4, action_dim=2)

    state = [0.1, 0.2, 0.3, 0.4]
    action = agent.select_action(state)

    assert isinstance(action, int), "Action should be integer"
    assert 0 <= action < 2, f"Action should be 0 or 1, got {action}"
    print("✓ test_dqn_select_action passed")


def test_sac_init():
    """Test SAC agent initialization"""
    set_seed(42)
    agent = SAC(state_dim=4, action_dim=2, max_action=1.0)

    assert agent.state_dim == 4, "State dim should be 4"
    assert agent.action_dim == 2, "Action dim should be 2"
    assert hasattr(agent, 'actor'), "Should have actor network"
    assert hasattr(agent, 'critic'), "Should have critic network"
    assert hasattr(agent, 'replay_buffer'), "Should have replay buffer"
    print("✓ test_sac_init passed")


def test_sac_select_action():
    """Test SAC action selection"""
    set_seed(42)
    agent = SAC(state_dim=4, action_dim=2, max_action=1.0)

    state = [0.1, 0.2, 0.3, 0.4]
    action = agent.select_action(state)

    assert isinstance(action, np.ndarray), "Action should be numpy array"
    assert action.shape == (2,), f"Action shape should be (2,), got {action.shape}"
    print("✓ test_sac_select_action passed")


def test_all_agents_save_load(tmp_path=None):
    """Test that all agents can save and load"""
    import tempfile

    set_seed(42)

    if tmp_path is None:
        tmp_path = tempfile.mkdtemp()

    agents_to_test = [
        ("REINFORCE", REINFORCE(state_dim=4, action_dim=2)),
        ("A2C", A2C(state_dim=4, action_dim=2)),
        ("DQN", DQN(state_dim=4, action_dim=2)),
        ("SAC", SAC(state_dim=4, action_dim=2, max_action=1.0)),
    ]

    for name, agent in agents_to_test:
        save_path = os.path.join(tmp_path, f"{name}_test.pt")

        # Save
        agent.save(save_path)
        assert os.path.exists(save_path), f"{name} should save to file"

        # Load - create new agent and load
        if name == "REINFORCE":
            new_agent = REINFORCE(state_dim=4, action_dim=2)
        elif name == "A2C":
            new_agent = A2C(state_dim=4, action_dim=2)
        elif name == "DQN":
            new_agent = DQN(state_dim=4, action_dim=2)
        else:
            new_agent = SAC(state_dim=4, action_dim=2, max_action=1.0)

        new_agent.load(save_path)

        # Verify by selecting action (should work without error)
        state = [0.1, 0.2, 0.3, 0.4]
        action = new_agent.select_action(state)
        assert action is not None, f"{name} should select action after load"

        print(f"✓ test_all_agents_save_load ({name}) passed")


if __name__ == "__main__":
    print("Running agent tests...")

    test_reinforce_init()
    test_reinforce_select_action()
    test_reinforce_update()

    test_a2c_init()
    test_a2c_select_action()

    test_dqn_init()
    test_dqn_select_action()

    test_sac_init()
    test_sac_select_action()

    test_all_agents_save_load()

    print("\n✅ All agent tests passed!")
