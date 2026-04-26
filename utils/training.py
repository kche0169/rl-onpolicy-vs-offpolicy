import gymnasium as gym
from typing import Any, Tuple, Dict
import torch


def train_reinforce(env: gym.Env, agent: Any, num_episodes: int = 1000, verbose: bool = True) -> list:
    """
    Train REINFORCE agent on the environment
    """
    episode_returns = []

    for episode in range(num_episodes):
        state, _ = env.reset()
        states = []
        actions = []
        log_probs = []
        rewards = []
        dones = []
        total_reward = 0

        done = False
        while not done:
            states.append(state)
            action, log_prob = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            actions.append(action)
            log_probs.append(log_prob)
            rewards.append(reward)
            dones.append(done)
            total_reward += reward

            state = next_state

        metrics = agent.update(states, actions, log_probs, rewards, dones)
        episode_returns.append(total_reward)

        if verbose and (episode + 1) % 100 == 0:
            avg_return = sum(episode_returns[-100:]) / min(100, len(episode_returns))
            print(f"Episode {episode + 1:4d} | Return: {total_reward:6.2f} | Avg(100): {avg_return:6.2f} | "
                  f"Policy Loss: {metrics['policy_loss']:.4f} | Value Loss: {metrics['value_loss']:.4f}")

    return episode_returns


def train_a2c(env: gym.Env, agent: Any, num_episodes: int = 1000, num_steps: int = 5, verbose: bool = True) -> list:
    """
    Train A2C agent on the environment
    """
    episode_returns = []

    for episode in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0

        states = []
        actions = []
        log_probs = []
        values = []
        rewards = []
        next_states = []
        dones = []

        done = False
        while not done:
            states_ep = []
            actions_ep = []
            log_probs_ep = []
            values_ep = []
            rewards_ep = []
            next_states_ep = []
            dones_ep = []

            for _ in range(num_steps):
                if done:
                    break
                action, log_prob, value = agent.select_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                states_ep.append(state)
                actions_ep.append(action)
                log_probs_ep.append(log_prob)
                values_ep.append(value)
                rewards_ep.append(reward)
                next_states_ep.append(next_state)
                dones_ep.append(done)
                total_reward += reward

                state = next_state

            states.extend(states_ep)
            actions.extend(actions_ep)
            log_probs.extend(log_probs_ep)
            values.extend(values_ep)
            rewards.extend(rewards_ep)
            next_states.extend(next_states_ep)
            dones.extend(dones_ep)

            metrics = agent.update(states_ep, actions_ep, log_probs_ep, values_ep, rewards_ep, next_states_ep, dones_ep)

        episode_returns.append(total_reward)

        if verbose and (episode + 1) % 100 == 0:
            avg_return = sum(episode_returns[-100:]) / min(100, len(episode_returns))
            print(f"Episode {episode + 1:4d} | Return: {total_reward:6.2f} | Avg(100): {avg_return:6.2f} | "
                  f"Policy Loss: {metrics.get('policy_loss', 0):.4f} | Value Loss: {metrics.get('value_loss', 0):.4f}")

    return episode_returns


def train_ppo(env: gym.Env, agent: Any, num_episodes: int = 200, num_steps: int = 2048, verbose: bool = True) -> list:
    """
    Train PPO agent on the environment
    """
    episode_returns = []

    for episode in range(num_episodes):
        states = []
        actions = []
        old_log_probs = []
        rewards = []
        next_states = []
        dones = []

        state, _ = env.reset()
        total_reward = 0

        done = False
        while len(states) < num_steps:
            if done:
                state, _ = env.reset()
                done = False

            action, log_prob, value = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            states.append(state)
            actions.append(action)
            old_log_probs.append(log_prob)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)
            total_reward += reward

            state = next_state

        metrics = agent.update(states, actions, old_log_probs, rewards, next_states, dones)
        episode_returns.append(total_reward)

        if verbose and (episode + 1) % 10 == 0:
            avg_return = sum(episode_returns[-10:]) / min(10, len(episode_returns))
            print(f"Episode {episode + 1:4d} | Return: {total_reward:6.2f} | Avg(10): {avg_return:6.2f} | "
                  f"Policy Loss: {metrics.get('policy_loss', 0):.4f} | Value Loss: {metrics.get('value_loss', 0):.4f}")

    return episode_returns


def train_dqn(env: gym.Env, agent: Any, num_episodes: int = 500, verbose: bool = True) -> list:
    """
    Train DQN agent on the environment
    """
    episode_returns = []

    for episode in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0

        done = False
        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.replay_buffer.push(state, action, reward, next_state, done)
            metrics = agent.update()
            total_reward += reward

            state = next_state

        episode_returns.append(total_reward)

        if verbose and (episode + 1) % 50 == 0:
            avg_return = sum(episode_returns[-50:]) / min(50, len(episode_returns))
            print(f"Episode {episode + 1:4d} | Return: {total_reward:6.2f} | Avg(50): {avg_return:6.2f} | "
                  f"Q Loss: {metrics.get('q_loss', 0):.4f} | Epsilon: {metrics.get('epsilon', 0):.4f}")

    return episode_returns


def train_sac(env: gym.Env, agent: Any, num_episodes: int = 500, verbose: bool = True) -> list:
    """
    Train SAC agent on the environment
    """
    episode_returns = []

    for episode in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0

        done = False
        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.replay_buffer.push(state, action, reward, next_state, done)
            metrics = agent.update()
            total_reward += reward

            state = next_state

        episode_returns.append(total_reward)

        if verbose and (episode + 1) % 50 == 0:
            avg_return = sum(episode_returns[-50:]) / min(50, len(episode_returns))
            print(f"Episode {episode + 1:4d} | Return: {total_reward:6.2f} | Avg(50): {avg_return:6.2f} | "
                  f"Q Loss: {metrics.get('q_loss', 0):.4f} | Actor Loss: {metrics.get('actor_loss', 0):.4f}")

    return episode_returns


def evaluate(env: gym.Env, agent: Any, num_eval_episodes: int = 10, render: bool = False) -> Tuple[float, float]:
    """
    Evaluate the agent on the environment
    """
    returns = []

    for _ in range(num_eval_episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            if render:
                env.render()
            # Handle different agent select_action signatures
            result = agent.select_action(state)
            if isinstance(result, tuple):
                action = result[0]
            else:
                action = result

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            state = next_state

        returns.append(total_reward)

    avg_return = sum(returns) / len(returns)
    std_return = (sum((r - avg_return) ** 2 for r in returns) / len(returns)) ** 0.5

    return avg_return, std_return
