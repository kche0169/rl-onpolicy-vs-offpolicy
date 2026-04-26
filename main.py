import sys
import os
import argparse

import gymnasium as gym
import numpy as np

from algorithms.on_policy.reinforce import REINFORCE
from algorithms.on_policy.a2c import A2C
from algorithms.on_policy.ppo import PPO
from algorithms.off_policy.dqn import DQN
from algorithms.off_policy.sac import SAC
from config.hyperparameters import (
    REINFORCEConfig,
    A2CConfig,
    PPOConfig,
    DQNConfig,
    SACConfig
)
from utils.training import (
    train_reinforce,
    train_a2c,
    train_ppo,
    train_dqn,
    train_sac,
    evaluate
)
from utils.utils import set_seed


def get_args():
    parser = argparse.ArgumentParser(description="RL Algorithm Comparison")
    parser.add_argument("--algo", type=str, default="reinforce",
                       choices=["reinforce", "a2c", "ppo", "dqn", "sac"],
                       help="Algorithm to use")
    parser.add_argument("--env", type=str, default="CartPole-v1",
                       help="Environment to use")
    parser.add_argument("--num_episodes", type=int, default=1000,
                       help="Number of training episodes")
    parser.add_argument("--num_eval_episodes", type=int, default=10,
                       help="Number of evaluation episodes")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--verbose", action="store_true", default=True,
                       help="Print training progress")
    return parser.parse_args()


def main():
    args = get_args()
    set_seed(args.seed)

    # Create environment
    env = gym.make(args.env)
    state_dim = env.observation_space.shape[0]

    # Check if discrete or continuous action space
    if hasattr(env.action_space, 'n'):
        action_dim = env.action_space.n
        discrete = True
        max_action = 1.0
    else:
        action_dim = env.action_space.shape[0]
        discrete = False
        max_action = float(env.action_space.high[0])

    # Create agent
    if args.algo == "reinforce":
        config = REINFORCEConfig()
        agent = REINFORCE(state_dim, action_dim, config)
    elif args.algo == "a2c":
        config = A2CConfig()
        agent = A2C(state_dim, action_dim, config)
    elif args.algo == "ppo":
        config = PPOConfig()
        agent = PPO(state_dim, action_dim, config)
    elif args.algo == "dqn":
        if not discrete:
            print("DQN only supports discrete action spaces!")
            return
        config = DQNConfig()
        agent = DQN(state_dim, action_dim, config)
    elif args.algo == "sac":
        if discrete:
            print("SAC only supports continuous action spaces!")
            return
        config = SACConfig()
        agent = SAC(state_dim, action_dim, max_action, config)
    else:
        print(f"Unknown algorithm: {args.algo}")
        return

    # Train agent
    print(f"Training {args.algo.upper()} on {args.env}...")
    if args.algo == "reinforce":
        returns = train_reinforce(env, agent, num_episodes=args.num_episodes, verbose=args.verbose)
    elif args.algo == "a2c":
        returns = train_a2c(env, agent, num_episodes=args.num_episodes, verbose=args.verbose)
    elif args.algo == "ppo":
        returns = train_ppo(env, agent, num_episodes=args.num_episodes, verbose=args.verbose)
    elif args.algo == "dqn":
        returns = train_dqn(env, agent, num_episodes=args.num_episodes, verbose=args.verbose)
    elif args.algo == "sac":
        returns = train_sac(env, agent, num_episodes=args.num_episodes, verbose=args.verbose)

    # Evaluate agent
    print("\nEvaluating...")
    avg_return, std_return = evaluate(env, agent, num_eval_episodes=args.num_eval_episodes)
    print(f"Average Return: {avg_return:.2f} ± {std_return:.2f}")


if __name__ == "__main__":
    main()
