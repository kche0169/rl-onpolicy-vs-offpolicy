"""
批量运行多种子实验的脚本
"""
import os
import json
import argparse
import gymnasium as gym
import numpy as np
from typing import Dict, List

from algorithms import REINFORCE, A2C, PPO, DQN, SAC
from config import (
    REINFORCEConfig,
    A2CConfig,
    PPOConfig,
    DQNConfig,
    SACConfig,
    get_env_config
)
from utils import (
    set_seed,
    get_device,
    train_reinforce,
    train_a2c,
    train_ppo,
    train_dqn,
    train_sac,
    evaluate,
    plot_returns,
    compare_algorithms
)


def run_single_experiment(
    algo: str,
    env_name: str,
    seed: int,
    num_episodes: int,
    num_eval_episodes: int,
    save_dir: str = "results"
) -> Dict:
    """运行单个实验"""
    set_seed(seed)

    # Create environment
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]

    if hasattr(env.action_space, 'n'):
        action_dim = env.action_space.n
        discrete = True
        max_action = 1.0
    else:
        action_dim = env.action_space.shape[0]
        discrete = False
        max_action = float(env.action_space.high[0])

    # Create agent
    if algo == "reinforce":
        config = REINFORCEConfig()
        agent = REINFORCE(state_dim, action_dim, config)
        returns = train_reinforce(env, agent, num_episodes=num_episodes, verbose=True)
    elif algo == "a2c":
        config = A2CConfig()
        agent = A2C(state_dim, action_dim, config)
        returns = train_a2c(env, agent, num_episodes=num_episodes, verbose=True)
    elif algo == "ppo":
        config = PPOConfig()
        agent = PPO(state_dim, action_dim, config)
        returns = train_ppo(env, agent, num_episodes=num_episodes, verbose=True)
    elif algo == "dqn":
        config = DQNConfig()
        agent = DQN(state_dim, action_dim, config)
        returns = train_dqn(env, agent, num_episodes=num_episodes, verbose=True)
    elif algo == "sac":
        config = SACConfig()
        agent = SAC(state_dim, action_dim, max_action, config)
        returns = train_sac(env, agent, num_episodes=num_episodes, verbose=True)
    else:
        raise ValueError(f"Unknown algorithm: {algo}")

    # Evaluate
    avg_return, std_return = evaluate(env, agent, num_eval_episodes=num_eval_episodes)

    # Save results
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{algo}_{env_name}_seed{seed}.json")
    with open(save_path, 'w') as f:
        json.dump({
            "algo": algo,
            "env": env_name,
            "seed": seed,
            "returns": returns,
            "eval_avg": avg_return,
            "eval_std": std_return,
        }, f)

    return {
        "algo": algo,
        "env": env_name,
        "seed": seed,
        "returns": returns,
        "eval_avg": avg_return,
        "eval_std": std_return,
    }


def run_experiments(
    algos: List[str],
    env_name: str,
    seeds: List[int],
    num_episodes: int,
    num_eval_episodes: int,
    save_dir: str = "results"
) -> Dict:
    """批量运行实验"""
    all_results = {}

    for algo in algos:
        print(f"\n{'='*50}")
        print(f"Running {algo.upper()} on {env_name}")
        print(f"{'='*50}")
        all_results[algo] = []
        for seed in seeds:
            print(f"\nSeed {seed}")
            result = run_single_experiment(
                algo, env_name, seed, num_episodes, num_eval_episodes, save_dir)
            all_results[algo].append(result)

    # Save all results
    all_save_path = os.path.join(save_dir, f"all_results_{env_name}.json")
    with open(all_save_path, 'w') as f:
        json.dump(all_results, f)

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="批量运行RL实验")
    parser.add_argument("--algos", type=str, nargs='+',
                       default=["reinforce", "a2c", "ppo", "dqn", "sac"],
                       help="要运行的算法列表")
    parser.add_argument("--env", type=str, default="CartPole-v1",
                       help="环境名称")
    parser.add_argument("--seeds", type=int, nargs='+',
                       default=[42, 123, 456, 789, 101112],
                       help="随机种子列表")
    parser.add_argument("--num_episodes", type=int, default=500,
                       help="训练回合数")
    parser.add_argument("--num_eval_episodes", type=int, default=10,
                       help="评估回合数")
    parser.add_argument("--save_dir", type=str, default="results",
                       help="保存目录")
    args = parser.parse_args()

    print("="*60)
    print("开始运行RL实验")
    print("="*60)
    print(f"算法: {args.algos}")
    print(f"环境: {args.env}")
    print(f"种子: {args.seeds}")
    print(f"训练回合: {args.num_episodes}")
    print(f"评估回合: {args.num_eval_episodes}")
    print(f"保存目录: {args.save_dir}")
    print("="*60)

    results = run_experiments(
        args.algos,
        args.env,
        args.seeds,
        args.num_episodes,
        args.num_eval_episodes,
        args.save_dir
    )

    print("\n实验完成!")
"""
Placeholder for running multiple experiments
"""
