"""
批量运行多种子实验的脚本

用法:
    python run_experiments.py --algos reinforce a2c ppo --env CartPole-v1 --seeds 42 123 456

输出:
    results/
    ├── reinforce_CartPole-v1_seed42.json   # 每个实验的详细数据
    ├── a2c_CartPole-v1_seed42.json
    ├── all_results_CartPole-v1.json         # 所有实验汇总
    ├── comparison_CartPole-v1.png           # 对比曲线图
    └── summary_CartPole-v1.txt              # 结果汇总表
"""
import os
import json
import argparse
import time
from datetime import datetime

import gymnasium as gym
import numpy as np
from typing import Dict, List

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
from utils.utils import set_seed, get_device
from utils.training import (
    train_reinforce,
    train_a2c,
    train_ppo,
    train_dqn,
    train_sac,
    evaluate
)
from utils.plotting import plot_returns, compare_algorithms


def create_agent(algo: str, state_dim: int, action_dim: int, max_action: float = 1.0):
    """根据算法名创建智能体"""
    if algo == "reinforce":
        return REINFORCE(state_dim, action_dim, REINFORCEConfig())
    elif algo == "a2c":
        return A2C(state_dim, action_dim, A2CConfig())
    elif algo == "ppo":
        return PPO(state_dim, action_dim, PPOConfig())
    elif algo == "dqn":
        return DQN(state_dim, action_dim, DQNConfig())
    elif algo == "sac":
        return SAC(state_dim, action_dim, max_action, SACConfig())
    else:
        raise ValueError(f"Unknown algorithm: {algo}")


TRAIN_FN = {
    "reinforce": train_reinforce,
    "a2c": train_a2c,
    "ppo": train_ppo,
    "dqn": train_dqn,
    "sac": train_sac,
}


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

    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]

    if hasattr(env.action_space, 'n'):
        action_dim = env.action_space.n
        max_action = 1.0
    else:
        action_dim = env.action_space.shape[0]
        max_action = float(env.action_space.high[0])

    agent = create_agent(algo, state_dim, action_dim, max_action)

    t0 = time.time()
    returns = TRAIN_FN[algo](env, agent, num_episodes=num_episodes, verbose=True)
    train_time = time.time() - t0

    avg_return, std_return = evaluate(env, agent, num_eval_episodes=num_eval_episodes)

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
            "train_time_sec": round(train_time, 1),
        }, f)

    return {
        "algo": algo,
        "env": env_name,
        "seed": seed,
        "returns": returns,
        "eval_avg": avg_return,
        "eval_std": std_return,
        "train_time_sec": round(train_time, 1),
    }


def print_summary(all_results: Dict, env_name: str, save_dir: str):
    """打印并保存结果汇总表"""
    header = f"{'Algorithm':<12} {'Avg Return':>12} {'Std Return':>12} {'Avg Time(s)':>12} {'Seeds':>8}"
    sep = "-" * len(header)

    lines = [
        f"\n{'='*len(header)}",
        f"Results Summary: {env_name}",
        f"{'='*len(header)}",
        header,
        sep,
    ]

    for algo, results in all_results.items():
        avg_returns = [r["eval_avg"] for r in results]
        std_returns = [r["eval_std"] for r in results]
        times = [r["train_time_sec"] for r in results]

        mean_avg = np.mean(avg_returns)
        mean_std = np.mean(std_returns)
        mean_time = np.mean(times)
        n_seeds = len(results)

        lines.append(
            f"{algo.upper():<12} {mean_avg:>12.2f} {mean_std:>12.2f} {mean_time:>12.1f} {n_seeds:>8}"
        )

    lines.append(sep)
    summary = "\n".join(lines)
    print(summary)

    summary_path = os.path.join(save_dir, f"summary_{env_name}.txt")
    with open(summary_path, 'w') as f:
        f.write(summary + "\n")
    print(f"\nSummary saved to {summary_path}")


def plot_comparison(all_results: Dict, env_name: str, save_dir: str, window: int = 50):
    """绘制算法对比图"""
    returns_dict = {}
    for algo, results in all_results.items():
        all_returns = [r["returns"] for r in results]
        min_len = min(len(r) for r in all_returns)
        trimmed = [r[:min_len] for r in all_returns]
        mean_returns = np.mean(trimmed, axis=0).tolist()
        returns_dict[algo] = mean_returns

    plot_path = os.path.join(save_dir, f"comparison_{env_name}.png")
    compare_algorithms(returns_dict, window=window, save_path=plot_path)


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
    total = len(algos) * len(seeds)
    count = 0

    for algo in algos:
        print(f"\n{'='*50}")
        print(f"Running {algo.upper()} on {env_name}")
        print(f"{'='*50}")
        all_results[algo] = []
        for seed in seeds:
            count += 1
            print(f"\n[{count}/{total}] Seed {seed}")
            result = run_single_experiment(
                algo, env_name, seed, num_episodes, num_eval_episodes, save_dir)
            all_results[algo].append(result)

    all_save_path = os.path.join(save_dir, f"all_results_{env_name}.json")
    with open(all_save_path, 'w') as f:
        json.dump(all_results, f)

    print_summary(all_results, env_name, save_dir)
    plot_comparison(all_results, env_name, save_dir)

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

    print(f"\nAll results saved to {args.save_dir}/")
    print("Done!")
