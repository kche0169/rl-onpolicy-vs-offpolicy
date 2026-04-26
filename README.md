# Multi-Algorithm Comparative Study: On-Policy vs Off-Policy Deep RL in Gymnasium Classic Control Tasks

## 项目介绍

本项目实现并对比了 On-Policy 和 Off-Policy 的深度强化学习算法在 Gymnasium 经典控制任务上的表现。

### 实现的算法

- **On-Policy**：
  - REINFORCE（带 baseline）
  - A2C
  - PPO（核心算法）
- **Off-Policy**：
  - DQN（适用于离散动作空间）
  - SAC（适用于连续动作空间）

### 推荐环境

- CartPole-v1（简单验证）
- LunarLander-v2（中等难度，离散动作）
- Pendulum-v1（连续动作空间）

## 项目结构

```
/home/kche/work/RL/
├── config/
│   ├── __init__.py
│   ├── hyperparameters.py       # 各算法的超参数配置
│   └── env_config.py             # 环境配置
├── algorithms/
│   ├── __init__.py
│   ├── base/
│   │   ├── __init__.py
│   │   ├── base_agent.py         # 基类（所有算法的公共接口）
│   │   └── replay_buffer.py      # 共享的ReplayBuffer（Off-Policy用）
│   ├── on_policy/
│   │   ├── __init__.py
│   │   ├── reinforce.py          # REINFORCE (带baseline)
│   │   ├── a2c.py                 # A2C
│   │   └── ppo.py                 # PPO (核心，手动实现)
│   └── off_policy/
│       ├── __init__.py
│       ├── dqn.py                 # DQN
│       └── sac.py                 # SAC (连续控制)
├── envs/
│   ├── __init__.py
│   └── wrapper.py                # 环境封装
├── utils/
│   ├── __init__.py
│   ├── training.py                # 训练循环、eval逻辑
│   ├── logger.py                  # 简单日志
│   ├── plotting.py                # 绘图功能
│   └── utils.py                   # 工具函数（seed设置等）
├── main.py                        # 主入口
├── run_experiments.py             # 实验批量运行脚本
├── requirements.txt
└── README.md
```

## 使用说明

### 安装依赖

```bash
pip install -r requirements.txt
```

### 运行单个算法

```bash
# 运行 REINFORCE（默认）
python main.py

# 指定算法
python main.py --algo a2c
python main.py --algo ppo
python main.py --algo dqn

# 指定环境和回合数
python main.py --algo reinforce --env CartPole-v1 --num_episodes 500
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--algo` | reinforce | 选择算法 (reinforce/a2c/ppo/dqn/sac) |
| `--env` | CartPole-v1 | 训练环境 |
| `--num_episodes` | 1000 | 训练回合数 |
| `--num_eval_episodes` | 10 | 评估回合数 |
| `--seed` | 42 | 随机种子 |
| `--verbose` | True | 是否打印训练进度 |

## 实验设置建议

- 使用 5-8 个不同随机种子
- 用 Stable-Baselines3 跑 Baseline 对比
- 记录每个算法的收敛速度、最终回报、训练稳定性

## Abstract

This project conducts a multi-algorithm comparative study of on-policy and off-policy deep reinforcement learning methods in Gymnasium classic control tasks. We implement and evaluate REINFORCE (with baseline), A2C, PPO (on-policy), as well as DQN and SAC (off-policy) on CartPole-v1, LunarLander-v2, and Pendulum-v1. Experiments with multiple random seeds analyze performance in terms of average return, convergence speed, stability, and sample efficiency. Results highlight the trade-offs between on-policy stability and off-policy sample efficiency, providing insights into algorithm selection for classic control problems.