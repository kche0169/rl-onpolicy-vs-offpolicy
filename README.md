# 强化学习算法对比项目

> On-Policy vs Off-Policy 深度强化学习算法在 Gymnasium 经典控制任务上的对比实现

---

## 📋 目录

- [项目简介](#项目简介)
- [安装说明](#安装说明)
- [快速开始](#快速开始)
- [项目结构](#项目结构)
- [算法说明](#算法说明)
- [运行测试](#运行测试)
- [可视化结果](#可视化结果)
- [实验建议](#实验建议)

---

## 项目简介

本项目从零实现了 5 种经典强化学习算法，在 Gymnasium 经典控制任务上进行对比实验，分析 On-Policy 与 Off-Policy 方法在收敛速度、稳定性和样本效率上的差异。

### 实现的算法

| 类型 | 算法 | 说明 |
|------|------|------|
| On-Policy | REINFORCE | 带 baseline 的策略梯度 |
| On-Policy | A2C | Advantage Actor-Critic |
| On-Policy | PPO | Proximal Policy Optimization（核心） |
| Off-Policy | DQN | Deep Q-Network（离散动作） |
| Off-Policy | SAC | Soft Actor-Critic（连续动作） |

### 实验环境

| 环境 | 难度 | 动作空间 | 用途 |
|------|------|---------|------|
| CartPole-v1 | 简单 | 离散 | 入门验证 |
| LunarLander-v3 | 中等 | 离散 | 核心对比 |
| Pendulum-v1 | 中等 | 连续 | SAC 验证 |

---

## 安装说明

```bash
# 1. 创建虚拟环境
conda create -n rl_study python=3.10 -y
conda activate rl_study

# 2. 安装依赖（使用清华镜像加速）
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

---

## 快速开始

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
| `--algo` | reinforce | 算法选择 (reinforce / a2c / ppo / dqn / sac) |
| `--env` | CartPole-v1 | 训练环境 |
| `--num_episodes` | 1000 | 训练回合数 |
| `--num_eval_episodes` | 10 | 评估回合数 |
| `--seed` | 42 | 随机种子 |
| `--verbose` | True | 打印训练进度 |

---

## 项目结构

```
RL/
├── config/                  # 配置
│   ├── hyperparameters.py   #   超参数
│   └── env_config.py        #   环境配置
├── algorithms/              # 算法
│   ├── base/
│   │   ├── base_agent.py    #   基类接口
│   │   └── replay_buffer.py #   经验回放
│   ├── on_policy/           #   On-Policy 算法
│   │   ├── reinforce.py
│   │   ├── a2c.py
│   │   └── ppo.py
│   └── off_policy/          #   Off-Policy 算法
│       ├── dqn.py
│       └── sac.py
├── envs/
│   └── wrapper.py           # 环境封装
├── utils/                   # 工具
│   ├── training.py          #   训练循环
│   ├── logger.py            #   日志
│   ├── plotting.py          #   绘图
│   └── utils.py             #   seed / device 等
├── tests/                   # 测试
│   ├── test_utils.py
│   ├── test_replay_buffer.py
│   └── test_agents.py
├── main.py                  # 主入口
├── run_experiments.py       # 批量实验
├── visualize_results.py     # 可视化结果
└── requirements.txt
```

---

## 算法说明

### REINFORCE with Baseline
- 策略网络输出动作概率，价值网络提供 baseline
- 优势函数：`A = G - V(s)`
- 损失函数：`L = -log π(a|s) × A`

### A2C (Advantage Actor-Critic)
- 同步版本的 Actor-Critic
- 用 TD 误差代替完整回报估算优势
- Actor 学习策略，Critic 学习价值函数

### PPO (Proximal Policy Optimization)
- 核心算法，策略梯度最先进的实现之一
- Clipping 机制限制策略更新幅度
- GAE（广义优势估算）减小方差

### DQN (Deep Q-Network)
- 适用于离散动作空间
- 经验回放打破数据相关性
- 目标网络稳定训练，ε-贪婪平衡探索

### SAC (Soft Actor-Critic)
- 适用于连续动作空间
- 最大熵框架：同时优化回报和策略随机性
- 双 Q 网络缓解过估计，自动调整温度系数

---

## 运行测试

```bash
# 运行全部测试
pytest tests/ -v

# 运行单个测试
pytest tests/test_agents.py -v

# 生成覆盖率报告
pytest tests/ --cov=. --cov-report=html
```

| 测试文件 | 覆盖内容 |
|---------|---------|
| `test_utils.py` | seed / device / compute_returns |
| `test_replay_buffer.py` | push / sample / capacity |
| `test_agents.py` | 初始化 / select_action / save & load |

---

## 可视化结果

### 1. 查看汇总表
```bash
cat results/summary_CartPole-v1.txt
```

### 2. 生成对比图
```bash
# 自动生成所有环境的对比图和汇总表
python visualize_results.py

# 只看特定环境
python visualize_results.py --env CartPole-v1
python visualize_results.py --env LunarLander-v3
python visualize_results.py --env Pendulum-v1
```

### 3. 查看图片
生成的图片会保存在 `results/` 目录下，文件名格式为 `comparison_*.png`。

---

## 实验建议

- 使用 **5-8 个随机种子**消除随机性影响
- 用 **Stable-Baselines3** 跑 Baseline 对比
- 关注指标：平均回报、收敛速度、训练稳定性、样本效率

---

## Abstract

This project conducts a multi-algorithm comparative study of on-policy and off-policy deep reinforcement learning methods in Gymnasium classic control tasks. We implement and evaluate REINFORCE (with baseline), A2C, PPO (on-policy), as well as DQN and SAC (off-policy) on CartPole-v1, LunarLander-v3, and Pendulum-v1. Experiments with multiple random seeds analyze performance in terms of average return, convergence speed, stability, and sample efficiency. Results highlight the trade-offs between on-policy stability and off-policy sample efficiency, providing insights into algorithm selection for classic control problems.
