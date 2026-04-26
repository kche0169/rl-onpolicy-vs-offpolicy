**Multi-Algorithm Comparative Study: On-Policy vs Off-Policy Deep RL in Gymnasium Classic Control Tasks** 

### 需要测的算法

- **On-Policy**（重点手动或半手动实现）：
  - REINFORCE（带 baseline）
  - A2C
  - PPO（核心算法）
- **Off-Policy**：
  - DQN（或 Double DQN，适用于 discrete）
  - SAC（适用于 continuous / 混合）

**为什么这些？**\
它们清晰覆盖 on-policy（稳定但样本效率较低） vs off-policy（样本高效但有时不稳定），差异明显，便于讨论收敛速度、稳定性、样本效率。

### 推荐环境（3 个即可，覆盖 discrete + continuous）

- CartPole-v1（简单验证）
- LunarLander-v2（中等难度，discrete）
- Acrobot-v1 或 Pendulum-v1（挑战探索 / continuous）

实验设置：至少 5–8 个随机种子，用 Stable-Baselines3 快速跑 baseline，REINFORCE 和 PPO 部分手动实现（参考 CleanRL）。

### 项目英文简述（Abstract，98 字）

This project conducts a multi-algorithm comparative study of on-policy and off-policy deep reinforcement learning methods in Gymnasium classic control tasks. We implement and evaluate REINFORCE (with baseline), A2C, PPO (on-policy), as well as DQN and SAC (off-policy) on CartPole-v1, LunarLander-v2, and Acrobot-v1. Experiments with multiple random seeds analyze performance in terms of average return, convergence speed, stability, and sample efficiency. Results highlight the trade-offs between on-policy stability and off-policy sample efficiency, providing insights into algorithm selection for classic control problems.

这个简述适合提案（1页摘要）或报告引言，直接对应标题，能体现 MDP 形式化、实验严谨性和对比重点。

如果你想调整算法数量、加具体环境，或需要超参数表 / 代码框架，告诉我，我马上帮你细化！时间还够，加油。

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
│       ├── dqn.py                 # DQN / Double DQN
│       └── sac.py                 # SAC (连续控制)
├── envs/
│   ├── __init__.py
│   └── wrapper.py                # 环境封装、reward scaling等
├── utils/
│   ├── __init__.py
│   ├── training.py                # 训练循环、eval逻辑
│   ├── logger.py                  # TensorBoard / wandb日志
│   └── utils.py                   # 工具函数（seed设置等）
├── main.py                        # 主入口
├── run_experiments.py             # 实验批量运行脚本
├── requirements.txt
└── README.md
```