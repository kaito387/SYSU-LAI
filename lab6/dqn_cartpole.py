"""
DQN (Deep Q-Network) for CartPole-v1
=====================================
实现DQN算法解决CartPole平衡问题。

DQN核心组件:
1. Q-Network: 用神经网络近似Q函数
2. Experience Replay: 打破样本相关性，稳定训练
3. Fixed Q-Target: 使用目标网络稳定训练
4. Double DQN: 减少Q值过高估计
5. Soft Target Update: 平滑目标网络更新

目标: reward收敛至 180+ (CartPole-v1 最大500)
"""

import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import copy

# ========== 配置 ==========
BASE_DIR = Path(__file__).parent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# 固定随机种子以提高可复现性
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

# 超参数
ENV_NAME = "CartPole-v1"
GAMMA = 0.99                # 折扣因子
LEARNING_RATE = 0.0005      # 学习率
BATCH_SIZE = 128            # 批大小
REPLAY_BUFFER_SIZE = 100000 # 经验回放缓冲区大小
TAU = 0.005                 # 软更新系数 (Polyak averaging)
EPSILON_START = 1.0         # 初始探索率
EPSILON_END = 0.02          # 最终探索率
EPSILON_DECAY_STEPS = 4000  # epsilon线性衰减总步数
NUM_EPISODES = 800          # 训练回合数（足够长以稳定收敛）
MIN_REPLAY_SIZE = 2000      # 开始训练前的最小经验数量
HIDDEN_DIM = 128            # 隐藏层维度


# ========== Q-Network ==========
class QNetwork(nn.Module):
    """
    Q值网络
    输入: state (4维: 位置, 速度, 角度, 角速度)
    输出: 每个动作的Q值 (2维: 左推, 右推)
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


# ========== Experience Replay Buffer ==========
class ReplayBuffer:
    """经验回放缓冲区，存储 (state, action, reward, next_state, done) 转移"""

    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.FloatTensor(np.array(states)).to(DEVICE),
            torch.LongTensor(np.array(actions)).to(DEVICE),
            torch.FloatTensor(np.array(rewards)).to(DEVICE),
            torch.FloatTensor(np.array(next_states)).to(DEVICE),
            torch.FloatTensor(np.array(dones)).to(DEVICE),
        )

    def __len__(self):
        return len(self.buffer)


# ========== DQN Agent (Double DQN + Soft Target Update) ==========
class DQNAgent:
    """DQN智能体，使用 Double DQN 和软更新"""

    def __init__(self, state_dim: int, action_dim: int):
        self.action_dim = action_dim
        self.total_steps = 0

        self.q_network = QNetwork(state_dim, action_dim, HIDDEN_DIM).to(DEVICE)
        self.target_network = QNetwork(state_dim, action_dim, HIDDEN_DIM).to(DEVICE)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=LEARNING_RATE)
        self.replay_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE)

    @property
    def epsilon(self) -> float:
        """线性衰减探索率"""
        if self.total_steps >= EPSILON_DECAY_STEPS:
            return EPSILON_END
        return EPSILON_START - (EPSILON_START - EPSILON_END) * (
            self.total_steps / EPSILON_DECAY_STEPS
        )

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """ε-greedy动作选择"""
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_dim)

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
            q_values = self.q_network(state_tensor)
            return q_values.argmax(dim=1).item()

    def update(self) -> float:
        """Double DQN 更新 + 软更新目标网络"""
        if len(self.replay_buffer) < MIN_REPLAY_SIZE:
            return 0.0

        states, actions, rewards, next_states, dones = \
            self.replay_buffer.sample(BATCH_SIZE)

        # Double DQN: Q网络选动作，目标网络评估
        with torch.no_grad():
            next_q_online = self.q_network(next_states)
            best_actions = next_q_online.argmax(dim=1)
            next_q_target = self.target_network(next_states)
            max_next_q = next_q_target.gather(1, best_actions.unsqueeze(1)).squeeze(1)
            target = rewards + GAMMA * max_next_q * (1 - dones)

        q_values = self.q_network(states)
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        loss = nn.SmoothL1Loss()(q_values, target)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()

        self._soft_update_target()
        return loss.item()

    def _soft_update_target(self):
        """θ_target = τ * θ_online + (1 - τ) * θ_target"""
        for target_param, online_param in zip(
            self.target_network.parameters(), self.q_network.parameters()
        ):
            target_param.data.copy_(
                TAU * online_param.data + (1.0 - TAU) * target_param.data
            )

    def save(self, path: str):
        torch.save(self.q_network.state_dict(), path)

    def load(self, path: str):
        self.q_network.load_state_dict(torch.load(path, weights_only=True))
        self.target_network.load_state_dict(self.q_network.state_dict())


# ========== 快速测试（用于模型选择） ==========
def quick_test(agent: DQNAgent, num_episodes: int = 5) -> float:
    """用 ε=0 快速测试当前模型，返回平均 reward"""
    env = gym.make(ENV_NAME)
    rewards = []
    for _ in range(num_episodes):
        state, _ = env.reset()
        total = 0
        done = False
        while not done:
            action = agent.select_action(state, training=False)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state = next_state
            total += reward
        rewards.append(total)
    env.close()
    return np.mean(rewards)


# ========== 训练循环 ==========
def train_dqn():
    """DQN训练主循环，定期测试选最佳模型"""
    env = gym.make(ENV_NAME)
    state_dim = env.observation_space.shape[0]   # 4
    action_dim = env.action_space.n               # 2

    agent = DQNAgent(state_dim, action_dim)

    episode_rewards = []        # 每局总奖励
    moving_avg_rewards = []     # 滑动平均 (100 eps)
    test_scores = []            # 记录定期测试分数

    best_test_score = 0.0
    best_model_state = None
    best_episode = 0

    for episode in range(1, NUM_EPISODES + 1):
        state, _ = env.reset()
        episode_reward = 0
        done = False

        while not done:
            action = agent.select_action(state, training=True)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.replay_buffer.push(state, action, reward, next_state, float(done))
            agent.total_steps += 1
            agent.update()
            state = next_state
            episode_reward += reward

        episode_rewards.append(episode_reward)

        moving_avg = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 \
            else np.mean(episode_rewards)
        moving_avg_rewards.append(moving_avg)

        # 每50回合做一次测试，选最佳模型
        if episode % 50 == 0 or episode == 1:
            test_score = quick_test(agent, num_episodes=5)
            test_scores.append((episode, test_score))
            if test_score > best_test_score:
                best_test_score = test_score
                best_model_state = copy.deepcopy(agent.q_network.state_dict())
                best_episode = episode
            print(f"Episode {episode:3d}/{NUM_EPISODES} | "
                  f"Reward: {episode_reward:4.0f} | "
                  f"Avg100: {moving_avg:6.2f} | "
                  f"Test: {test_score:6.1f} | "
                  f"BestTest: {best_test_score:6.1f} | "
                  f"Epsilon: {agent.epsilon:.4f}")

    env.close()

    # 加载最佳模型
    if best_model_state is not None:
        agent.q_network.load_state_dict(best_model_state)
        agent.target_network.load_state_dict(best_model_state)
        print(f"\n加载最佳模型 (Ep{best_episode}, TestScore={best_test_score:.1f})")

    agent.save(str(BASE_DIR / "dqn_model.pth"))
    print(f"模型已保存至: {BASE_DIR / 'dqn_model.pth'}")

    return episode_rewards, moving_avg_rewards, test_scores, agent


# ========== 完整测试 ==========
def test_dqn(agent: DQNAgent, num_episodes: int = 20):
    """测试训练好的DQN智能体"""
    env = gym.make(ENV_NAME)
    test_rewards = []

    for episode in range(1, num_episodes + 1):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        while not done:
            action = agent.select_action(state, training=False)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state = next_state
            episode_reward += reward
        test_rewards.append(episode_reward)
        print(f"Test Episode {episode:2d}: Reward = {episode_reward}")

    env.close()
    avg_reward = np.mean(test_rewards)
    std_reward = np.std(test_rewards)
    print(f"\n测试结果: 平均Reward = {avg_reward:.2f} ± {std_reward:.2f}")
    print(f"最高: {np.max(test_rewards):.0f}, 最低: {np.min(test_rewards):.0f}")
    return test_rewards


# ========== 绘图 ==========
def plot_results(episode_rewards: list, moving_avg_rewards: list,
                 test_scores: list):
    """绘制训练曲线，标注最佳模型位置"""
    fig, ax = plt.subplots(figsize=(12, 5))

    episodes = range(1, len(episode_rewards) + 1)

    # 每局 Reward（半透明散点）
    ax.scatter(episodes, episode_rewards, alpha=0.2, color='steelblue',
               s=8, edgecolors='none', label='Episode Reward')

    # 滑动平均
    ax.plot(episodes, moving_avg_rewards, color='crimson', linewidth=2,
            label='Moving Average (100 eps)')

    # 定期测试分数（大号标记）
    if test_scores:
        test_eps, test_vals = zip(*test_scores)
        ax.scatter(test_eps, test_vals, color='darkgreen', s=60,
                   marker='D', zorder=5, label='Validation Score (ε=0, 5 eps)')

    # 目标线
    ax.axhline(y=180, color='green', linestyle='--', linewidth=1.5,
               label='Target (180)')
    ax.axhline(y=500, color='gray', linestyle=':', linewidth=1,
               label='Env Max (500)')

    # 标注最佳模型
    best_idx = np.argmax(moving_avg_rewards)
    best_avg = moving_avg_rewards[best_idx]
    ax.axvline(x=best_idx + 1, color='orange', linestyle='--', linewidth=1.2,
               alpha=0.7)
    ax.annotate(f'Best Checkpoint\nEp {best_idx + 1}, Avg100={best_avg:.0f}',
                xy=(best_idx + 1, best_avg),
                xytext=(best_idx + 1 + 50, best_avg - 60),
                fontsize=9, color='darkorange',
                arrowprops=dict(arrowstyle='->', color='darkorange', alpha=0.7))

    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Total Reward', fontsize=12)
    ax.set_title('DQN Training on CartPole-v1 (Double DQN + Soft Target Update)',
                 fontsize=13)
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, len(episode_rewards) + 5)

    plt.tight_layout()
    plt.savefig(str(BASE_DIR / "training_results.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"训练曲线已保存至: {BASE_DIR / 'training_results.png'}")


# ========== 主函数 ==========
def main():
    print("=" * 60)
    print("DQN on CartPole-v1 (Double DQN + Soft Target Update)")
    print(f"目标: 收敛至 Reward >= 180, Seed={SEED}")
    print("=" * 60)

    episode_rewards, moving_avg_rewards, test_scores, agent = train_dqn()
    plot_results(episode_rewards, moving_avg_rewards, test_scores)

    # 结果统计
    final_avg_100 = np.mean(episode_rewards[-100:])
    max_reward = np.max(episode_rewards)
    print(f"\n{'=' * 60}")
    print("训练结果汇总")
    print(f"{'=' * 60}")
    print(f"最近100局平均Reward: {final_avg_100:.2f}")
    print(f"最高单局Reward:       {max_reward:.0f}")
    print(f"总训练步数:           {agent.total_steps}")

    converged_ep = next(
        (i + 1 for i, r in enumerate(moving_avg_rewards) if r >= 180), None)
    if converged_ep is not None:
        print(f"✓ 收敛至 180+ 于第 {converged_ep} 回合")
    else:
        print(f"✗ 未达180收敛目标 (最佳Avg100: {max(moving_avg_rewards):.2f})")

    # 最终测试
    print(f"\n{'=' * 60}")
    print("测试阶段 (20 episodes, ε=0)")
    print(f"{'=' * 60}")
    test_rewards = test_dqn(agent, num_episodes=20)

    test_avg = np.mean(test_rewards)
    print(f"\n{'=' * 60}")
    if test_avg >= 180:
        print(f"✓ 达标! 测试平均Reward: {test_avg:.2f} >= 180")
    else:
        print(f"✗ 未完全达标. 测试平均Reward: {test_avg:.2f} < 180")


if __name__ == "__main__":
    main()
