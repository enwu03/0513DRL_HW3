"""
HW3-2: Enhanced DQN Variants for Player Mode
Based on the Gridworld environment from:
https://github.com/DeepReinforcementLearning/DeepReinforcementLearningInAction/tree/master/Chapter%203

Implements Double DQN and Dueling DQN using the original Gridworld.py.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt

# Import the original Gridworld environment from the DRL in Action repo
from Gridworld import Gridworld

# ============================================================
# 1. Replay Buffer
# ============================================================
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done
    def __len__(self):
        return len(self.buffer)

# ============================================================
# 2. Network Architectures
# ============================================================

# Standard network for Double DQN
class StandardNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(StandardNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 150),
            nn.ReLU(),
            nn.Linear(150, 100),
            nn.ReLU(),
            nn.Linear(100, output_dim)
        )
    def forward(self, x):
        return self.net(x)

# Dueling network: splits into Value stream and Advantage stream
class DuelingNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DuelingNet, self).__init__()
        self.feature_layer = nn.Sequential(nn.Linear(input_dim, 150), nn.ReLU())
        # Value stream: estimates V(s)
        self.value_stream = nn.Sequential(nn.Linear(150, 64), nn.ReLU(), nn.Linear(64, 1))
        # Advantage stream: estimates A(s, a)
        self.advantage_stream = nn.Sequential(nn.Linear(150, 64), nn.ReLU(), nn.Linear(64, output_dim))

    def forward(self, x):
        features = self.feature_layer(x)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        # Q(s,a) = V(s) + (A(s,a) - mean(A))
        return value + (advantage - advantage.mean(dim=1, keepdim=True))

# ============================================================
# 3. Helper
# ============================================================
action_set = {0: 'u', 1: 'd', 2: 'l', 3: 'r'}

def get_state(game):
    state = game.board.render_np().reshape(1, 64) + np.random.rand(1, 64) / 100.0
    return state.flatten().astype(np.float32)

# ============================================================
# 4. Unified Agent Class
# ============================================================
class Agent:
    def __init__(self, variant_name):
        self.variant_name = variant_name
        self.use_double = (variant_name == "Double DQN")

        if self.use_double:
            self.model = StandardNet(64, 4)
            self.target_model = StandardNet(64, 4)
        else:
            self.model = DuelingNet(64, 4)
            self.target_model = DuelingNet(64, 4)

        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.loss_fn = nn.MSELoss()
        self.buffer = ReplayBuffer(1000)

        self.batch_size = 200
        self.gamma = 0.9
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99
        self.sync_freq = 50
        self.global_step = 0
        self.rewards_log = []

    def train_one_episode(self):
        # Use the original Gridworld with 'player' mode (random start, fixed objects)
        game = Gridworld(size=4, mode='player')
        state = get_state(game)
        total_reward = 0
        done = False
        steps = 0

        while not done:
            self.global_step += 1
            steps += 1

            if random.random() < self.epsilon:
                action = random.randint(0, 3)
            else:
                state_t = torch.FloatTensor(state).unsqueeze(0)
                with torch.no_grad():
                    q_values = self.model(state_t)
                action = q_values.argmax().item()

            game.makeMove(action_set[action])
            next_state = get_state(game)
            reward = game.reward()

            if reward != -1:
                done = True
            elif steps >= 50:
                done = True

            self.buffer.push(state, action, reward, next_state, float(done))
            state = next_state
            total_reward += reward

            if len(self.buffer) >= self.batch_size:
                s, a, r, ns, d = self.buffer.sample(self.batch_size)
                s = torch.FloatTensor(s)
                a = torch.LongTensor(a).unsqueeze(1)
                r = torch.FloatTensor(r).unsqueeze(1)
                ns = torch.FloatTensor(ns)
                d = torch.FloatTensor(d).unsqueeze(1)

                q_values = self.model(s).gather(1, a)
                with torch.no_grad():
                    if self.use_double:
                        # Double DQN: Main network selects action, Target network evaluates
                        best_actions = self.model(ns).argmax(1).unsqueeze(1)
                        next_q_values = self.target_model(ns).gather(1, best_actions)
                    else:
                        # Dueling DQN: standard target evaluation
                        next_q_values = self.target_model(ns).max(1)[0].unsqueeze(1)

                    target_q = r + self.gamma * next_q_values * (1 - d)

                loss = self.loss_fn(q_values, target_q)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if self.global_step % self.sync_freq == 0:
                    self.target_model.load_state_dict(self.model.state_dict())

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.rewards_log.append(total_reward)
        return total_reward

# ============================================================
# 5. Simultaneous Training and Plotting
# ============================================================
if __name__ == '__main__':
    episodes = 500
    agent_double = Agent("Double DQN")
    agent_dueling = Agent("Dueling DQN")

    double_avg_rewards = []
    dueling_avg_rewards = []

    print("\n--- Starting Simultaneous Training (Player Mode) ---")
    for ep in range(1, episodes + 1):
        r_double = agent_double.train_one_episode()
        r_dueling = agent_dueling.train_one_episode()

        # Every 10 episodes, compute the average of the last 10
        if ep % 10 == 0:
            avg_double = np.mean(agent_double.rewards_log[-10:])
            avg_dueling = np.mean(agent_dueling.rewards_log[-10:])
            double_avg_rewards.append(avg_double)
            dueling_avg_rewards.append(avg_dueling)
            print(f"Episode {ep:3d} | Double DQN Reward: {avg_double:7.2f} | Dueling DQN Reward: {avg_dueling:7.2f}")

    # Save the plot
    plt.figure(figsize=(10, 6))
    x_axis = range(10, episodes + 1, 10)
    plt.plot(x_axis, double_avg_rewards, label='Double DQN', marker='o')
    plt.plot(x_axis, dueling_avg_rewards, label='Dueling DQN', marker='x')
    plt.title('Performance Comparison in Player Mode')
    plt.xlabel('Episodes')
    plt.ylabel('Average Reward (Last 10 Episodes)')
    plt.legend()
    plt.grid(True)
    plt.savefig('hw3_2_comparison.png')
    print("\nSaved comparison plot to hw3_2_comparison.png")
