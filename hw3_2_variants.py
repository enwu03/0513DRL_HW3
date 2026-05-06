import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt

# 1. Environment Definition (Player Mode - Random Start)
class PlayerGridEnv:
    def __init__(self):
        self.rows = 4
        self.cols = 4
        self.goal = (0, 0)
        self.pit = (0, 1)
        self.wall = (1, 1)
        self.reset()

    def reset(self):
        # Randomize player start position, avoiding goal, pit, and wall
        while True:
            r = random.randint(0, self.rows - 1)
            c = random.randint(0, self.cols - 1)
            if (r, c) not in [self.goal, self.pit, self.wall]:
                self.player = [r, c]
                break
        self.steps = 0
        return self._get_state()

    def _get_state(self):
        state = np.zeros(self.rows * self.cols, dtype=np.float32)
        idx = self.player[0] * self.cols + self.player[1]
        state[idx] = 1.0
        return state

    def step(self, action):
        self.steps += 1
        dx = [-1, 1, 0, 0]
        dy = [0, 0, -1, 1]
        
        new_x = self.player[0] + dx[action]
        new_y = self.player[1] + dy[action]
        
        if new_x < 0 or new_x >= self.rows or new_y < 0 or new_y >= self.cols:
            new_x, new_y = self.player[0], self.player[1]
        if (new_x, new_y) == self.wall:
            new_x, new_y = self.player[0], self.player[1]
            
        self.player = [new_x, new_y]
        state = self._get_state()
        
        if tuple(self.player) == self.goal:
            return state, 10.0, True
        elif tuple(self.player) == self.pit:
            return state, -10.0, True
        elif self.steps >= 50:
            return state, -1.0, True
        else:
            return state, -1.0, False

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

# 2. Networks
class StandardNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(StandardNet, self).__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, 64), nn.ReLU(), nn.Linear(64, output_dim))
    def forward(self, x): return self.net(x)

class DuelingNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DuelingNet, self).__init__()
        self.feature_layer = nn.Sequential(nn.Linear(input_dim, 64), nn.ReLU())
        self.value_stream = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))
        self.advantage_stream = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, output_dim))
    def forward(self, x):
        features = self.feature_layer(x)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        return value + (advantage - advantage.mean(dim=1, keepdim=True))

# 3. Unified Agent Class
class Agent:
    def __init__(self, variant_name):
        self.env = PlayerGridEnv()
        self.variant_name = variant_name
        self.use_double = (variant_name == "Double DQN")
        
        if self.use_double:
            self.model = StandardNet(16, 4)
            self.target_model = StandardNet(16, 4)
        else:
            self.model = DuelingNet(16, 4)
            self.target_model = DuelingNet(16, 4)
            
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.005)
        self.loss_fn = nn.MSELoss()
        self.buffer = ReplayBuffer(2000)
        
        self.batch_size = 32
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99
        self.sync_freq = 50
        self.global_step = 0
        self.rewards_log = []

    def train_one_episode(self):
        state = self.env.reset()
        total_reward = 0
        done = False
        
        while not done:
            self.global_step += 1
            if random.random() < self.epsilon:
                action = random.randint(0, 3)
            else:
                state_t = torch.FloatTensor(state).unsqueeze(0)
                with torch.no_grad():
                    q_values = self.model(state_t)
                action = q_values.argmax().item()
                
            next_state, reward, done = self.env.step(action)
            self.buffer.push(state, action, reward, next_state, done)
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
                        best_actions = self.model(ns).argmax(1).unsqueeze(1)
                        next_q_values = self.target_model(ns).gather(1, best_actions)
                    else:
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

# 4. Simultaneous Training and Plotting
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
