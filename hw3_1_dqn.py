import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt
import pandas as pd

# 1. Environment Definition (Static Mode)
class StaticGridEnv:
    def __init__(self):
        self.rows = 4
        self.cols = 4
        self.start = (0, 3)
        self.goal = (0, 0)
        self.pit = (0, 1)
        self.wall = (1, 1)
        self.reset()

    def reset(self):
        self.player = list(self.start)
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

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
    def forward(self, x):
        return self.net(x)

def train_agent(use_replay=True, episodes=500):
    env = StaticGridEnv()
    model = DQN(16, 4)
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    loss_fn = nn.MSELoss()
    
    buffer = ReplayBuffer(2000)
    batch_size = 32 if use_replay else 1
    gamma = 0.99
    epsilon = 1.0
    epsilon_min = 0.01
    epsilon_decay = 0.99
    
    loss_history = []
    print(f"\nStarting Training (Static Mode) | Replay Buffer: {use_replay}")
    
    for episode in range(1, episodes + 1):
        state = env.reset()
        done = False
        
        while not done:
            if random.random() < epsilon:
                action = random.randint(0, 3)
            else:
                state_t = torch.FloatTensor(state).unsqueeze(0)
                with torch.no_grad():
                    q_values = model(state_t)
                action = q_values.argmax().item()
                
            next_state, reward, done = env.step(action)
            
            if use_replay:
                buffer.push(state, action, reward, next_state, done)
                if len(buffer) >= batch_size:
                    s, a, r, ns, d = buffer.sample(batch_size)
                    s = torch.FloatTensor(s)
                    a = torch.LongTensor(a).unsqueeze(1)
                    r = torch.FloatTensor(r).unsqueeze(1)
                    ns = torch.FloatTensor(ns)
                    d = torch.FloatTensor(d).unsqueeze(1)
                    
                    q_values = model(s).gather(1, a)
                    with torch.no_grad():
                        next_q_values = model(ns).max(1)[0].unsqueeze(1)
                        target_q = r + gamma * next_q_values * (1 - d)
                        
                    loss = loss_fn(q_values, target_q)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    loss_history.append(loss.item())
            else:
                # Naive Online Learning (batch size 1 directly from transition)
                s = torch.FloatTensor(state).unsqueeze(0)
                a = torch.LongTensor([[action]])
                r = torch.FloatTensor([[reward]])
                ns = torch.FloatTensor(next_state).unsqueeze(0)
                d = torch.FloatTensor([[done]])
                
                q_values = model(s).gather(1, a)
                with torch.no_grad():
                    next_q_values = model(ns).max(1)[0].unsqueeze(1)
                    target_q = r + gamma * next_q_values * (1 - d)
                    
                loss = loss_fn(q_values, target_q)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_history.append(loss.item())
                
            state = next_state
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        
    return loss_history

def smooth(data, window=50):
    return pd.Series(data).rolling(window=window).mean()

if __name__ == '__main__':
    # 1. Train Naive DQN (No Replay Buffer)
    naive_loss = train_agent(use_replay=False, episodes=500)
    plt.figure(figsize=(8, 5))
    plt.plot(smooth(naive_loss), color='orange', alpha=0.8)
    plt.title('Naive DQN Training Loss (Static Mode)')
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig('naive_dqn_loss.png')
    plt.close()
    
    # 2. Train ER DQN (With Replay Buffer)
    er_loss = train_agent(use_replay=True, episodes=500)
    plt.figure(figsize=(8, 5))
    plt.plot(smooth(er_loss), color='blue', alpha=0.8)
    plt.title('Experience Replay DQN Training Loss (Static Mode)')
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig('er_dqn_loss.png')
    plt.close()
    
    print("Saved naive_dqn_loss.png and er_dqn_loss.png")
