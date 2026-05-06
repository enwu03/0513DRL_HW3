"""
HW3-1: Naive DQN for Static Mode
Based on the Gridworld environment from:
https://github.com/DeepReinforcementLearning/DeepReinforcementLearningInAction/tree/master/Chapter%203

This script implements both Naive (online) DQN and Experience Replay DQN
using the original Gridworld.py and GridBoard.py from the textbook repo.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt
import pandas as pd
import copy

# Import the original Gridworld environment from the DRL in Action repo
from Gridworld import Gridworld

# ============================================================
# 1. Experience Replay Buffer
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
# 2. DQN Neural Network
# ============================================================
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 150),
            nn.ReLU(),
            nn.Linear(150, 100),
            nn.ReLU(),
            nn.Linear(100, output_dim)
        )
    def forward(self, x):
        return self.net(x)

# ============================================================
# 3. Helper: get state from the original Gridworld object
# ============================================================
action_set = {0: 'u', 1: 'd', 2: 'l', 3: 'r'}

def get_state(game):
    """
    Extract the state from the Gridworld object as a flattened numpy array.
    Uses board.render_np() which returns a 4-channel (Player, Goal, Pit, Wall) 
    representation, matching the textbook's approach.
    """
    state = game.board.render_np().reshape(1, 64) + np.random.rand(1, 64) / 100.0
    return state.flatten().astype(np.float32)

# ============================================================
# 4. Training function
# ============================================================
def train_agent(use_replay=True, episodes=500):
    gamma = 0.9
    epsilon = 1.0
    epsilon_min = 0.01
    epsilon_decay = 0.99
    
    model = DQN(64, 4)
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    buffer = ReplayBuffer(1000)
    batch_size = 200 if use_replay else 1
    
    loss_history = []
    mode_name = "Experience Replay" if use_replay else "Naive (Online)"
    print(f"\nStarting Training (Static Mode) | Mode: {mode_name}")
    
    for episode in range(1, episodes + 1):
        game = Gridworld(size=4, mode='static')
        state = get_state(game)
        done = False
        steps = 0
        
        while not done:
            steps += 1
            # Epsilon-greedy action selection
            if random.random() < epsilon:
                action = random.randint(0, 3)
            else:
                state_t = torch.FloatTensor(state).unsqueeze(0)
                with torch.no_grad():
                    q_values = model(state_t)
                action = q_values.argmax().item()
            
            # Take action using the original Gridworld's makeMove
            game.makeMove(action_set[action])
            next_state = get_state(game)
            reward = game.reward()
            
            # Check terminal conditions
            if reward != -1:  # Hit goal (+10) or pit (-10)
                done = True
            elif steps >= 50:  # Timeout
                done = True
            
            if use_replay:
                # --- Experience Replay Mode ---
                buffer.push(state, action, reward, next_state, float(done))
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
                # --- Naive (Online) Mode ---
                s = torch.FloatTensor(state).unsqueeze(0)
                a = torch.LongTensor([[action]])
                r_t = torch.FloatTensor([[reward]])
                ns = torch.FloatTensor(next_state).unsqueeze(0)
                d_t = torch.FloatTensor([[float(done)]])
                
                q_values = model(s).gather(1, a)
                with torch.no_grad():
                    next_q_values = model(ns).max(1)[0].unsqueeze(1)
                    target_q = r_t + gamma * next_q_values * (1 - d_t)
                
                loss = loss_fn(q_values, target_q)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_history.append(loss.item())
            
            state = next_state
            
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        
    return loss_history

# ============================================================
# 5. Smoothing and Plotting
# ============================================================
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
    plt.savefig('hw3_1_naive_dqn_loss.png')
    plt.close()
    
    # 2. Train ER DQN (With Replay Buffer)
    er_loss = train_agent(use_replay=True, episodes=500)
    plt.figure(figsize=(8, 5))
    plt.plot(smooth(er_loss), color='blue', alpha=0.8)
    plt.title('Experience Replay DQN Training Loss (Static Mode)')
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig('hw3_1_er_dqn_loss.png')
    plt.close()
    
    print("\nSaved hw3_1_naive_dqn_loss.png and hw3_1_er_dqn_loss.png")
