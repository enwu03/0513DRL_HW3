import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

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
        # 16-dim one-hot vector representing player position
        state = np.zeros(self.rows * self.cols, dtype=np.float32)
        idx = self.player[0] * self.cols + self.player[1]
        state[idx] = 1.0
        return state

    def step(self, action):
        self.steps += 1
        # Actions: 0: Up, 1: Down, 2: Left, 3: Right
        dx = [-1, 1, 0, 0]
        dy = [0, 0, -1, 1]
        
        new_x = self.player[0] + dx[action]
        new_y = self.player[1] + dy[action]
        
        # Check boundaries
        if new_x < 0 or new_x >= self.rows or new_y < 0 or new_y >= self.cols:
            new_x, new_y = self.player[0], self.player[1]
            
        # Check wall
        if (new_x, new_y) == self.wall:
            new_x, new_y = self.player[0], self.player[1]
            
        self.player = [new_x, new_y]
        state = self._get_state()
        
        # Rewards and Done
        if tuple(self.player) == self.goal:
            return state, 10.0, True
        elif tuple(self.player) == self.pit:
            return state, -10.0, True
        elif self.steps >= 50: # Timeout limit to prevent infinite loops
            return state, -1.0, True
        else:
            return state, -1.0, False # Step penalty

# 2. Experience Replay Buffer
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

# 3. DQN Network
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

# 4. Main Training Loop
def train():
    env = StaticGridEnv()
    input_dim = 16
    output_dim = 4
    
    model = DQN(input_dim, output_dim)
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    loss_fn = nn.MSELoss()
    
    buffer = ReplayBuffer(2000)
    batch_size = 32
    gamma = 0.99
    epsilon = 1.0
    epsilon_min = 0.01
    epsilon_decay = 0.99
    
    episodes = 1000
    rewards_log = []
    
    print("Starting DQN Training (Static Mode)...")
    
    for episode in range(1, episodes + 1):
        state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            # Epsilon-Greedy
            if random.random() < epsilon:
                action = random.randint(0, 3)
            else:
                state_t = torch.FloatTensor(state).unsqueeze(0)
                with torch.no_grad():
                    q_values = model(state_t)
                action = q_values.argmax().item()
                
            next_state, reward, done = env.step(action)
            buffer.push(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            
            # Train model
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
                
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        rewards_log.append(total_reward)
        
        if episode % 100 == 0:
            avg_reward = np.mean(rewards_log[-100:])
            print(f"Episode {episode:4d} | Avg Reward (last 100): {avg_reward:7.2f} | Epsilon: {epsilon:.2f}")

if __name__ == '__main__':
    train()
