"""
HW3-4 (Bonus): Rainbow DQN for Random Mode
Based on the Gridworld environment from:
https://github.com/DeepReinforcementLearning/DeepReinforcementLearningInAction/tree/master/Chapter%203

Rainbow DQN integrates the following improvements over vanilla DQN:
  1. Double DQN          - Decouples action selection and evaluation
  2. Dueling DQN         - Separates Value and Advantage streams
  3. Prioritized Replay  - Samples important transitions more frequently
  4. Multi-step Returns  - Propagates rewards faster via n-step bootstrapping
  5. Noisy Networks      - Replaces epsilon-greedy with parametric noise for exploration
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import math
from collections import deque
import matplotlib.pyplot as plt
import pandas as pd

from Gridworld import Gridworld

# ============================================================
# 1. Helper
# ============================================================
action_set = {0: 'u', 1: 'd', 2: 'l', 3: 'r'}

def get_state(game):
    state = game.board.render_np().reshape(1, 64) + np.random.rand(1, 64) / 100.0
    return state.flatten().astype(np.float32)

# ============================================================
# 2. Noisy Linear Layer
#    Replaces epsilon-greedy with learned exploration noise.
#    The network learns WHEN and WHERE to explore.
# ============================================================
class NoisyLinear(nn.Module):
    """
    Factorized Gaussian Noisy Linear layer (Fortunato et al., 2018).
    Instead of epsilon-greedy, exploration is driven by learnable noise
    parameters that the network can adjust during training.
    """
    def __init__(self, in_features, out_features, sigma_init=0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Learnable parameters
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))

        # Factorized noise buffers (not learnable)
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))

        self.sigma_init = sigma_init
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        bound = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-bound, bound)
        self.bias_mu.data.uniform_(-bound, bound)
        self.weight_sigma.data.fill_(self.sigma_init / math.sqrt(self.in_features))
        self.bias_sigma.data.fill_(self.sigma_init / math.sqrt(self.in_features))

    @staticmethod
    def _scale_noise(size):
        x = torch.randn(size)
        return x.sign() * x.abs().sqrt()

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.outer(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x):
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)

# ============================================================
# 3. Prioritized Experience Replay Buffer
#    Samples transitions proportional to their TD-error.
#    High-error transitions are replayed more often.
# ============================================================
class PrioritizedReplayBuffer:
    """
    Sum-tree based Prioritized Experience Replay (Schaul et al., 2016).
    Transitions with higher TD-error get sampled more frequently,
    improving learning efficiency on difficult transitions.
    """
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha  # prioritization exponent (0 = uniform, 1 = full priority)
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.max_priority = 1.0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.priorities[self.position] = self.max_priority ** self.alpha
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        N = len(self.buffer)
        priorities = self.priorities[:N]
        probs = priorities / priorities.sum()

        indices = np.random.choice(N, batch_size, p=probs, replace=False)

        # Importance sampling weights to correct for biased sampling
        weights = (N * probs[indices]) ** (-beta)
        weights = weights / weights.max()  # normalize

        batch = [self.buffer[i] for i in indices]
        state, action, reward, next_state, done = map(np.stack, zip(*batch))

        return state, action, reward, next_state, done, indices, weights.astype(np.float32)

    def update_priorities(self, indices, td_errors):
        for idx, td_error in zip(indices, td_errors):
            priority = (abs(td_error) + 1e-6) ** self.alpha
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)

    def __len__(self):
        return len(self.buffer)

# ============================================================
# 4. Multi-step Return Buffer
#    Accumulates n-step returns for faster reward propagation.
# ============================================================
class NStepBuffer:
    """
    Stores the last n transitions and computes the n-step discounted return.
    Instead of R + gamma * Q(s'), we use R1 + gamma*R2 + ... + gamma^n * Q(s_n),
    which propagates reward signals much faster through the network.
    """
    def __init__(self, n_step=3, gamma=0.9):
        self.n_step = n_step
        self.gamma = gamma
        self.buffer = deque(maxlen=n_step)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def get(self):
        """Compute the n-step return and return the first state + last next_state."""
        state, action = self.buffer[0][0], self.buffer[0][1]
        n_step_reward = 0
        for i, (_, _, r, _, d) in enumerate(self.buffer):
            n_step_reward += (self.gamma ** i) * r
            if d:
                break
        last_next_state = self.buffer[-1][3]
        last_done = self.buffer[-1][4]
        return state, action, n_step_reward, last_next_state, last_done

    def is_ready(self):
        return len(self.buffer) == self.n_step

    def reset(self):
        self.buffer.clear()

# ============================================================
# 5. Rainbow DQN Network (Dueling + Noisy)
# ============================================================
class RainbowNet(nn.Module):
    """
    Combines Dueling architecture with NoisyLinear layers.
    - Dueling: Q(s,a) = V(s) + A(s,a) - mean(A)
    - Noisy: Exploration via learned noise (no epsilon-greedy needed)
    """
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(input_dim, 150),
            nn.ReLU()
        )
        # Value stream with noisy layers
        self.value_stream = nn.Sequential(
            NoisyLinear(150, 64),
            nn.ReLU(),
            NoisyLinear(64, 1)
        )
        # Advantage stream with noisy layers
        self.advantage_stream = nn.Sequential(
            NoisyLinear(150, 64),
            nn.ReLU(),
            NoisyLinear(64, output_dim)
        )

    def forward(self, x):
        features = self.feature(x)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        return value + (advantage - advantage.mean(dim=1, keepdim=True))

    def reset_noise(self):
        """Reset noise in all NoisyLinear layers."""
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()

# ============================================================
# 6. Rainbow Agent
# ============================================================
class RainbowAgent:
    def __init__(self, n_step=3, gamma=0.9, lr=1e-3, buffer_size=10000,
                 batch_size=64, sync_freq=200, beta_start=0.4, beta_frames=10000):
        self.gamma = gamma
        self.n_step = n_step
        self.batch_size = batch_size
        self.sync_freq = sync_freq

        # Networks
        self.model = RainbowNet(64, 4)
        self.target_model = RainbowNet(64, 4)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=2000, gamma=0.9)

        # Buffers
        self.replay_buffer = PrioritizedReplayBuffer(buffer_size)
        self.n_step_buffer = NStepBuffer(n_step, gamma)

        # Beta annealing for importance sampling
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame_count = 0

        self.rewards_log = []
        self.loss_log = []

    def get_beta(self):
        """Linearly anneal beta from beta_start to 1.0."""
        return min(1.0, self.beta_start + self.frame_count * (1.0 - self.beta_start) / self.beta_frames)

    def select_action(self, state):
        """No epsilon-greedy! Exploration is handled by NoisyNet."""
        state_t = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state_t)
        return q_values.argmax().item()

    def train_one_episode(self):
        game = Gridworld(size=4, mode='random')
        state = get_state(game)
        total_reward = 0
        done = False
        steps = 0
        self.n_step_buffer.reset()

        while not done:
            self.frame_count += 1
            steps += 1

            # Noisy network handles exploration automatically
            action = self.select_action(state)

            game.makeMove(action_set[action])
            next_state = get_state(game)
            reward = game.reward()

            if reward != -1:
                done = True
            elif steps >= 50:
                done = True

            # Push to n-step buffer
            self.n_step_buffer.push(state, action, reward, next_state, float(done))

            # When n-step buffer is full, compute n-step return and store
            if self.n_step_buffer.is_ready():
                s, a, r, ns, d = self.n_step_buffer.get()
                self.replay_buffer.push(s, a, r, ns, d)

            # If episode ends, flush remaining transitions
            if done:
                while len(self.n_step_buffer.buffer) > 0:
                    s, a, r, ns, d = self.n_step_buffer.get()
                    self.replay_buffer.push(s, a, r, ns, d)
                    self.n_step_buffer.buffer.popleft()

            state = next_state
            total_reward += reward

            # Train
            if len(self.replay_buffer) >= self.batch_size:
                self._update()

            # Sync target network
            if self.frame_count % self.sync_freq == 0:
                self.target_model.load_state_dict(self.model.state_dict())

        self.rewards_log.append(total_reward)
        return total_reward

    def _update(self):
        beta = self.get_beta()
        s, a, r, ns, d, indices, weights = self.replay_buffer.sample(self.batch_size, beta)

        s = torch.FloatTensor(s)
        a = torch.LongTensor(a).unsqueeze(1)
        r = torch.FloatTensor(r).unsqueeze(1)
        ns = torch.FloatTensor(ns)
        d = torch.FloatTensor(d).unsqueeze(1)
        weights = torch.FloatTensor(weights).unsqueeze(1)

        # Current Q values
        q_values = self.model(s).gather(1, a)

        with torch.no_grad():
            # Double DQN: main selects, target evaluates
            best_actions = self.model(ns).argmax(1).unsqueeze(1)
            next_q = self.target_model(ns).gather(1, best_actions)
            target_q = r + (self.gamma ** self.n_step) * next_q * (1 - d)

        # Weighted MSE loss (importance sampling correction)
        td_errors = (q_values - target_q).detach().cpu().numpy().flatten()
        loss = (weights * F.mse_loss(q_values, target_q, reduction='none')).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        self.scheduler.step()

        # Update priorities
        self.replay_buffer.update_priorities(indices, td_errors)

        # Reset noise after each update
        self.model.reset_noise()
        self.target_model.reset_noise()

        self.loss_log.append(loss.item())

# ============================================================
# 7. Training, Plotting, and Testing
# ============================================================
def smooth(data, window=50):
    return pd.Series(data).rolling(window=window).mean()

def run_tests(agent, num_tests=20):
    print(f"\n--- Running {num_tests} Tests in Random Mode (Rainbow DQN) ---")
    agent.model.eval()
    success_count = 0

    for i in range(1, num_tests + 1):
        game = Gridworld(size=4, mode='random')
        state = get_state(game)
        done = False
        steps = 0
        status = "Failed (Timeout)"

        while not done:
            state_t = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                q_vals = agent.model(state_t)
            action = q_vals.argmax().item()

            game.makeMove(action_set[action])
            next_state = get_state(game)
            reward = game.reward()
            state = next_state
            steps += 1

            if reward == 10:
                status = "Success"
                success_count += 1
                done = True
            elif reward == -10:
                status = "Failed (Hit Pit)"
                done = True
            elif steps >= 50:
                done = True

        print(f"Test {i:2d} | Steps: {steps:2d} | Status: {status}")

    rate = success_count / num_tests * 100
    print(f"\nOverall Success Rate: {success_count}/{num_tests} ({rate:.1f}%)")
    return success_count, num_tests

if __name__ == '__main__':
    print("\n=== HW3-4 (Bonus): Rainbow DQN for Random Mode ===\n")

    agent = RainbowAgent(
        n_step=3,
        gamma=0.9,
        lr=1e-3,
        buffer_size=10000,
        batch_size=64,
        sync_freq=200,
        beta_start=0.4,
        beta_frames=15000
    )

    episodes = 1000
    print(f"Training for {episodes} episodes...")
    for ep in range(1, episodes + 1):
        reward = agent.train_one_episode()
        if ep % 50 == 0:
            avg = np.mean(agent.rewards_log[-50:])
            print(f"Episode {ep:4d} | Avg Reward (last 50): {avg:7.2f}")

    # --- Plot 1: Reward curve ---
    plt.figure(figsize=(10, 5))
    plt.plot(smooth(agent.rewards_log, window=50), color='#8b5cf6', alpha=0.9, linewidth=2)
    plt.title('Rainbow DQN - Average Reward (Random Mode)')
    plt.xlabel('Episodes')
    plt.ylabel('Reward (Smoothed)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig('hw3_4_rainbow_reward.png')
    plt.close()
    print("\nSaved hw3_4_rainbow_reward.png")

    # --- Plot 2: Loss curve ---
    plt.figure(figsize=(10, 5))
    plt.plot(smooth(agent.loss_log, window=100), color='#f43f5e', alpha=0.9, linewidth=2)
    plt.title('Rainbow DQN - Training Loss (Random Mode)')
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig('hw3_4_rainbow_loss.png')
    plt.close()
    print("Saved hw3_4_rainbow_loss.png")

    # --- Test ---
    success, total = run_tests(agent, num_tests=20)
