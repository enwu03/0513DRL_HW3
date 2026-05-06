"""
HW3-3: Enhanced DQN for Random Mode with PyTorch Lightning
Based on the Gridworld environment from:
https://github.com/DeepReinforcementLearning/DeepReinforcementLearningInAction/tree/master/Chapter%203

Integrates: Gradient Clipping, LR Scheduling, AdamW, Large Replay Buffer.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
import numpy as np
import random
from collections import deque
import logging
import os
import pandas as pd
import matplotlib.pyplot as plt

# Import the original Gridworld environment from the DRL in Action repo
from Gridworld import Gridworld

logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)

# ============================================================
# 1. Helper
# ============================================================
action_set = {0: 'u', 1: 'd', 2: 'l', 3: 'r'}

def get_state(game):
    state = game.board.render_np().reshape(1, 64) + np.random.rand(1, 64) / 100.0
    return state.flatten().astype(np.float32)

# ============================================================
# 2. Replay Buffer & Dummy Dataset
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

class DummyDataset(Dataset):
    def __init__(self, size=15000): self.size = size
    def __len__(self): return self.size
    def __getitem__(self, idx): return torch.tensor(0)

# ============================================================
# 3. LitDQN Module (Dueling + Double architecture)
# ============================================================
class LitDQN(pl.LightningModule):
    def __init__(self, input_dim=64, output_dim=4, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()

        # Main network (Dueling architecture)
        self.feature_layer = nn.Sequential(nn.Linear(input_dim, 150), nn.ReLU())
        self.value_stream = nn.Sequential(nn.Linear(150, 64), nn.ReLU(), nn.Linear(64, 1))
        self.adv_stream = nn.Sequential(nn.Linear(150, 64), nn.ReLU(), nn.Linear(64, output_dim))

        # Target network
        self.target_feature_layer = nn.Sequential(nn.Linear(input_dim, 150), nn.ReLU())
        self.target_value_stream = nn.Sequential(nn.Linear(150, 64), nn.ReLU(), nn.Linear(64, 1))
        self.target_adv_stream = nn.Sequential(nn.Linear(150, 64), nn.ReLU(), nn.Linear(64, output_dim))

        self.sync_target()

        # Environment (random mode - all objects randomized)
        self.game = Gridworld(size=4, mode='random')
        self.state = get_state(self.game)
        self.buffer = ReplayBuffer(capacity=10000)

        self.batch_size = 64
        self.gamma = 0.9
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.998
        self.sync_freq = 200
        self.max_steps_per_episode = 50

        self.episode_reward = 0
        self.episode_count = 0
        self.episode_steps = 0
        self.loss_fn = nn.MSELoss()

    def sync_target(self):
        self.target_feature_layer.load_state_dict(self.feature_layer.state_dict())
        self.target_value_stream.load_state_dict(self.value_stream.state_dict())
        self.target_adv_stream.load_state_dict(self.adv_stream.state_dict())

    def get_q_values(self, x, is_target=False):
        if is_target:
            f = self.target_feature_layer(x)
            v = self.target_value_stream(f)
            a = self.target_adv_stream(f)
        else:
            f = self.feature_layer(x)
            v = self.value_stream(f)
            a = self.adv_stream(f)
        return v + (a - a.mean(dim=1, keepdim=True))

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.9)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def training_step(self, batch, batch_idx):
        # Epsilon-greedy action selection
        if random.random() < self.epsilon:
            action = random.randint(0, 3)
        else:
            state_t = torch.FloatTensor(self.state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_vals = self.get_q_values(state_t)
            action = q_vals.argmax().item()

        # Take action using the original Gridworld's makeMove
        self.game.makeMove(action_set[action])
        next_state = get_state(self.game)
        reward = self.game.reward()
        self.episode_steps += 1

        # Check terminal conditions
        done = False
        if reward != -1:
            done = True
        elif self.episode_steps >= self.max_steps_per_episode:
            done = True

        self.buffer.push(self.state, action, reward, next_state, float(done))
        self.state = next_state
        self.episode_reward += reward

        if done:
            # Reset environment
            self.game = Gridworld(size=4, mode='random')
            self.state = get_state(self.game)
            self.log('episode_reward', self.episode_reward, prog_bar=False)
            self.episode_count += 1
            self.episode_reward = 0
            self.episode_steps = 0
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        if len(self.buffer) < self.batch_size:
            return None

        s, a, r, ns, d = self.buffer.sample(self.batch_size)
        s = torch.FloatTensor(s).to(self.device)
        a = torch.LongTensor(a).unsqueeze(1).to(self.device)
        r = torch.FloatTensor(r).unsqueeze(1).to(self.device)
        ns = torch.FloatTensor(ns).to(self.device)
        d = torch.FloatTensor(d).unsqueeze(1).to(self.device)

        q_values = self.get_q_values(s).gather(1, a)

        with torch.no_grad():
            # Double DQN: Main network selects, Target network evaluates
            best_actions = self.get_q_values(ns).argmax(1).unsqueeze(1)
            next_q_values = self.get_q_values(ns, is_target=True).gather(1, best_actions)
            target_q = r + self.gamma * next_q_values * (1 - d)

        loss = self.loss_fn(q_values, target_q)
        self.log('train_loss', loss, prog_bar=False)

        if self.global_step % self.sync_freq == 0:
            self.sync_target()

        return loss

# ============================================================
# 4. Plotting and Testing
# ============================================================
def plot_loss(log_dir="lightning_logs"):
    if not os.path.exists(log_dir):
        print("No lightning_logs found.")
        return

    versions = [d for d in os.listdir(log_dir) if d.startswith('version_')]
    if not versions:
        return

    latest_version = sorted(versions, key=lambda v: int(v.split('_')[1]))[-1]
    csv_path = os.path.join(log_dir, latest_version, 'metrics.csv')

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        if 'train_loss' in df.columns:
            loss_data = df['train_loss'].dropna()
            smoothed_loss = loss_data.rolling(window=50).mean()

            plt.figure(figsize=(10, 6))
            plt.plot(smoothed_loss.values, alpha=0.8, color='crimson')
            plt.title('Training Loss (Smoothed) - Random Mode')
            plt.xlabel('Training Steps')
            plt.ylabel('Loss (MSE)')
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.savefig('hw3_3_loss.png')
            print("\nSuccessfully saved smoothed training loss plot to hw3_3_loss.png")

def run_tests(model, num_tests=10):
    print(f"\n--- Running {num_tests} Tests in Random Mode ---")
    model.eval()

    success_count = 0
    for i in range(1, num_tests + 1):
        game = Gridworld(size=4, mode='random')
        state = get_state(game)
        done = False
        steps = 0
        status = "Failed (Hit Pit or Timeout)"

        while not done:
            state_t = torch.FloatTensor(state).unsqueeze(0).to(model.device)
            with torch.no_grad():
                q_vals = model.get_q_values(state_t)
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
                status = "Failed (Timeout)"
                done = True

        print(f"Test {i:2d} | Steps Taken: {steps:2d} | Status: {status}")

    print(f"\nOverall Success Rate: {success_count}/{num_tests} ({success_count/num_tests*100:.1f}%)")

# ============================================================
# 5. Main
# ============================================================
if __name__ == '__main__':
    model = LitDQN(lr=1e-3)
    dataloader = DataLoader(DummyDataset(size=15000), batch_size=1)

    print("\n--- Starting PyTorch Lightning Training (Random Mode) ---")
    trainer = pl.Trainer(
        max_steps=15000,
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
        enable_checkpointing=False,
        enable_progress_bar=False,
        logger=True
    )

    trainer.fit(model, train_dataloaders=dataloader)

    # 1. Plot Loss
    plot_loss()

    # 2. Run Test Loop
    run_tests(model, num_tests=10)
