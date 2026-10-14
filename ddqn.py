# ddqn.py

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

from params import lr, epsilon_start, batch_size, gamma, epsilon_end, epsilon_decay

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Standard Q-Network

class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_size),
        )

    def forward(self, x):
        return self.net(x)


# Dueling Q-Network 

class DuelingQNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()

        self.feature = nn.Sequential(
            nn.Linear(state_size, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
        )
        self.value_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_size),
        )

    def forward(self, x):
        feat = self.feature(x)
        V    = self.value_stream(feat)
        A    = self.advantage_stream(feat)
        # Q(s,a) = V(s) + A(s,a) - mean_a A(s,a)   
        return V + (A - A.mean(dim=1, keepdim=True))


# Replay Buffer 

class ReplayBuffer:
    def __init__(self, capacity=50_000):   
        self.buffer = deque(maxlen=capacity)

    def push(self, s, a, r, s_next, done):
        self.buffer.append((s, a, r, s_next, done))

    def sample(self, batch_size):
        batch              = random.sample(self.buffer, batch_size)
        s, a, r, s_next, done = zip(*batch)
        return (np.array(s, dtype=np.float32),
                np.array(a, dtype=np.int64),
                np.array(r, dtype=np.float32),
                np.array(s_next, dtype=np.float32),
                np.array(done, dtype=np.float32))

    def __len__(self):
        return len(self.buffer)


# DDQN Agent 

class DDQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size  = state_size
        self.action_size = action_size

        self.q_net      = QNetwork(state_size, action_size).to(device)
        self.target_net = QNetwork(state_size, action_size).to(device)

        self.optimizer  = optim.AdamW(self.q_net.parameters(), lr=lr, weight_decay=1e-4)
        self.buffer     = ReplayBuffer()
        self.epsilon    = epsilon_start
        self.update_target()

    def update_target(self):
        self.target_net.load_state_dict(self.q_net.state_dict())

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_size)
        s = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            return torch.argmax(self.q_net(s)).item()

    def train_step(self):
        if len(self.buffer) < batch_size:
            return None

        s, a, r, s_next, done = self.buffer.sample(batch_size)

        s      = torch.FloatTensor(s).to(device)
        a      = torch.LongTensor(a).unsqueeze(1).to(device)
        r      = torch.FloatTensor(r).unsqueeze(1).to(device)
        s_next = torch.FloatTensor(s_next).to(device)
        done   = torch.FloatTensor(done).unsqueeze(1).to(device)

        q_val        = self.q_net(s).gather(1, a)
        next_actions = torch.argmax(self.q_net(s_next), dim=1, keepdim=True)  # DDQN
        next_q       = self.target_net(s_next).gather(1, next_actions)
        target       = r + gamma * next_q * (1 - done)

        loss = nn.SmoothL1Loss()(q_val, target.detach())   # Huber loss 
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
        self.optimizer.step()
        return loss.item()

    def decay_epsilon(self):
        self.epsilon = max(epsilon_end, self.epsilon * epsilon_decay)


# Dueling DDQN Agent

class DuelingDDQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size  = state_size
        self.action_size = action_size

        self.q_net      = DuelingQNetwork(state_size, action_size).to(device)
        self.target_net = DuelingQNetwork(state_size, action_size).to(device)

        self.optimizer  = optim.AdamW(self.q_net.parameters(), lr=lr, weight_decay=1e-4)
        self.buffer     = ReplayBuffer()
        self.epsilon    = epsilon_start
        self.update_target()

    def update_target(self):
        self.target_net.load_state_dict(self.q_net.state_dict())

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_size)
        s = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            return torch.argmax(self.q_net(s)).item()

    def train_step(self):
        if len(self.buffer) < batch_size:
            return None

        s, a, r, s_next, done = self.buffer.sample(batch_size)

        s      = torch.FloatTensor(s).to(device)
        a      = torch.LongTensor(a).unsqueeze(1).to(device)
        r      = torch.FloatTensor(r).unsqueeze(1).to(device)
        s_next = torch.FloatTensor(s_next).to(device)
        done   = torch.FloatTensor(done).unsqueeze(1).to(device)

        q_val        = self.q_net(s).gather(1, a)
        next_actions = torch.argmax(self.q_net(s_next), dim=1, keepdim=True)
        next_q       = self.target_net(s_next).gather(1, next_actions)
        target       = r + gamma * next_q * (1 - done)

        loss = nn.SmoothL1Loss()(q_val, target.detach())
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
        self.optimizer.step()
        return loss.item()

    def decay_epsilon(self):
        self.epsilon = max(epsilon_end, self.epsilon * epsilon_decay)
