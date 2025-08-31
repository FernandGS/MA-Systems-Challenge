# dqn_agent_multi.py
# Independent DQN agents for multi-truck training.

import random
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, action_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DQNAgent:
    def __init__(self, obs_dim: int, action_dim: int, cfg: dict):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        hidden = cfg.get("HIDDEN", 128)
        self.q_net = QNetwork(obs_dim, action_dim, hidden).to(self.device)
        self.target_net = QNetwork(obs_dim, action_dim, hidden).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=cfg.get("LR", 1e-3))
        self.gamma = cfg.get("GAMMA", 0.99)

        self.eps = cfg.get("EPS_START", 1.0)
        # Accept either EPS_END (from your CONFIG) or EPS_MIN
        self.eps_min = cfg.get("EPS_END", cfg.get("EPS_MIN", 0.05))
        self.eps_decay = cfg.get("EPS_DECAY", 0.995)

        self.buffer = deque(maxlen=cfg.get("BUFFER_SIZE", 50_000))
        self.batch_size = cfg.get("BATCH_SIZE", 64)
        self.tau = cfg.get("TAU", 0.01)  # soft update mixing factor
        self.action_dim = action_dim

    # ε-greedy policy
    def act(self, state):
        if random.random() < self.eps:
            return random.randrange(self.action_dim)
        s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q = self.q_net(s)
        return int(q.argmax(dim=1).item())#
    
    def act_eval(self, state):
        s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q = self.q_net(s)
        return int(q.argmax(dim=1).item())

    def store(self, s, a, r, s2, d):
        self.buffer.append((s, a, r, s2, d))

    def update(self):
        if len(self.buffer) < self.batch_size:
            return

        batch = random.sample(self.buffer, self.batch_size)
        s, a, r, s2, d = map(np.array, zip(*batch))

        s  = torch.tensor(s,  dtype=torch.float32, device=self.device)
        a  = torch.tensor(a,  dtype=torch.int64,   device=self.device).unsqueeze(1)
        r  = torch.tensor(r,  dtype=torch.float32, device=self.device).unsqueeze(1)
        s2 = torch.tensor(s2, dtype=torch.float32, device=self.device)
        d  = torch.tensor(d,  dtype=torch.float32, device=self.device).unsqueeze(1)

        # Q(s,a)
        q = self.q_net(s).gather(1, a)

        # target = r + (1-d) * gamma * max_a' Q_target(s', a')
        with torch.no_grad():
            q2 = self.target_net(s2).max(dim=1, keepdim=True)[0]
            target = r + (1.0 - d) * self.gamma * q2

        loss = (q - target).pow(2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Soft target update
        with torch.no_grad():
            for tgt, src in zip(self.target_net.parameters(), self.q_net.parameters()):
                tgt.data.mul_(1.0 - self.tau).add_(self.tau * src.data)

        # Decay exploration
        self.eps = max(self.eps_min, self.eps * self.eps_decay)
