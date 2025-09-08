"""Advanced DQN Agent with optional Double / Dueling / Prioritized Replay.

Clean rewrite to fix indentation issues introduced in prior patch merges.
"""

import random
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.utils as nn_utils


class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden: int = 128, dueling: bool = False):
        super().__init__()
        self.dueling = dueling
        self.feature = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU()
        )
        if dueling:
            self.val_head = nn.Sequential(
                nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1)
            )
            self.adv_head = nn.Sequential(
                nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, action_dim)
            )
        else:
            self.out_head = nn.Linear(hidden, action_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.feature(x)
        if self.dueling:
            v = self.val_head(h)
            a = self.adv_head(h)
            return v + a - a.mean(dim=1, keepdim=True)
        return self.out_head(h)


class DQNAgent:
    def __init__(self, obs_dim: int, action_dim: int, cfg: dict):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        hidden = cfg.get("HIDDEN", 128)
        dueling = bool(cfg.get("DQN_DUELING", True))
        self.double = bool(cfg.get("DQN_DOUBLE", True))
        self.prioritized = bool(cfg.get("DQN_PRIORITIZED", False))
        self.alpha = float(cfg.get("PRIORITY_ALPHA", 0.6))
        self.beta = float(cfg.get("PRIORITY_BETA", 0.4))
        self.beta_inc = float(cfg.get("PRIORITY_BETA_INC", 1e-5))
        self.epsilon_priority = 1e-6

        self.q_net = QNetwork(obs_dim, action_dim, hidden, dueling=dueling).to(self.device)
        self.target_net = QNetwork(obs_dim, action_dim, hidden, dueling=dueling).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=cfg.get("LR", 1e-3))
        self.gamma = cfg.get("GAMMA", 0.99)

        # Exploration schedule
        self.eps = cfg.get("EPS_START", 1.0)
        self.eps_min = cfg.get("EPS_END", cfg.get("EPS_MIN", 0.05))
        self.eps_decay = cfg.get("EPS_DECAY", 0.995)

        # Replay buffer (optionally prioritized)
        self.buffer = deque(maxlen=cfg.get("BUFFER_SIZE", 50_000))
        self.priorities = deque(maxlen=cfg.get("BUFFER_SIZE", 50_000))
        self.batch_size = cfg.get("BATCH_SIZE", 64)
        self.tau = cfg.get("TAU", 0.01)  # soft target update factor
        self.action_dim = action_dim

    def act(self, state):
        if random.random() < self.eps:
            return random.randrange(self.action_dim)
        s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q = self.q_net(s)
        return int(q.argmax(dim=1).item())

    def act_eval(self, state):
        s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q = self.q_net(s)
        return int(q.argmax(dim=1).item())

    def store(self, s, a, r, s2, d):
        if self.prioritized:
            max_p = max(self.priorities) if self.priorities else 1.0
            self.priorities.append(max_p)
        self.buffer.append((s, a, r, s2, d))

    def update(self):
        if len(self.buffer) < self.batch_size:
            return

        if self.prioritized:
            probs = np.array(self.priorities, dtype=np.float32)
            probs = probs ** self.alpha
            probs /= probs.sum()
            idxs = np.random.choice(len(self.buffer), self.batch_size, p=probs)
            batch = [self.buffer[i] for i in idxs]
            weights = (len(self.buffer) * probs[idxs]) ** (-self.beta)
            self.beta = min(1.0, self.beta + self.beta_inc)
            weights /= max(1e-8, weights.max())
            weights_t = torch.tensor(weights, dtype=torch.float32, device=self.device).unsqueeze(1)
        else:
            batch = random.sample(self.buffer, self.batch_size)
            idxs = None
            weights_t = torch.ones((self.batch_size, 1), dtype=torch.float32, device=self.device)

        s, a, r, s2, d = map(np.array, zip(*batch))
        s = torch.tensor(s, dtype=torch.float32, device=self.device)
        a = torch.tensor(a, dtype=torch.int64, device=self.device).unsqueeze(1)
        r = torch.tensor(r, dtype=torch.float32, device=self.device).unsqueeze(1)
        s2 = torch.tensor(s2, dtype=torch.float32, device=self.device)
        d = torch.tensor(d, dtype=torch.float32, device=self.device).unsqueeze(1)

        q = self.q_net(s).gather(1, a)
        with torch.no_grad():
            if self.double:
                next_q_main = self.q_net(s2)
                next_actions = next_q_main.argmax(dim=1, keepdim=True)
                q2 = self.target_net(s2).gather(1, next_actions)
            else:
                q2 = self.target_net(s2).max(dim=1, keepdim=True)[0]
            target = r + (1.0 - d) * self.gamma * q2

        td_error = target - q
        loss = (F.smooth_l1_loss(q, target, reduction='none') * weights_t).mean()
        self.optimizer.zero_grad()
        loss.backward()
        nn_utils.clip_grad_norm_(self.q_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        with torch.no_grad():
            for tgt, src in zip(self.target_net.parameters(), self.q_net.parameters()):
                tgt.data.mul_(1.0 - self.tau).add_(self.tau * src.data)

        if self.prioritized and idxs is not None:
            abs_err = td_error.detach().abs().cpu().numpy().flatten() + self.epsilon_priority
            for i, idx in enumerate(idxs):
                self.priorities[idx] = float(abs_err[i])

        self.eps = max(self.eps_min, self.eps * self.eps_decay)
