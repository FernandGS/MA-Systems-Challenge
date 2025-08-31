# dqn_agent_multi.py
# Independent DQN agents for multi-truck training.

import random, numpy as np, torch, torch.nn as nn, torch.optim as optim
from collections import deque

class QNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, action_dim)
        )
    def forward(self, x): return self.net(x)

class DQNAgent:
    def __init__(self, obs_dim, action_dim, cfg):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_net = QNetwork(obs_dim, action_dim).to(self.device)
        self.target_net = QNetwork(obs_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=cfg.get("LR",1e-3))
        self.gamma = cfg.get("GAMMA",0.99)
        self.eps = cfg.get("EPS_START",1.0)
        self.eps_min = cfg.get("EPS_MIN",0.05)
        self.eps_decay = cfg.get("EPS_DECAY",0.995)

        self.buffer = deque(maxlen=cfg.get("BUFFER_SIZE",50000))
        self.batch_size = cfg.get("BATCH_SIZE",64)
        self.tau = cfg.get("TAU",0.01)
        self.action_dim = action_dim

    def act(self, state):
        if random.random() < self.eps:
            return random.randrange(self.action_dim)
        s = torch.tensor(state,dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad(): q = self.q_net(s)
        return int(q.argmax().item())

    def store(self, s,a,r,s2,d): self.buffer.append((s,a,r,s2,d))

    def update(self):
        if len(self.buffer) < self.batch_size: return
        batch = random.sample(self.buffer, self.batch_size)
        s,a,r,s2,d = map(np.array, zip(*batch))
        s  = torch.tensor(s ,dtype=torch.float32).to(self.device)
        a  = torch.tensor(a ,dtype=torch.int64 ).unsqueeze(1).to(self.device)
        r  = torch.tensor(r ,dtype=torch.float32).unsqueeze(1).to(self.device)
        s2 = torch.tensor(s2,dtype=torch.float32).to(self.device)
        d  = torch.tensor(d ,dtype=torch.float32).unsqueeze(1).to(self.device)

        q = self.q_net(s).gather(1,a)
        with torch.no_grad():
            q2 = self.target_net(s2).max(1)[0].unsqueeze(1)
            target = r + (1-d)*self.gamma*q2
        loss = (q-target).pow(2).mean()
        self.optimizer.zero_grad(); loss.backward(); self.optimizer.step()

        # soft update
        for t, s in zip(self.target_net.parameters(), self.q_net.parameters()):
            t.data.copy_(t.data*(1-self.tau) + s.data*self.tau)

        self.eps = max(self.eps_min, self.eps*self.eps_decay)
