# eval_dqn.py (shared version)
import torch
from dqn_env import MultiTruckEnv
from dqn_agent import DQNAgent

def load_agents(cfg, path):
    env = MultiTruckEnv(cfg)
    agent = DQNAgent(env.obs_dim, env.action_space.n, cfg)
    sd = torch.load(path, map_location="cpu")
    agent.q_net.load_state_dict(sd)
    agent.eps = 0.0
    return env, agent

def rollout_greedy(env, agent):
    obs_all = env.reset()
    total = [0.0]*env.n_agents
    done = [False]*env.n_agents
    last_info = {}

    while not all(done):
        acts = [agent.act_eval(obs_all[i]) for i in range(env.n_agents)]
        obs_all, r, done, info = env.step(acts)
        last_info = info
        for i in range(env.n_agents):
            total[i] += r[i]

    avg = sum(total)/len(total)
    return avg, env.sim, last_info
