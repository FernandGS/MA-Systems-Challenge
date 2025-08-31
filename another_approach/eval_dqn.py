# eval_dqn.py
import torch
from dqn_env_multi import MultiTruckEnv
from dqn_agent_multi import DQNAgent

def load_agents(cfg, paths):
    env = MultiTruckEnv(cfg)
    agents = [DQNAgent(env.obs_dim, env.action_space.n, cfg) for _ in range(env.n_agents)]
    for i, p in enumerate(paths):
        sd = torch.load(p, map_location="cpu")
        agents[i].q_net.load_state_dict(sd)
        agents[i].eps = 0.0  # no exploration at eval
    return env, agents

def rollout_greedy(env, agents):
    obs_all = env.reset()
    total = [0.0]*env.n_agents
    done = [False]*env.n_agents
    while not all(done):
        acts = [agents[i].act_eval(obs_all[i]) for i in range(env.n_agents)]
        obs_all, r, done, info = env.step(acts)
        for i in range(env.n_agents):
            total[i] += r[i]
    return sum(total)/len(total), env.sim, info
