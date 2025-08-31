import os, time, torch
from tqdm import trange
from dqn_env_multi import MultiTruckEnv
from dqn_agent_multi import DQNAgent

def train_multi(cfg, episodes=200, verbose=True):
    env = MultiTruckEnv(cfg)
    agents = [DQNAgent(env.obs_dim, env.action_space.n, cfg) for _ in range(env.n_agents)]
    rewards_hist = []

    for ep in trange(episodes, desc="Training Episodes", unit="ep"):
        obs_all = env.reset()
        total_rewards = [0.0]*env.n_agents
        done = [False]*env.n_agents

        while not all(done):
            acts = [agents[i].act(obs_all[i]) for i in range(env.n_agents)]
            obs2_all, rewards, done, info = env.step(acts)

            for i in range(env.n_agents):
                agents[i].store(obs_all[i], acts[i], rewards[i], obs2_all[i], done[i])
                agents[i].update()
                total_rewards[i] += rewards[i]

            obs_all = obs2_all

        avg_reward = sum(total_rewards)/env.n_agents
        rewards_hist.append(avg_reward)
        if verbose:
            print(f"Ep {ep} avg reward={avg_reward:.2f} eps={agents[0].eps:.2f}")

        stamp = time.strftime("%Y%m%d-%H%M%S")
    os.makedirs("models", exist_ok=True)
    paths = []
    for i, ag in enumerate(agents):
        p = f"models/dqn_truck{i}_{stamp}.pt"
        torch.save(ag.q_net.state_dict(), p)
        paths.append(p)
    return agents, rewards_hist, paths
