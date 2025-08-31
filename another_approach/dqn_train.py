# dqn_train_multi.py (shared policy variant)

import os, time, torch
from tqdm import trange
from dqn_env import MultiTruckEnv
from dqn_agent import DQNAgent  # reuse your agent class

def train_multi(cfg, episodes=200, verbose=True, save_checkpoints=True):
    env = MultiTruckEnv(cfg)

    # ONE shared agent
    agent = DQNAgent(env.obs_dim, env.action_space.n, cfg)

    rewards_hist = []

    for ep in trange(episodes, desc="Training Episodes", unit="ep"):
        obs_all = env.reset()
        total_rewards = [0.0]*env.n_agents
        done = [False]*env.n_agents

        while not all(done):
            # per-truck action from the SAME policy
            acts = [agent.act(obs_all[i]) for i in range(env.n_agents)]
            next_obs_all, rewards, done, info = env.step(acts)

            # push ALL transitions into the same buffer
            for i in range(env.n_agents):
                agent.store(obs_all[i], acts[i], rewards[i], next_obs_all[i], done[i])
                total_rewards[i] += rewards[i]

            # do one (or a few) updates per env step
            agent.update()

            obs_all = next_obs_all

        avg_reward = sum(total_rewards)/env.n_agents
        rewards_hist.append(avg_reward)
        if verbose:
            print(f"Ep {ep} avg reward={avg_reward:.2f} eps={agent.eps:.2f}")

    # save a single checkpoint (optional)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    paths = []
    if save_checkpoints:
        os.makedirs("models", exist_ok=True)
        p = f"models/dqn_shared_{stamp}.pt"
        torch.save(agent.q_net.state_dict(), p)
        paths.append(p)

    # Return a list for compatibility (N identical “agents”), but it’s the same policy
    return [agent], rewards_hist, paths
