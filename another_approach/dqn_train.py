# dqn_train_multi.py (shared policy variant)

import os, time, torch
from tqdm import trange
from dqn_env import MultiTruckEnv
from dqn_agent import DQNAgent  # reuse your agent class

def _eval_greedy_cost(cfg, agent):
    """Greedy one-shot eval; returns total_eur for selection."""
    env = MultiTruckEnv(cfg)
    agent_eps_backup = agent.eps
    agent.eps = 0.0
    obs_all = env.reset()
    done = [False] * env.n_agents
    last_info = {}
    while not all(done):
        acts = [agent.act_eval(obs_all[i]) for i in range(env.n_agents)]
        obs_all, _r, done, info = env.step(acts)
        last_info = info
    agent.eps = agent_eps_backup
    return float(last_info.get("costs", {}).get("total_eur", float("inf")))

def train_multi(cfg, episodes=200, verbose=True, save_checkpoints=True, eval_every=5):
    env = MultiTruckEnv(cfg)

    # ONE shared agent
    agent = DQNAgent(env.obs_dim, env.action_space.n, cfg)

    rewards_hist = []
    best_cost = float("inf")
    best_path = None

    stamp = time.strftime("%Y%m%d-%H%M%S")
    last_path = None

    for ep in trange(episodes, desc="Training Episodes", unit="ep"):
        obs_all = env.reset()
        total_rewards = [0.0]*env.n_agents
        done = [False]*env.n_agents

        while not all(done):
            acts = [agent.act(obs_all[i]) for i in range(env.n_agents)]
            next_obs_all, rewards, done, info = env.step(acts)

            for i in range(env.n_agents):
                agent.store(obs_all[i], acts[i], rewards[i], next_obs_all[i], done[i])
                total_rewards[i] += rewards[i]

            agent.update()
            obs_all = next_obs_all

        avg_reward = sum(total_rewards)/env.n_agents
        rewards_hist.append(avg_reward)
        if verbose:
            print(f"Ep {ep} avg reward={avg_reward:.2f} eps={agent.eps:.2f}")

        # Periodic greedy eval -> keep the best by ECONOMIC objective
        if (ep + 1) % eval_every == 0:
            cost = _eval_greedy_cost(cfg, agent)
            if verbose:
                print(f"[eval @ ep {ep+1}] total_eur={cost:.2f} (best={best_cost:.2f})")
            if cost < best_cost and save_checkpoints:
                os.makedirs("models", exist_ok=True)
                best_cost = cost
                best_path = f"models/dqn_shared_best_{stamp}.pt"
                torch.save(agent.q_net.state_dict(), best_path)

        # (optional) also save rolling last snapshot so you can inspect it
        if save_checkpoints:
            os.makedirs("models", exist_ok=True)
            last_path = f"models/dqn_shared_last_{stamp}.pt"
            torch.save(agent.q_net.state_dict(), last_path)

    paths = []
    if save_checkpoints:
        if last_path: paths.append(last_path)
        if best_path: paths.append(best_path)

    # Return a list for compatibility (N identical “agents”), but it’s the same policy
    return [agent], rewards_hist, paths
