# hyperopt_dqn.py
from __future__ import annotations

import math
import optuna
from optuna.pruners import MedianPruner
from typing import Literal, Dict, Any, List
from copy import deepcopy

from config import CONFIG
from dqn_train import train_multi
from dqn_env import MultiTruckEnv
from dqn_agent import DQNAgent


def eval_greedy(cfg: Dict[str, Any], agents: List[DQNAgent]) -> Dict[str, Any]:
    """Run one greedy episode. Supports shared-policy (len(agents)==1) or per-truck."""
    env = MultiTruckEnv(cfg)
    for ag in agents:
        ag.eps = 0.0
    shared = (len(agents) == 1)
    policy = agents[0] if shared else None

    obs_all = env.reset()
    totals = [0.0] * env.n_agents
    done = [False] * env.n_agents
    last_info = {}

    while not all(done):
        acts = ([policy.act_eval(obs_all[i]) for i in range(env.n_agents)]
                if shared else
                [agents[i].act_eval(obs_all[i]) for i in range(env.n_agents)])
        obs_all, rewards, done, info = env.step(acts)
        last_info = info
        for i in range(env.n_agents):
            totals[i] += rewards[i]

    return {
        "avg_reward": sum(totals) / env.n_agents,
        "costs": last_info.get("costs", {})
    }


def objective(
    trial: optuna.Trial,
    episodes: int = 10,
    optimize_for: Literal["reward", "cost"] = "cost",
    seeds_per_trial: int = 3,          # average across seeds for stability
    steps_per_day_hpo: int = 200,      # shorter episodes for HPO speed
) -> float:
    """
    HPO tunes ONLY:
      - DQN hyperparameters
      - N_TRUCKS (fleet size)

    All business/economic parameters and demand settings are kept as in CONFIG.
    """
    # ---- Base cfg (don’t mutate global CONFIG) ----
    base_cfg = deepcopy(CONFIG)
    base_cfg["STEPS_PER_DAY"] = steps_per_day_hpo  # speed-up for HPO

    # ---------- DQN knobs ----------
    base_cfg["LR"]         = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    base_cfg["GAMMA"]      = trial.suggest_float("gamma", 0.90, 0.999)
    base_cfg["EPS_DECAY"]  = trial.suggest_float("eps_decay", 0.99, 0.9999)
    base_cfg["BATCH_SIZE"] = trial.suggest_categorical("batch_size", [32, 64, 128])
    base_cfg["HIDDEN"]     = trial.suggest_categorical("hidden", [64, 128, 256])
    base_cfg["EPS_END"]    = trial.suggest_float("eps_end", 0.01, 0.10)

    # ---------- Fleet size ONLY ----------
    base_cfg["N_TRUCKS"]   = trial.suggest_int("n_trucks", 2, 8)

    # ---------- Multi-seed averaging ----------
    scores: List[float] = []
    for k in range(seeds_per_trial):
        cfg = deepcopy(base_cfg)
        cfg["SEED"] = trial.suggest_int(f"seed_{k}", 1, 10_000)

        # Train (shared policy); don’t save checkpoints during HPO
        agents, rewards_hist, _ = train_multi(cfg, episodes=episodes, verbose=False, save_checkpoints=False)

        # (Optional) pruning signal based on recent rewards
        if len(rewards_hist) >= 5:
            trial.report(sum(rewards_hist[-5:]) / 5.0, step=k)
            if trial.should_prune():
                raise optuna.TrialPruned()

        # Greedy eval under the SAME config (economics unchanged)
        metrics = eval_greedy(cfg, agents)
        if optimize_for == "reward":
            scores.append(metrics["avg_reward"])
        else:
            total_eur = float(metrics["costs"].get("total_eur", 0.0))
            if math.isnan(total_eur) or math.isinf(total_eur):
                total_eur = 1e9
            scores.append(-total_eur)  # maximize negative cost

    return float(sum(scores) / len(scores))


def run_hyperopt(
    n_trials: int = 20,
    episodes: int = 10,
    optimize_for: Literal["reward", "cost"] = "cost",
) -> optuna.Study:
    study = optuna.create_study(
        direction="maximize",
        pruner=MedianPruner(n_startup_trials=5),
    )

    def _obj(tr: optuna.Trial) -> float:
        return objective(
            tr,
            episodes=episodes,
            optimize_for=optimize_for,
            seeds_per_trial=3,
            steps_per_day_hpo=200,
        )

    study.optimize(_obj, n_trials=n_trials, show_progress_bar=True)
    print("\n=== Optuna Results ===")
    print("Best value:", study.best_value)
    print("Best hyperparameters:", study.best_params)
    return study


if __name__ == "__main__":
    run_hyperopt(
        n_trials=5,
        episodes=5,          # small for speed; raise later (e.g., 30–50)
        optimize_for="cost",  # or "reward"
    )
