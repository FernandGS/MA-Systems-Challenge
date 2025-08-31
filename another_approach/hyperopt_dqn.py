# hyperopt_dqn.py
# Hyperparameter & economics optimization for multi-agent DQN with Optuna.
# - Starts from CONFIG and overrides only sampled knobs.
# - Trains briefly, then runs a greedy eval (eps=0) to compute an objective.
#
# Objective modes:
#   - "reward": maximize average per-episode reward
#   - "cost":   maximize (- total_eur) from the final eval (== minimize € cost)

from __future__ import annotations

import optuna
from typing import Literal, Dict, Any, List
from config import CONFIG
from dqn_train_multi import train_multi
from dqn_env_multi import MultiTruckEnv
from dqn_agent_multi import DQNAgent

# ---------------------------
# Greedy evaluation (eps = 0)
# ---------------------------
def eval_greedy(cfg: Dict[str, Any], agents: List[DQNAgent]) -> Dict[str, Any]:
    """Run one greedy episode with given agents (eps=0), return metrics."""
    env = MultiTruckEnv(cfg)
    # switch to greedy (no exploration)
    for ag in agents:
        ag.eps = 0.0

    obs_all = env.reset()
    totals = [0.0] * env.n_agents
    done = [False] * env.n_agents
    last_info = {}

    while not all(done):
        acts = [agents[i].act_eval(obs_all[i]) for i in range(env.n_agents)]
        obs_all, rewards, done, info = env.step(acts)
        last_info = info
        for i in range(env.n_agents):
            totals[i] += rewards[i]

    avg_reward = sum(totals) / env.n_agents
    costs = last_info.get("costs", {})
    return {"avg_reward": avg_reward, "costs": costs}


# ---------------------------
# Objective (Optuna)
# ---------------------------
def objective(
    trial: optuna.Trial,
    episodes: int = 20,
    optimize_for: Literal["reward", "cost"] = "reward",
    allow_env_changes: bool = False,
) -> float:
    """
    episodes: short training length per trial
    optimize_for: "reward" or "cost"
    allow_env_changes: if True, allows Optuna to change environment size/demand.
                       Keep False for apples-to-apples across trials.
    """
    # Start from global CONFIG (copy!)
    cfg = CONFIG.copy()

    # ---------- Algorithmic knobs (DQN) ----------
    cfg["LR"]         = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    cfg["GAMMA"]      = trial.suggest_float("gamma", 0.90, 0.999)
    cfg["EPS_DECAY"]  = trial.suggest_float("eps_decay", 0.99, 0.9999)
    cfg["BATCH_SIZE"] = trial.suggest_categorical("batch_size", [32, 64, 128])
    cfg["HIDDEN"]     = trial.suggest_categorical("hidden", [64, 128, 256])
    # Optional: exploration floor (useful if early convergence is bad)
    cfg["EPS_END"]    = trial.suggest_float("eps_end", 0.01, 0.10)  # overrides CONFIG's 0.05

    # ---------- Economics / penalties (business knobs) ----------
    # These affect reward because your reward = -€ costs.
    cfg["OVERFLOW_PENALTY_EUR"] = trial.suggest_float("overflow_eur", 200.0, 1500.0)
    cfg["WAGE_PER_HOUR"]        = trial.suggest_float("wage_eur_h", 15.0, 40.0)
    cfg["ENERGY_EUR_PER_UNIT"]  = trial.suggest_float("energy_eur_unit", 0.10, 0.60)
    cfg["MAINT_EUR_PER_KM"]     = trial.suggest_float("maint_eur_km", 0.03, 0.15)

    # ---------- Optional environment knobs ----------
    # Turn on only if you want to explore different demand/fleet setups;
    # otherwise keep fixed to compare policies fairly.
    if allow_env_changes:
        cfg["N_TRUCKS"] = trial.suggest_int("n_trucks", 2, 6)
        cfg["N_BINS"]   = trial.suggest_int("n_bins", 6, 20)
        # demand per step range (lo,hi)
        lo = trial.suggest_int("bin_fill_lo", 0, 2)
        hi = trial.suggest_int("bin_fill_hi", max(1, lo+1), 4)
        cfg["BIN_FILL_PER_STEP"] = (lo, hi)
        # capacity could also be tuned
        cfg["TRUCK_CAPACITY"] = trial.suggest_int("truck_cap", 60, 140)

    # (Optional) change seed per trial for robustness
    cfg["SEED"] = trial.suggest_int("seed", 1, 10_000)

    # ---------------- Train (short) ----------------
    # We keep model saving off for speed / cleanliness during search
    agents, rewards_hist = train_multi(cfg, episodes=episodes, verbose=False)

    # ---------------- Evaluate (greedy) ----------------
    eval_metrics = eval_greedy(cfg, agents)
    avg_reward = eval_metrics["avg_reward"]
    total_eur = float(eval_metrics["costs"].get("total_eur", 0.0))

    # ---------------- Return objective ----------------
    if optimize_for == "reward":
        # Maximize true average reward
        return avg_reward
    else:
        # Maximize negative total cost (== minimize cost)
        return -total_eur


# ---------------------------
# Entrypoint
# ---------------------------
def run_hyperopt(
    n_trials: int = 30,
    episodes: int = 20,
    optimize_for: Literal["reward", "cost"] = "cost",
    allow_env_changes: bool = False,
    seed: int | None = 42,
) -> optuna.Study:
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    def _obj(tr: optuna.Trial) -> float:
        return objective(
            tr,
            episodes=episodes,
            optimize_for=optimize_for,
            allow_env_changes=allow_env_changes,
        )

    study.optimize(_obj, n_trials=n_trials)

    print("\n=== Optuna Results ===")
    print("Best value:", study.best_value)
    print("Best hyperparameters:", study.best_params)
    return study


if __name__ == "__main__":
    # Defaults: optimize for COST (i.e., minimize €), fixed environment for fairness
    # Adjust episodes/trials upward when you see healthy spread in objective values.
    run_hyperopt(
        n_trials=20,
        episodes=20,
        optimize_for="cost",      # or "reward"
        allow_env_changes=False,  # set True if you want to co-design env sizing
        seed=42,
    )
