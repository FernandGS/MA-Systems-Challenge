# hyperopt_dqn.py
from __future__ import annotations

import math
from copy import deepcopy
from typing import Dict, Any, List, Literal, Sequence

import optuna
from optuna.pruners import MedianPruner

from config import CONFIG
from dqn_train import train_multi
from dqn_env import MultiTruckEnv
from dqn_agent import DQNAgent


# ----------------------------
# Evaluation (greedy rollout)
# ----------------------------
def eval_greedy(cfg: Dict[str, Any], agents: List[DQNAgent]) -> Dict[str, Any]:
    """
    Run one greedy episode. Supports shared-policy (len(agents)==1) or per-truck.
    Returns {"avg_reward": float, "costs": {...}}.
    """
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


# ----------------------------
# Core objective helper
# ----------------------------
def _run_trial_average(
    base_cfg: Dict[str, Any],
    episodes: int,
    seeds: Sequence[int],
    optimize_for: Literal["reward", "cost"] = "cost",
    trial: optuna.Trial | None = None,
) -> float:
    """
    Trains/evaluates for each seed in `seeds`, returns the average objective.
    No business/economic params are changed here (they should be set in base_cfg).
    """
    scores: List[float] = []
    for k, seed in enumerate(seeds):
        cfg = deepcopy(base_cfg)
        cfg["SEED"] = int(seed)

        # Train (shared policy); don't save checkpoints during HPO
        agents, rewards_hist, _ = train_multi(cfg, episodes=episodes, verbose=False, save_checkpoints=False)

        # Optional pruning signal based on recent rewards
        if trial is not None and len(rewards_hist) >= 5:
            trial.report(sum(rewards_hist[-5:]) / 5.0, step=k)
            if trial.should_prune():
                raise optuna.TrialPruned()

        metrics = eval_greedy(cfg, agents)
        if optimize_for == "reward":
            scores.append(metrics["avg_reward"])
        else:
            total_eur = float(metrics["costs"].get("total_eur", 0.0))
            if math.isnan(total_eur) or math.isinf(total_eur):
                total_eur = 1e9
            scores.append(-total_eur)  # maximize negative cost

    return float(sum(scores) / len(scores))


# ----------------------------
# Stage 1: tune hyperparameters ONLY
# ----------------------------
def _objective_stage1(
    trial: optuna.Trial,
    *,
    episodes: int,
    seeds: Sequence[int],
    steps_per_day_hpo: int,
    optimize_for: Literal["reward", "cost"],
) -> float:
    cfg = deepcopy(CONFIG)
    cfg["STEPS_PER_DAY"] = steps_per_day_hpo  # faster episodes for HPO
    # KEEP N_TRUCKS from CONFIG in stage 1 (no fleet-size tuning here)

    # ---- DQN knobs only ----
    cfg["LR"]         = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    cfg["GAMMA"]      = trial.suggest_float("gamma", 0.90, 0.999)
    cfg["EPS_DECAY"]  = trial.suggest_float("eps_decay", 0.99, 0.9999)
    cfg["BATCH_SIZE"] = trial.suggest_categorical("batch_size", [32, 64, 128])
    cfg["HIDDEN"]     = trial.suggest_categorical("hidden", [64, 128, 256])
    cfg["EPS_END"]    = trial.suggest_float("eps_end", 0.01, 0.10)

    return _run_trial_average(
        cfg, episodes=episodes, seeds=seeds, optimize_for=optimize_for, trial=trial
    )


def run_stage1_hparams(
    n_trials: int = 20,
    episodes: int = 15,
    seeds: Sequence[int] = (101, 202, 303),
    steps_per_day_hpo: int = 200,
    optimize_for: Literal["reward", "cost"] = "cost",
) -> optuna.Study:
    """
    Returns an Optuna Study with best DQN hyperparameters for the fixed environment
    (including the fixed N_TRUCKS from CONFIG).
    """
    study = optuna.create_study(direction="maximize", pruner=MedianPruner(n_startup_trials=5))
    study.optimize(
        lambda tr: _objective_stage1(
            tr,
            episodes=episodes,
            seeds=seeds,
            steps_per_day_hpo=steps_per_day_hpo,
            optimize_for=optimize_for,
        ),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    print("\n=== Stage 1 Results (Hyperparams) ===")
    print("Best value:", study.best_value)
    print("Best hyperparameters:", study.best_params)
    return study


# ----------------------------
# Stage 2: tune fleet size ONLY
# ----------------------------
def _objective_stage2(
    trial: optuna.Trial,
    *,
    fixed_hparams: Dict[str, Any],
    episodes: int,
    seeds: Sequence[int],
    steps_per_day_hpo: int,
    optimize_for: Literal["reward", "cost"],
    fleet_range: Sequence[int],
) -> float:
    cfg = deepcopy(CONFIG)
    cfg["STEPS_PER_DAY"] = steps_per_day_hpo

    # Fix best hyperparameters from Stage 1
    for k, v in fixed_hparams.items():
        cfg[k] = v

    # Tune fleet size only
    cfg["N_TRUCKS"] = trial.suggest_categorical("n_trucks", list(fleet_range))

    return _run_trial_average(
        cfg, episodes=episodes, seeds=seeds, optimize_for=optimize_for, trial=trial
    )


def run_stage2_fleet(
    fixed_hparams: Dict[str, Any],
    n_trials: int = 12,
    episodes: int = 15,
    seeds: Sequence[int] = (404, 505, 606),
    steps_per_day_hpo: int = 200,
    optimize_for: Literal["reward", "cost"] = "cost",
    fleet_range: Sequence[int] = (2, 3, 4, 5, 6, 7, 8),
) -> optuna.Study:
    """
    Returns an Optuna Study for fleet size search with the DQN hyperparams fixed.
    """
    study = optuna.create_study(direction="maximize", pruner=MedianPruner(n_startup_trials=3))
    study.optimize(
        lambda tr: _objective_stage2(
            tr,
            fixed_hparams=fixed_hparams,
            episodes=episodes,
            seeds=seeds,
            steps_per_day_hpo=steps_per_day_hpo,
            optimize_for=optimize_for,
            fleet_range=fleet_range,
        ),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    print("\n=== Stage 2 Results (Fleet Size) ===")
    print("Best value:", study.best_value)
    print("Best params:", study.best_params)
    return study


# ----------------------------
# Convenience: run both stages
# ----------------------------
def run_hpo_two_stage(
    *,
    stage1_trials: int = 20,
    stage1_episodes: int = 15,
    stage1_seeds: Sequence[int] = (101, 202, 303),

    stage2_trials: int = 12,
    stage2_episodes: int = 15,
    stage2_seeds: Sequence[int] = (404, 505, 606),
    fleet_range: Sequence[int] = (2, 3, 4, 5, 6, 7, 8),

    steps_per_day_hpo: int = 200,
    optimize_for: Literal["reward", "cost"] = "cost",
):
    """
    Runs Stage 1 (hyperparams) then Stage 2 (fleet size) and prints the final recommended config.
    """
    # Stage 1
    study1 = run_stage1_hparams(
        n_trials=stage1_trials,
        episodes=stage1_episodes,
        seeds=stage1_seeds,
        steps_per_day_hpo=steps_per_day_hpo,
        optimize_for=optimize_for,
    )
    best_hparams = {
        # Only pick the keys we actually tuned in stage 1:
        "LR": study1.best_params["lr"],
        "GAMMA": study1.best_params["gamma"],
        "EPS_DECAY": study1.best_params["eps_decay"],
        "BATCH_SIZE": study1.best_params["batch_size"],
        "HIDDEN": study1.best_params["hidden"],
        "EPS_END": study1.best_params["eps_end"],
    }

    # Stage 2
    study2 = run_stage2_fleet(
        fixed_hparams=best_hparams,
        n_trials=stage2_trials,
        episodes=stage2_episodes,
        seeds=stage2_seeds,
        steps_per_day_hpo=steps_per_day_hpo,
        optimize_for=optimize_for,
        fleet_range=fleet_range,
    )

    best_cfg = deepcopy(CONFIG)
    best_cfg.update(best_hparams)
    best_cfg["N_TRUCKS"] = study2.best_params["n_trucks"]

    print("\n=== Recommended Combined Config ===")
    print({k: best_cfg[k] for k in ["LR", "GAMMA", "EPS_DECAY", "EPS_END", "BATCH_SIZE", "HIDDEN", "N_TRUCKS"]})
    return study1, study2, best_cfg


# ----------------------------
# Script entry points
# ----------------------------
if __name__ == "__main__":
    # Example: run both stages quickly. Tweak trials/episodes as you like.
    run_hpo_two_stage(
        stage1_trials=12,
        stage1_episodes=12,
        stage1_seeds=(101, 202, 303),

        stage2_trials=8,
        stage2_episodes=12,
        stage2_seeds=(404, 505, 606),
        fleet_range=(2, 3, 4, 5, 6, 7, 8),

        steps_per_day_hpo=200,
        optimize_for="cost",
    )