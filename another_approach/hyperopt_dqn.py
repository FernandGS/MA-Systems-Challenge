# hyperopt_dqn.py
# Hyperparameter optimization for multi-agent DQN with Optuna

import optuna
from dqn_train_multi import train_multi
from dqn_env_multi import MultiTruckEnv

def objective(trial):
    cfg = {
        # --- environment ---
        "MAP_SIZE": (220,160),
        "SEED": 42,
        "N_BINS": 12,
        "N_TRUCKS": 3,
        "BIN_CAPACITY": 100,
        "BIN_FILL_PER_STEP": (0,2),
        "TRUCK_CAPACITY": 80,
        "ENERGY_MAX": 100.0,
        "ENERGY_PER_M": 0.06,
        "ENERGY_RESERVE_M": 30.0,
        "DT": 1.0,
        "STEPS_PER_DAY": 300,
        "WAGE_PER_HOUR": 25.0,
        "OVERFLOW_PENALTY_EUR": 500.0,

        # --- hyperparameters sampled by Optuna ---
        "LR": trial.suggest_loguniform("lr", 1e-5, 1e-2),
        "GAMMA": trial.suggest_uniform("gamma", 0.90, 0.999),
        "EPS_DECAY": trial.suggest_uniform("eps_decay", 0.99, 0.9999),
        "BATCH_SIZE": trial.suggest_categorical("batch_size", [32, 64, 128]),
        "HIDDEN": trial.suggest_categorical("hidden", [64, 128, 256]),
    }

    agents, rewards_hist = train_multi(cfg, episodes=20, verbose=False)
    # Objective = maximize average reward (or minimize cost)
    return sum(rewards_hist[-5:]) / 5  # avg of last 5 episodes

def run_hyperopt(n_trials=30):
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    print("Best hyperparameters:", study.best_params)
    return study

if __name__ == "__main__":
    run_hyperopt(20)
