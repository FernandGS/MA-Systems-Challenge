# main.py
# Entry point for the Trash Collection multi-agent simulation
# Supports: baseline sim, DQN training, greedy eval

import argparse
from config import CONFIG
from city import City
from sim import Simulation
from visualize import preview
from eval_dqn import load_agents, rollout_greedy

# Optional DQN imports
try:
    from dqn_train_multi import train_multi
except ImportError:
    train_multi = None


def run_baseline(cfg):
    """Run a baseline sim (no RL)."""
    city = City(cfg)
    sim = Simulation(cfg, city)
    sim.run(cfg["STEPS_PER_DAY"])
    return sim, sim.summary_costs()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["baseline", "dqn", "eval"], default="baseline")
    parser.add_argument("--episodes", type=int, default=50, help="Episodes for DQN training")
    parser.add_argument(
        "--model_paths",
        nargs="*",
        default=None,
        help="Paths to per-truck .pt files for --mode eval (order: truck0, truck1, ...)",
    )
    args = parser.parse_args()

    cfg = CONFIG

    if args.mode == "baseline":
        sim, costs = run_baseline(cfg)
        preview(sim)
        sim.export_json(cfg["JSON_EXPORT_PATH"])
        print("Summary costs:", costs)

    elif args.mode == "dqn":
        if train_multi is None:
            print("DQN not available. Please check dqn_train_multi.py")
            return
        # train_multi in deiner Version gibt (agents, rewards_hist, paths) zurück
        agents, rewards_hist, paths = train_multi(cfg, episodes=args.episodes, verbose=True)
        print("Training complete. Saved model paths:", paths)

        # Danach nur eine schnelle Baseline-Runde für eine animierte Vorschau + Kosten
        sim, costs = run_baseline(cfg)
        preview(sim)
        print("Summary costs:", costs)
        # Hinweis: Für interaktive KPIs -> streamlit run dashboard_app.py

    elif args.mode == "eval":
        if not args.model_paths or len(args.model_paths) < cfg["N_TRUCKS"]:
            print(f"Provide --model_paths for at least {cfg['N_TRUCKS']} trucks.")
            return
        env, agents = load_agents(cfg, args.model_paths[:cfg["N_TRUCKS"]])
        avg_r, sim, info = rollout_greedy(env, agents)
        print(f"Eval average reward: {avg_r:.3f}")
        costs = info.get("costs", {})
        print("Costs:", costs)
        preview(sim)
        sim.export_json(cfg["JSON_EXPORT_PATH"])

    else:
        raise ValueError(f"unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
