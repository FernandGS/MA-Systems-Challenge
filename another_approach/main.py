# main.py
# Entry point for the Trash Collection multi-agent simulation
# Supports: baseline sim, DQN training, greedy eval

import argparse
from config import CONFIG
from city import City
from sim import Simulation
from visualize import preview
from dqn_train import train_multi
from eval_dqn import load_agents, rollout_greedy


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
        help="For shared policy eval, provide a single .pt path.",
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
            raise RuntimeError("train_multi could not be imported.")
        agents, rewards_hist, paths = train_multi(cfg, episodes=args.episodes, verbose=True)
        print("Saved model:", paths[0] if paths else "(not saved)")

    elif args.mode == "eval":
        if not args.model_paths:
            print("Provide --model_paths path/to/dqn_shared_xxx.pt")
            return
        env, agent = load_agents(cfg, args.model_paths[0])
        avg_r, sim_rl, info = rollout_greedy(env, agent)
        print(f"Eval average reward: {avg_r:.3f}")
        print("Costs:", info.get("costs", {}))

        # visualize & export the RL run you just evaluated:
        preview(sim_rl)
        sim_rl.export_json(cfg["JSON_EXPORT_PATH"])

    else:
        raise ValueError(f"unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
