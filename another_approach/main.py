# main.py
# Entry point for the Trash Collection multi-agent simulation
# Supports: baseline sim, DQN training, KPI dashboard

import argparse
from config import CONFIG
from city import City
from sim import Simulation
from visualize import preview
from dashboard import show_dashboard

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
    parser.add_argument("--mode", choices=["baseline","dqn","dashboard"], default="baseline")
    parser.add_argument("--episodes", type=int, default=50)
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
        agents, rewards_hist = train_multi(cfg, episodes=args.episodes)
        print("Training complete.")
        # After training, we could run eval sim:
        sim, costs = run_baseline(cfg)
        from dashboard import show_dashboard
        import streamlit as st
        show_dashboard(sim, costs, rewards_hist)

    elif args.mode == "dashboard":
        sim, costs = run_baseline(cfg)
        import streamlit as st
        show_dashboard(sim, costs)

if __name__ == "__main__":
    main()
