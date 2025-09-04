#!/usr/bin/env python3
# train_dqn.py
#
# One-stop DQN pretrainer for your waste-collection sim.
# - Runs N episodes x M steps with POLICY="dqn"
# - Saves per-truck weights to --weights-dir (default: dqn_weights)
# - Prints simple KPIs per episode so you can see progress
#
# Examples:
#   python train_dqn.py --episodes 20 --steps 2000 --weights-dir dqn_weights
#   python train_dqn.py --episodes 10 --steps 1500 --agents 4 --bins 18 --seed 123
#   python train_dqn.py --episodes 8  --steps 2500 --save-every 1000
#
# Tip: later, run eval-only with:
#   python viz_rich.py --mode live --policy dqn --weights-dir dqn_weights --freeze-dqn --steps 400 --ids

import argparse, os
from copy import deepcopy
from config import CONFIG
from city import City
from sim import Simulation

def kpis(sim):
    total_pickup = sum(e.get("amount", 0) for e in sim.events if e.get("type") == "pickup")
    overflows     = sum(1 for e in sim.events if e.get("type") == "overflow")
    dumps         = sum(1 for e in sim.events if e.get("type") == "drop")
    km            = sum(getattr(t, "km_total", 0.0) for t in sim.trucks)
    return dict(pickup=total_pickup, overflows=overflows, dumps=dumps, km=km)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--steps",    type=int, default=2000)
    ap.add_argument("--weights-dir", type=str, default="dqn_weights")
    ap.add_argument("--save-every",  type=int, default=1000, help="autosave period (steps)")
    ap.add_argument("--seed",     type=int, default=None)
    ap.add_argument("--agents",   type=int, default=None, help="override N_TRUCKS")
    ap.add_argument("--bins",     type=int, default=None, help="override N_BINS")
    ap.add_argument("--bin-cap",  type=int, default=None)
    ap.add_argument("--planner",  choices=["graph","grid"], default="graph")
    args = ap.parse_args()

    cfg = deepcopy(CONFIG)
    cfg["POLICY"] = "dqn"
    cfg["DQN_WEIGHTS_DIR"] = args.weights_dir
    cfg["DQN_SAVE_EVERY_STEPS"] = int(args.save_every)
    if args.seed is not None:    cfg["SEED"] = int(args.seed)
    if args.agents is not None:  cfg["N_TRUCKS"] = int(args.agents)
    if args.bins is not None:    cfg["N_BINS"] = int(args.bins)
    if args.bin_cap is not None: cfg["BIN_CAPACITY"] = int(args.bin_cap)

    # reasonable exploration defaults
    cfg["EPS_START"] = cfg.get("EPS_START", 0.8)
    cfg["EPS_END"]   = cfg.get("EPS_END",   0.05)
    cfg["EPS_DECAY"] = cfg.get("EPS_DECAY", 0.997)

    os.makedirs(cfg["DQN_WEIGHTS_DIR"], exist_ok=True)

    for ep in range(1, args.episodes+1):
        city = City(cfg)
        sim  = Simulation(cfg=cfg, city=city, planner=args.planner)
        sim.run(args.steps)
        m = kpis(sim)
        print(f"[ep {ep:02d}/{args.episodes}] steps={args.steps}  "
              f"pickup={m['pickup']}  overflows={m['overflows']}  dumps={m['dumps']}  km={m['km']:.1f}")

    print(f"[done] weights saved under: {os.path.abspath(cfg['DQN_WEIGHTS_DIR'])}")
    print("Use them with viz_rich:  --policy dqn --weights-dir dqn_weights --freeze-dqn")

if __name__ == "__main__":
    main()
