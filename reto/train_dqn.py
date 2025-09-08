import argparse, os, sys, json, statistics, time
from copy import deepcopy

# Ensure unity_hybrid is importable
ROOT = os.path.dirname(os.path.abspath(__file__))
HYB = os.path.join(ROOT, 'unity_hybrid')
if HYB not in sys.path:
    sys.path.insert(0, HYB)

from config import CONFIG  # type: ignore
from city import City  # type: ignore
from sim import Simulation  # type: ignore


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
    ap.add_argument("--eval-eps", type=int, default=0, help="number of evaluation episodes run after each training episode (frozen epsilon=0)")
    ap.add_argument("--early-stop", type=float, default=None, help="early stop when moving avg pickup per 1k steps exceeds this")
    ap.add_argument("--ma-window", type=int, default=5, help="moving average window for early stop")
    ap.add_argument("--log-jsonl", type=str, default="train_metrics.jsonl", help="append metrics as JSON lines")
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

    metrics_history = []  # store pickup per 1k steps for early stop
    best_score = None
    best_dir = os.path.join(cfg["DQN_WEIGHTS_DIR"], "best")
    os.makedirs(best_dir, exist_ok=True)

    log_path = args.log_jsonl
    log_fh = open(log_path, "a", buffering=1) if log_path else None

    t0 = time.time()
    for ep in range(1, args.episodes+1):
        city = City(cfg)
        sim  = Simulation(cfg=cfg, city=city, planner=args.planner)
        sim.run(args.steps)
        m = kpis(sim)
        pickup_rate = m['pickup'] / max(1, args.steps/1000.0)
        metrics_history.append(pickup_rate)
        ma = None
        if len(metrics_history) >= args.ma_window:
            ma = statistics.mean(metrics_history[-args.ma_window:])
        # extract epsilon stats (agents may differ)
        epsilons = {}
        try:
            # sim.policy may be DQNManager
            epsilons = getattr(sim.policy, 'last_eps', {}) or {}
        except Exception:
            pass
        eps_avg = statistics.mean(epsilons.values()) if epsilons else None
        msg = (f"[ep {ep:02d}/{args.episodes}] steps={args.steps} pickup={m['pickup']} overflows={m['overflows']} "
               f"dumps={m['dumps']} km={m['km']:.1f} pickup_rate={pickup_rate:.2f}" +
               (f" ma({args.ma_window})={ma:.2f}" if ma is not None else "") +
               (f" eps_avg={eps_avg:.3f}" if eps_avg is not None else ""))
        print(msg)
        if log_fh:
            rec = {"episode": ep, **m, "pickup_rate": pickup_rate, "ma": ma, "eps_avg": eps_avg, "epsilons": epsilons, "steps": args.steps, "t_sec": time.time()-t0}
            log_fh.write(json.dumps(rec) + "\n")
        # Save best (by pickup_rate)
        is_best = best_score is None or pickup_rate > best_score
        if is_best:
            best_score = pickup_rate
            # copy weight files
            for f in os.listdir(cfg["DQN_WEIGHTS_DIR"]):
                if f.endswith('.pt'):
                    src = os.path.join(cfg["DQN_WEIGHTS_DIR"], f)
                    dst = os.path.join(best_dir, f)
                    try:
                        import shutil
                        shutil.copy2(src, dst)
                    except Exception:
                        pass
        # Optional evaluation episodes with epsilon=0 (greedy)
        if args.eval_eps > 0:
            eval_pickups = []
            saved_eps = {}
            # force epsilon 0
            try:
                if hasattr(sim.policy, 'agents'):
                    for tid, agent in sim.policy.agents.items():
                        saved_eps[tid] = agent.eps
                        agent.eps = 0.0
            except Exception:
                pass
            for _ in range(args.eval_eps):
                ecity = City(cfg)
                esim  = Simulation(cfg=cfg, city=ecity, planner=args.planner)
                # freeze learning by setting update method no-op? simplest: skip
                esim.run(args.steps)
                em = kpis(esim)
                eval_pickups.append(em['pickup'])
            if eval_pickups:
                avg_eval = statistics.mean(eval_pickups)
                print(f"   eval: episodes={args.eval_eps} avg_pickup={avg_eval:.1f}")
                if log_fh:
                    log_fh.write(json.dumps({"episode": ep, "eval_avg_pickup": avg_eval, "eval_eps": args.eval_eps}) + "\n")
            # restore epsilon
            try:
                if hasattr(sim.policy, 'agents'):
                    for tid, agent in sim.policy.agents.items():
                        if tid in saved_eps:
                            agent.eps = saved_eps[tid]
            except Exception:
                pass
        # Early stopping
        if args.early_stop and ma is not None and ma >= args.early_stop:
            print(f"[early-stop] moving average {ma:.2f} >= {args.early_stop}; stopping at episode {ep}")
            break

    if log_fh:
        log_fh.close()

    print(f"[done] weights saved under: {os.path.abspath(cfg['DQN_WEIGHTS_DIR'])}")
    print("Use them with viz_rich:  --policy dqn --weights-dir dqn_weights --freeze-dqn")


if __name__ == "__main__":
    main()
