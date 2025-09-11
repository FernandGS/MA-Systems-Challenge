import argparse, os, sys, json, statistics, time, csv
from copy import deepcopy
from typing import Dict, Any, Tuple, List, Optional

# Ensure unity_hybrid is importable
ROOT = os.path.dirname(os.path.abspath(__file__))
HYB = os.path.join(ROOT, 'unity_hybrid')
if HYB not in sys.path:
    sys.path.insert(0, HYB)

from config import CONFIG  # type: ignore
from city import City  # type: ignore
from sim import Simulation  # type: ignore

# Optional TensorBoard
try:
    from torch.utils.tensorboard import SummaryWriter  # type: ignore
except Exception:  # pragma: no cover
    SummaryWriter = None  # type: ignore


def episode_reward(events, cfg: Dict[str, Any]) -> float:
    """Compute shaped reward from recent events list.
    This approximates internal RL shaping for reporting/selection."""
    rp = cfg.get('RL_REWARD_PICKUP', 0.1)
    rd = cfg.get('RL_REWARD_DUMP', 2.0)
    ro = cfg.get('RL_REWARD_OVERFLOW', -20.0)
    total = 0.0
    for e in events:
        et = e.get('type')
        if et == 'pickup':
            total += rp * e.get('amount', 0)
        elif et == 'drop':
            total += rd
        elif et == 'overflow':
            total += ro
    return total


def kpis(sim, cfg) -> Dict[str, Any]:
    pickups_amt = sum(e.get("amount", 0) for e in sim.events if e.get("type") == "pickup")
    pickups_cnt = sum(1 for e in sim.events if e.get("type") == "pickup")
    overflows   = sum(1 for e in sim.events if e.get("type") == "overflow")
    dumps       = sum(1 for e in sim.events if e.get("type") == "drop")
    km          = sum(getattr(t, "km_total", 0.0) for t in sim.trucks)
    shaped_r    = episode_reward(sim.events, cfg)
    return dict(pickup_amount=pickups_amt, pickup_events=pickups_cnt, overflows=overflows, dumps=dumps, km=km, reward=shaped_r)


def run_one(cfg, steps: int, planner: str) -> Tuple[Simulation, Dict[str, Any]]:
    city = City(cfg)
    sim  = Simulation(cfg=cfg, city=city, planner=planner)
    sim.run(steps)
    return sim, kpis(sim, cfg)


def build_arg_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument('--episodes', type=int, default=10)
    ap.add_argument('--steps',    type=int, default=2000, help='Steps per training episode')
    ap.add_argument('--weights-dir', type=str, default='dqn_weights')
    ap.add_argument('--save-every',  type=int, default=1000, help='(unused if internal autosave) retained for compatibility')
    ap.add_argument('--eval-every', type=int, default=0, help='Greedy evaluation cadence (episodes); 0 disables')
    ap.add_argument('--eval-episodes', type=int, default=1, help='Number of eval episodes when triggered')
    ap.add_argument('--early-stop-metric', choices=['pickup_rate','reward'], default='pickup_rate')
    ap.add_argument('--early-stop', type=float, default=None, help='Stop if moving average of chosen metric >= value')
    ap.add_argument('--ma-window', type=int, default=5)
    ap.add_argument('--log-jsonl', type=str, default='train_metrics.jsonl')
    ap.add_argument('--csv', type=str, default='training_metrics.csv')
    ap.add_argument('--seed',     type=int, default=None)
    ap.add_argument('--agents',   type=int, default=None, help='Override N_TRUCKS')
    ap.add_argument('--bins',     type=int, default=None, help='Override N_BINS')
    ap.add_argument('--bin-cap',  type=int, default=None)
    ap.add_argument('--planner',  choices=['graph','grid'], default='graph')
    ap.add_argument('--quiet', action='store_true')
    return ap


def augment_arg_parser(ap: argparse.ArgumentParser):
    ap.add_argument('--tensorboard', action='store_true', help='Enable TensorBoard logging if available')
    ap.add_argument('--tb-dir', type=str, default='runs', help='TensorBoard root directory')
    ap.add_argument('--resume', action='store_true', help='Resume from existing CSV & weights dir')
    ap.add_argument('--load-weights', type=str, default=None, help='Explicit weights dir to load before training')
    ap.add_argument('--sweep', type=str, default=None, help='Path to JSON specifying list of sweep configs')
    ap.add_argument('--sweep-key', type=str, default='name', help='Key in each sweep config used for naming run subdir')
    ap.add_argument('--tag', type=str, default='', help='Tag appended to run name / TB logdir')
    ap.add_argument('--validate-weights', action='store_true', help='Validate existing weight files before training')
    ap.add_argument('--strict-weights', action='store_true', help='Fail instead of warn on weight mismatch')
    return ap


def read_last_episode(csv_path: str) -> int:
    if not os.path.exists(csv_path):
        return 0
    try:
        last_ep = 0
        with open(csv_path, 'r') as f:
            rdr = csv.reader(f)
            header = next(rdr, None)
            epi_idx = None
            if header:
                for i,h in enumerate(header):
                    if h == 'episode':
                        epi_idx = i; break
            for row in rdr:
                if not row: continue
                try:
                    epv = int(row[epi_idx]) if epi_idx is not None else int(row[0])
                    last_ep = max(last_ep, epv)
                except Exception:
                    continue
        return last_ep
    except Exception:
        return 0


def validate_weights(dir_path: str, cfg: Dict[str, Any], strict: bool=False):
    expected = cfg.get('N_TRUCKS')
    if not os.path.isdir(dir_path):
        msg = f"[weights] directory '{dir_path}' not found"
        if strict:
            raise FileNotFoundError(msg)
        print(msg)
        return
    files = sorted(f for f in os.listdir(dir_path) if f.endswith('.pt'))
    count = len(files)
    if count == 0:
        print(f"[weights] no .pt files in '{dir_path}' (fresh training)")
        return
    if count != expected:
        msg = f"[weights] found {count} .pt files, expected {expected} for N_TRUCKS"
        if strict:
            raise RuntimeError(msg)
        else:
            print('[warn]', msg)
    else:
        print(f"[weights] validated {count}/{expected} agent weight files")


def run_training(args, base_cfg: Dict[str, Any], run_name: str, tb_root: Optional[str]=None):
    cfg = deepcopy(base_cfg)
    os.makedirs(cfg['DQN_WEIGHTS_DIR'], exist_ok=True)
    if getattr(args,'validate_weights',False):
        validate_weights(cfg['DQN_WEIGHTS_DIR'], cfg, strict=getattr(args,'strict_weights',False))
    best_metric_val = None
    best_dir = os.path.join(cfg['DQN_WEIGHTS_DIR'], 'best')
    os.makedirs(best_dir, exist_ok=True)

    jsonl_path = args.log_jsonl if hasattr(args,'log_jsonl') else 'train_metrics.jsonl'
    csv_path = args.csv if hasattr(args,'csv') else 'training_metrics.csv'

    resume_start = 1
    if args.resume:
        # detect last episode index
        last_ep = read_last_episode(csv_path)
        if last_ep > 0:
            resume_start = last_ep + 1
            print(f"[resume] Detected existing metrics up to episode {last_ep}, resuming at {resume_start}")

    log_fh = open(jsonl_path, 'a', buffering=1) if jsonl_path else None
    csv_new = not os.path.exists(csv_path) or resume_start==1
    csv_fh = open(csv_path, 'a', newline='') if csv_path else None
    csv_writer = csv.writer(csv_fh) if csv_fh else None
    if csv_writer and csv_new:
        csv_writer.writerow(['episode','phase','steps','pickup_amount','pickup_events','overflows','dumps','km','reward','metric','ma','epsilon_avg','elapsed_sec','run'])

    writer = None
    if getattr(args,'tensorboard',False) and SummaryWriter:
        tb_dir = os.path.join(args.tb_dir, run_name)
        writer = SummaryWriter(log_dir=tb_dir)
        if not getattr(args,'quiet',False):
            print(f"[tb] Logging TensorBoard to {tb_dir}")
    elif getattr(args,'tensorboard',False) and not SummaryWriter:
        print('[tb] TensorBoard requested but torch.utils.tensorboard not available.')

    metrics_series: List[float] = []
    t0 = time.time()

    def extract_eps(sim_obj):
        eps_vals = []
        try:
            policy = getattr(sim_obj, 'policy', None) or getattr(sim_obj, 'rl', None)
            last_eps = getattr(policy, 'last_eps', {})
            for v in (last_eps or {}).values():
                if v is not None:
                    eps_vals.append(v)
        except Exception:
            pass
        return statistics.mean(eps_vals) if eps_vals else None

    for ep in range(resume_start, args.episodes+1):
        sim, stats = run_one(cfg, args.steps, args.planner)
        pickup_rate = stats['pickup_amount'] / max(1, args.steps/1000.0)
        metric_val = pickup_rate if args.early_stop_metric=='pickup_rate' else stats['reward']
        metrics_series.append(metric_val)
        ma = statistics.mean(metrics_series[-args.ma_window:]) if len(metrics_series) >= args.ma_window else None
        eps_avg = extract_eps(sim)
        elapsed = time.time()-t0
        if csv_writer:
            csv_writer.writerow([ep,'train',args.steps,stats['pickup_amount'],stats['pickup_events'],stats['overflows'],stats['dumps'],f"{stats['km']:.2f}",f"{stats['reward']:.2f}",f"{metric_val:.2f}" if metric_val is not None else '',f"{ma:.2f}" if ma is not None else '', f"{eps_avg:.3f}" if eps_avg is not None else '', f"{elapsed:.1f}", run_name])
        if log_fh:
            rec = {"episode": ep, "phase":"train", **stats, "pickup_rate": pickup_rate, "metric": metric_val, "ma": ma, "epsilon_avg": eps_avg, "steps": args.steps, "elapsed_sec": elapsed, "run": run_name}
            log_fh.write(json.dumps(rec)+'\n')
        if writer:
            writer.add_scalar('train/pickup_amount', stats['pickup_amount'], ep)
            writer.add_scalar('train/reward', stats['reward'], ep)
            writer.add_scalar('train/pickup_rate', pickup_rate, ep)
            if eps_avg is not None:
                writer.add_scalar('train/epsilon', eps_avg, ep)
            if ma is not None:
                writer.add_scalar(f'ma/{args.early_stop_metric}', ma, ep)
        if not getattr(args,'quiet',False):
            base = f"[{run_name}] ep {ep:03d}/{args.episodes} pickups_amt={stats['pickup_amount']:.1f} events={stats['pickup_events']} overflows={stats['overflows']} dumps={stats['dumps']} km={stats['km']:.1f} reward={stats['reward']:.1f} metric={metric_val:.2f}"
            if ma is not None:
                base += f" ma({args.ma_window})={ma:.2f}"
            if eps_avg is not None:
                base += f" eps={eps_avg:.3f}"
            print(base)
        # save best
        if best_metric_val is None or metric_val > best_metric_val:
            best_metric_val = metric_val
            try:
                import shutil
                for f in os.listdir(cfg['DQN_WEIGHTS_DIR']):
                    if f.endswith('.pt'):
                        shutil.copy2(os.path.join(cfg['DQN_WEIGHTS_DIR'], f), os.path.join(cfg['DQN_WEIGHTS_DIR'],'best', f))
                with open(os.path.join(cfg['DQN_WEIGHTS_DIR'],'best','best.json'),'w') as jf:
                    json.dump({'episode': ep, 'metric': metric_val, 'metric_type': args.early_stop_metric, 'run': run_name}, jf)
            except Exception:
                pass
        # evaluation
        if args.eval_every>0 and ep % args.eval_every==0:
            eval_cfg = deepcopy(cfg)
            eval_cfg['EPS_START']=0.0; eval_cfg['EPS_END']=0.0; eval_cfg['EPS_DECAY']=1.0
            eval_metrics = []
            for eidx in range(args.eval_episodes):
                esim, estats = run_one(eval_cfg, args.steps, args.planner)
                ep_rate = estats['pickup_amount'] / max(1, args.steps/1000.0)
                emetric = ep_rate if args.early_stop_metric=='pickup_rate' else estats['reward']
                eval_metrics.append(emetric)
                if csv_writer:
                    csv_writer.writerow([ep,'eval',args.steps,estats['pickup_amount'],estats['pickup_events'],estats['overflows'],estats['dumps'],f"{estats['km']:.2f}",f"{estats['reward']:.2f}",f"{emetric:.2f}",'','', f"{time.time()-t0:.1f}", run_name])
                if log_fh:
                    log_fh.write(json.dumps({'episode':ep,'phase':'eval',**estats,'eval_metric':emetric,'steps':args.steps,'run':run_name})+'\n')
                if writer:
                    writer.add_scalar('eval/metric', emetric, ep)
            if not getattr(args,'quiet',False) and eval_metrics:
                print(f"   [{run_name}] eval mean={statistics.mean(eval_metrics):.2f} ({args.early_stop_metric}) over {len(eval_metrics)} eps")
        # early stop
        if args.early_stop is not None and ma is not None and ma >= args.early_stop:
            if not getattr(args,'quiet',False):
                print(f"[{run_name}] early-stop: moving average {ma:.2f} >= {args.early_stop} ({args.early_stop_metric}) at episode {ep}")
            break
    if writer:
        writer.close()
    if log_fh: log_fh.close()
    if csv_writer: csv_fh.close()
    if not getattr(args,'quiet',False):
        print(f"[{run_name}] done best {args.early_stop_metric}={best_metric_val:.2f} | weights dir: {os.path.abspath(cfg['DQN_WEIGHTS_DIR'])}")


def main():
    from argparse import Namespace
    from math import inf
    ap = build_arg_parser()
    augment_arg_parser(ap)
    args = ap.parse_args()

    base_cfg = deepcopy(CONFIG)
    base_cfg['POLICY'] = 'dqn'
    base_cfg['DQN_WEIGHTS_DIR'] = getattr(args,'weights_dir','dqn_weights')
    if args.seed is not None:    base_cfg['SEED'] = int(args.seed)
    if args.agents is not None:  base_cfg['N_TRUCKS'] = int(args.agents)
    if args.bins is not None:    base_cfg['N_BINS'] = int(args.bins)
    if args.bin_cap is not None: base_cfg['BIN_CAPACITY'] = int(args.bin_cap)
    base_cfg.setdefault('EPS_START', 0.8)
    base_cfg.setdefault('EPS_END', 0.05)
    base_cfg.setdefault('EPS_DECAY', 0.997)

    def apply_weights_load(cfg, load_path):
        # DQN agents load automatically at Simulation init if weights exist; we rely on same dir.
        if load_path:
            cfg['DQN_WEIGHTS_DIR'] = load_path
            if not os.path.isdir(load_path):
                print(f"[warn] load-weights dir '{load_path}' not found; continuing without preload")

    if args.sweep:
        # Expect JSON: list of objects overriding subset of cfg/args fields.
        with open(args.sweep,'r') as sf:
            sweep_items = json.load(sf)
        if not isinstance(sweep_items, list):
            print('[error] sweep file must contain a JSON list'); return
        for item in sweep_items:
            # build run-specific args clone
            run_cfg = deepcopy(base_cfg)
            run_args = deepcopy(args)
            # override config keys
            for k,v in item.items():
                if k in run_cfg:
                    run_cfg[k]=v
            # also allow episodes/steps override
            if 'episodes' in item: run_args.episodes = int(item['episodes'])
            if 'steps' in item: run_args.steps = int(item['steps'])
            tag_val = item.get(args.sweep_key, f"run_{int(time.time())}")
            if args.tag:
                tag_val = f"{tag_val}_{args.tag}"
            # unique weights dir per run
            run_cfg['DQN_WEIGHTS_DIR'] = os.path.join(base_cfg['DQN_WEIGHTS_DIR'], tag_val)
            apply_weights_load(run_cfg, args.load_weights)
            run_training(run_args, run_cfg, run_name=tag_val, tb_root=args.tb_dir)
    else:
        apply_weights_load(base_cfg, args.load_weights)
        run_name = 'main'+(f"_{args.tag}" if args.tag else '')
        run_training(args, base_cfg, run_name=run_name, tb_root=args.tb_dir)


if __name__ == '__main__':
    main()
