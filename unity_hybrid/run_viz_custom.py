# run_viz_custom.py
from config import CONFIG
from city import City
from sim import Simulation
import viz_rich
import os

cfg = CONFIG.copy()
cfg.update({
    "SEED": 42,
    "N_TRUCKS": 4,
    "N_BINS": 36,
    "BIN_CAPACITY": 100,
    "TRUCK_SPEED_MPS": 2,
    "SIDEWALK_OFFSET_M": 8,
    "OPPORTUNISTIC_FILL_FRAC": 0.6,
    "URGENCY_HORIZON_S": 900,
    "COVERAGE_BIAS": 0.8,
    "SERVICE_COOLDOWN_S": 800,
    "POLICY": "dqn",
})

# Attempt to auto-use existing DQN weights if directory exists
weights_dir = os.path.join(os.path.dirname(__file__), '..', 'dqn_weights')
weights_dir = os.path.abspath(weights_dir)
if os.path.isdir(weights_dir) and any(f.endswith('.pt') or f.endswith('.pth') for f in os.listdir(weights_dir)):
    cfg['DQN_WEIGHTS_DIR'] = os.path.basename(weights_dir)
    # Switch to eval-only (no further training) if user wants deterministic replay
    cfg['DQN_TRAIN_ENABLED'] = True
    cfg['EPS_START'] = cfg.get('EPS_END', 0.05)
    cfg['EPS_DECAY'] = 1.0

city = City(cfg)
sim = Simulation(cfg=cfg, city=city, planner="graph")
viz_rich.live(sim, steps=12000, show_ids=True, interval_ms=0)