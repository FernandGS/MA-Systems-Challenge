# config.py
# Global parameters for simulation, training, and optimization

CONFIG = {
    # ---------- run modes ----------
    "RUN_SIM": True,
    "RUN_VIZ": True,
    "RUN_EXPORT": True,
    "RUN_DQN_TRAIN": False,
    "RUN_HYPEROPT": False,

    # ---------- world ----------
    "MAP_SIZE": (220.0, 160.0),
    "SEED": 42,
    "WAYPOINTS": [
        (20,130),(60,130),(100,130),(160,120),
        (30,85),(95,125),(150,75),
        (30,40),(100,35),(160,30),(95,80)
    ],
    "ROADS": [
        [0,1],[1,2],[2,3],
        [0,4],[4,6],[6,2],
        [4,8],[8,9],[9,10],
        [6,10],[10,7],[7,3],
        [9,7]
    ],
    "DEPOT": (150, 75),
    "SIDEWALK_OFFSET_M": 2.0,
    "N_BINS": 8,
    "BIN_CAPACITY": 100,

    # ---------- bin dynamics ----------
    "BIN_FILL_PER_STEP": (0, 1),
    "SERVICE_TIME_S": 25.0,

    # ---------- trucks ----------
    "N_TRUCKS": 4,
    "TRUCK_CAPACITY": 300,
    "TRUCK_SPEED_MPS": 6.0,
    "TRUCK_ACC_MPS2": 2.5,
    "TRUCK_DEC_MPS2": 4.0,
    "TRUCK_RADIUS_M": 1.5,
    "SAFE_GAP_M": 3.0,
    "ENERGY_MAX": 100.0,
    "ENERGY_PER_M": 0.06,
    "ENERGY_RESERVE_M": 30.0,
    "APPROACH_RADIUS_M": 3.0,

    # ---------- costs (€) ----------
    "WAGE_PER_HOUR": 25.0,
    "ENERGY_EUR_PER_UNIT": 0.30,
    "MAINT_EUR_PER_KM": 0.06,
    "OVERFLOW_PENALTY_EUR": 500.0,
    "OUTAGE_PENALTY_EUR": 1000.0,

    # ---------- negotiation / dispatch ----------
    "URGENCY_HORIZON_S": 120,
    "OPPORTUNISTIC_FILL_FRAC": 0.60,

    # ---------- RL ----------
    "DT": 1.0,
    "STEPS_PER_DAY": 600,
    "GAMMA": 0.9609734014656335,
    "LR": 0.0005084369486447114,
    "EPS_START": 1.0,
    "EPS_END": 0.03711065334643615,
    "EPS_DECAY": 0.9917886334001346,
    "BUFFER_SIZE": 50000,
    "BATCH_SIZE": 128,
    "TARGET_UPDATE": 100,
    "REWARD_SCALE": 0.01,
    "MAX_PENALTIES_PER_TICK": 8,

    # ---------- anti-churn & batching (NEW, conservative defaults) ----------
    # How long to keep following a planned route before WAIT is honored
    "ROUTE_FREEZE_STEPS": 6,
    # Minimum steps after an assignment before the truck is considered idle again
    "ASSIGN_HOLD_STEPS": 10,
    # After deciding to go depot, keep intent for this many steps
    "DEPOT_LOCK_STEPS": 8,
    # Auto-plan depot when load >= NEAR_FULL_FRAC * capacity (or low energy)
    "NEAR_FULL_FRAC": 0.90,

    # ---------- export ----------
    "JSON_EXPORT_PATH": "sim_day.json"
}
