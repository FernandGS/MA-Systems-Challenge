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
    "TRUCK_CAPACITY": 100,
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
    # Bins predicted to overflow within this horizon count as "urgent".
    "URGENCY_HORIZON_S": 120,
    # Collect bins already above this fill fraction even if not yet urgent.
    "OPPORTUNISTIC_FILL_FRAC": 0.60,

    # ---------- RL ----------
    "DT": 1.0,
    "STEPS_PER_DAY": 600,
    "GAMMA": 0.9266691575152085,
    "LR": 0.00505982903772765,
    "EPS_START": 1.0,
    "EPS_END": 0.08485580666713127,
    "EPS_DECAY": 0.998863940629782,
    "BUFFER_SIZE": 50000,
    "BATCH_SIZE": 32,
    "TARGET_UPDATE": 100,

    # ---------- export ----------
    "JSON_EXPORT_PATH": "sim_day.json"
}
