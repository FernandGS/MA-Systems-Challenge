# Hybrid configuration (graph defaults; grid planner optional)

CONFIG = {
    # World (graph)
    "MAP_SIZE": (247.0, 232.2),
    "SEED": 42,
    # Road layout mode: 'manual' uses WAYPOINTS/ROADS below; 'grid' builds an orthogonal grid across MAP_SIZE
    "ROAD_LAYOUT": "manual",  # 'manual' | 'grid'
    # Either set rows/cols (blocks) or spacing; margin keeps roads inside the border
    "GRID_ROWS": 6,
    "GRID_COLS": 6,
    "GRID_SPACING": 20.0,
    "GRID_MARGIN": 10.0,
    # Waypoints are the intersection centers (Unity x,z -> sim x,y)
    # Rows ordered by increasing z (4.702, 64.705, 124.700, 164.691)
    "WAYPOINTS": [
        # z = 4.702
        (0.019, 4.702), (60.049, 4.702), (120.324, 4.702), (180.045, 4.702), (239.856, 4.702),
        # z = 64.705
        (0.019, 64.705), (60.049, 64.705), (120.324, 64.705), (180.045, 64.705), (239.856, 64.705),
        # z = 124.700
        (0.019, 124.700), (60.049, 124.700), (120.324, 124.700), (180.045, 124.700), (239.856, 124.700),
        # z = 164.691 (no columns at x=60.049 or x=180.045 here)
        (0.019, 164.691), (120.324, 164.691), (239.856, 164.691)
    ],
    # Connect neighbors horizontally (rows) and vertically (columns) where roads exist
    "ROADS": [
        # Row z=4.702
        [0,1],[1,2],[2,3],[3,4],
        # Row z=64.705
        [5,6],[6,7],[7,8],[8,9],
        # Row z=124.700
        [10,11],[11,12],[12,13],[13,14],
        # Row z=164.691
        [15,16],[16,17],
        # Column x≈0.019 (full height)
        [0,5],[5,10],[10,15],
        # Column x≈60.049 (up to z=124.700)
        [1,6],[6,11],
        # Column x≈120.324 (full height)
        [2,7],[7,12],[12,16],
        # Column x≈180.045 (up to z=124.700)
        [3,8],[8,13],
        # Column x≈239.856 (full height)
        [4,9],[9,14],[14,17]
    ],
    "DEPOT": (150.2066, 87.38293),  # updated discharging point
    "SIDEWALK_OFFSET_M": 2.0,
    # Approx half road width in Unity units to offset bins beyond roadway
    "ROAD_HALF_WIDTH": 4.0,
    # Extra margin beyond road+sidewalk to keep bins safely off-road (helps horizontal roads)
    "SIDEWALK_MARGIN_M": 0.75,
    # Minimum spacing between bin placements along any road segment (alias of BIN_MIN_SPACING)
    # If legacy config provides BIN_MIN_SPACING and this is absent, code maps it.
    "MIN_BIN_SEP_M": 6.0,
    # Additional clearance to push bins even further from curb (merged from BIN_CLEARANCE_M in alt configs)
    "BIN_CLEARANCE_M": 1.0,

    # Bins and trucks
    "N_BINS": 12,
    "BIN_CAPACITY": 100,
    "BIN_FILL_PER_STEP": (0, 1),
    "N_TRUCKS": 4,
    "TRUCK_CAPACITY": 300,
    "TRUCK_SPEED_MPS": 2.0,
    # Radius within which a truck is considered to have 'arrived' at a bin/depot.
    # Alternative tested value was 4.0 (more lenient). Keep tight value for precise snapping.
    "APPROACH_RADIUS_M": 1.2,

    # Energy and costs
    "ENERGY_MAX": 100.0,
    "ENERGY_PER_M": 0.06,
    "ENERGY_RESERVE_M": 30.0,
    "WAGE_PER_HOUR": 25.0,
    "ENERGY_EUR_PER_UNIT": 0.30,
    "MAINT_EUR_PER_KM": 0.06,
    "OVERFLOW_PENALTY_EUR": 500.0,
    "OUTAGE_PENALTY_EUR": 1000.0,

    # Dispatch priorities
    "URGENCY_HORIZON_S": 120,
    "OPPORTUNISTIC_FILL_FRAC": 0.60,

    # Anti-churn windows
    "ROUTE_FREEZE_STEPS": 6,
    "ASSIGN_HOLD_STEPS": 10,
    "DEPOT_LOCK_STEPS": 8,
    "NEAR_FULL_FRAC": 0.90,
    # Coverage bias: encourage trucks to pick farther bins when nothing urgent remains (0..1)
    "COVERAGE_BIAS": 0.8,
    # After a bin is serviced, wait this many seconds before it can be assigned again
    "SERVICE_COOLDOWN_S": 300.0,

    # Time
    "DT": 1.0,
    "STEPS_PER_DAY": 1200,

    # Policy: 'auction' (default) or 'dqn'
    "POLICY": "auction",
    # DQN candidate set size
    "DQN_K_CANDS": 6,
    # Reward shaping (tuned for stability)
    "RL_REWARD_PICKUP": 0.1,     # per unit collected
    "RL_REWARD_DUMP": 2.0,
    "RL_REWARD_OVERFLOW": -20.0,
    "RL_COST_PER_KM": 10.0,
    # DQN hyperparameters (tuned)
    "HIDDEN": 256,
    "LR": 5e-4,
    "GAMMA": 0.99,
    "EPS_START": 0.8,
    "EPS_END": 0.05,            # minimum epsilon
    "EPS_DECAY": 0.997,         # multiplicative per update
    "BUFFER_SIZE": 100_000,
    "BATCH_SIZE": 128,
    "TAU": 0.005,
    # Advanced DQN toggles
    "DQN_DUELING": True,
    "DQN_DOUBLE": True,
    "DQN_PRIORITIZED": False,
    "PRIORITY_ALPHA": 0.6,
    "PRIORITY_BETA": 0.4,
    "PRIORITY_BETA_INC": 1e-5,
    # Persistence
    "DQN_WEIGHTS_DIR": "dqn_weights",  # folder relative to repo root
    "DQN_SAVE_EVERY_STEPS": 200,        # save frequency during sim (steps)

    # Turn behavior
    "UTURN_PENALTY": 200.0,            # additional cost for immediate backtracking; larger -> less likely
    "FORBID_UTURN_IF_ALTERNATIVE": True, # if True, will forbid backtracking when any neighbor alternative exists

    # Lane & spacing controls
    "ENABLE_LANES": True,          # if true apply directional lateral offset so trucks keep to side
    "LANE_OFFSET_M": 1.2,          # lateral offset magnitude from road center
    "MIN_FOLLOW_GAP_STEPS": 1,     # minimum frame gap between trucks occupying same node sequence
    "ANTI_TAILGATE_EXTRA_HOLD": 1, # additional hold steps inserted to enforce gap

    # Grid planner (optional)
    "GRID_SIZE": 150,
    "STREETS_MASK_PNG": None,
    "STREETS_MASK_THRESHOLD": 0.5,
    "STREETS_MASK_INVERT_Y": False,
    "DILATE_PASSES": 2,
}
