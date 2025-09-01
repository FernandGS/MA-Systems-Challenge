# Hybrid configuration (graph defaults; grid planner optional)

CONFIG = {
    # World (graph)
    "MAP_SIZE": (220.0, 160.0),
    "SEED": 42,
    # Road layout mode: 'manual' uses WAYPOINTS/ROADS below; 'grid' builds an orthogonal grid across MAP_SIZE
    "ROAD_LAYOUT": "grid",  # 'manual' | 'grid'
    # Grid layout parameters (used when ROAD_LAYOUT == 'grid')
    # Either set rows/cols (blocks) or spacing; margin keeps roads inside the border
    "GRID_ROWS": 6,
    "GRID_COLS": 6,
    "GRID_SPACING": 20.0,
    "GRID_MARGIN": 10.0,
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

    # Bins and trucks
    "N_BINS": 12,
    "BIN_CAPACITY": 100,
    "BIN_FILL_PER_STEP": (0, 1),
    "N_TRUCKS": 4,
    "TRUCK_CAPACITY": 300,
    "TRUCK_SPEED_MPS": 2.0,
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

    # Grid planner (optional)
    "GRID_SIZE": 150,
    "STREETS_MASK_PNG": None,
    "STREETS_MASK_THRESHOLD": 0.5,
    "STREETS_MASK_INVERT_Y": False,
    "DILATE_PASSES": 2,
}
