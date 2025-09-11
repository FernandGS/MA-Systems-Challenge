# Unity Hybrid Simulation Exporter

A self-contained, production-ready hybrid of your `ced/` city-graph simulator and `agen_simulation.py` grid exporter. It generates a Unity-friendly SimData JSON with:

- Road-respecting shortest paths (Dijkstra on city graph) OR grid A\* fallback
- Auction-based dispatch (no two trucks for the same bin); explicit ASSIGN/SERVICE/DUMP/RECHARGE/OVERFLOW events
- Continuous operation: bins fill stochastically; trucks pick up, dump at depot when full/low energy, and continue
- Unity export: agents with per-frame pathObj, bins with initial/remaining, events, metrics

## Features

- Two planners:
  - Graph planner (default): uses `WAYPOINTS`/`ROADS` to plan road-constrained routes
  - Grid planner (optional): A\* on an NxN grid with optional PNG mask (streets/sidewalks), with dilation/Manhattan fallback
- Robust assignment:
  - Greedy auction with urgency and fill-priority
  - Assign-hold and route-freeze windows to avoid churn
  - Pre-check reachability before locking a bin; depot redirection when needed
- Export compatible with your Unity player: `grid`, `agents`, `bins`, `events`, `metrics`

### New: Negotiation + RL + Separation

- Negotiation (contract-net style): bins announce when urgent/near-full; trucks bid; lowest cost wins (`negotiation.py`).
- RL layer: if `POLICY="dqn"`, a Q-learning manager proposes actions per-truck (depot/bin/idle). If PyTorch DQN is available it is used; otherwise, a lightweight tabular fallback is used (`rl_tabular.py`).
- Collision avoidance: after movement, trucks are separated by `SAFE_DISTANCE_M` to avoid overlap.

## Folder layout

- `config.py` — scenario knobs (map, costs, RL-agnostic sim parameters)
- `city.py` — graph world + Dijkstra planner; bin placement near sidewalks
- `grid_planner.py` — optional grid A\* planner with mask loading and dilation
- `agents.py` — Truck and Bin; motion, loading/unloading, costs, counters
- `dispatch.py` — auction (ASSIGN) + reachability checks
- `sim.py` — orchestrates fill → assign → move → costs → logs (frames/events)
- `export_unity.py` — CLI to run sim and produce `sim_run_pathObj.json` + `full_log.json`

## Install

Python 3.9+.

Optional packages:

- `Pillow` for PNG mask loading if using grid planner
- `matplotlib` if you later want to preview

## Run (PowerShell)

```powershell
# Default: graph planner
python .\unity_hybrid\export_unity.py --steps 3000

# Override scenario
python .\unity_hybrid\export_unity.py --trucks 5 --bins 18 --bin-cap 120 --steps 4000

# Use grid planner with a streets mask
python .\unity_hybrid\export_unity.py --planner grid --grid-size 150 --streets-mask ".\assets\streets.png" --invert-y --dilate-passes 2
```

Outputs:

- `sim_run_pathObj.json` — Unity SimData
- `full_log.json` — detailed run log

## JSON schema (Unity)

- grid: `{ width, height, depot: [x,y] }`
- agents: `[{ id, start: [x,y], pathObj: [{x,y}...], distance, collected, capacity }]`
- bins: `[{ id, pos: [x,y], initial, remaining }]`
- events: `[{ t, type, agent, bin, amount? }]` where type in { ASSIGN, SERVICE, DUMP, RECHARGE, OVERFLOW }
- metrics: `{ total_collected, avg_distance_per_agent, steps }`

## Constraints & assumptions

- Graph planner keeps trucks on roads strictly; bins are placed near road curbs
- Grid planner expects streets as passable=1; optional sidewalks snapping
- Depot is the only dump/recharge point
- Time step `DT` in seconds; wage/energy/maintenance accrue each step

## Troubleshooting

- Empty/flat paths in Unity: we export per-frame positions to pathObj; your Unity viewer should step frames directly
- Trucks stuck on grid: enable `--dilate-passes 2` or switch to graph planner (default)
- Over-assignment: auction enforces uniqueness and assignment hold windows
