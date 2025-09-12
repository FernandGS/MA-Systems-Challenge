# Waste Collection Hybrid Simulation

A compact hybrid simulation platform that models multi-agent waste collection on a 2D road network and exposes results to a Unity visualizer. It includes:

- A FastAPI server that runs simulations and returns Unity-friendly JSON payloads
- A Python simulation engine with auction/market policies and optional DQN RL policy
- A training script and saved DQN weights management
- Rich local visualization tools and analytics utilities
- Unity C# scripts to render and animate the scenario from exported payloads

This README doubles as a user manual and an implementation guide. Commands are shown for Windows PowerShell.

## Highlights

- Simulation core with intersection arbitration, cell reservation, anti-swap, anti-tailgating, lane offsets, dwell handling, and optional grid/graph planners
- Policies: auction, market, and DQN (dueling/double with optional prioritized replay)
- REST API to run synchronous and asynchronous simulations, with defaults persistence and a small built-in UI
- Unity integration via a concise SimData JSON schema; supports compressed paths with dwell counts
- Analytics CLI to compute KPIs from saved payloads; CSV/JSON exports

## Prerequisites

- Python: 3.10–3.13 (3.11+ recommended). Create a venv and install `requirements.txt`. Install PyTorch if you plan to use/train the DQN policy.
- OS: Windows 10/11 (commands use PowerShell). Linux/macOS should work with equivalent shell commands.
- Unity: 2021.3 LTS or newer (tested up to 2023.x).
- GPU: Optional but recommended for RL training (PyTorch).
- Network: Port 8000 open locally for the FastAPI server.
- Unity assets: Create `Assets/StreamingAssets/` if using Local File mode.

## Quick Start

1. Create a Python environment and install dependencies

- PowerShell

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

1. Run the REST server (makes Unity integration easiest)

```powershell
uvicorn server_rest:app --host 127.0.0.1 --port 8000 --reload
```

Open <http://127.0.0.1:8000> to use the built-in control panel.

1. Try a one-off simulation via API

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/simulate?steps=300&num_agents=3&num_waste_locations=12" -Method GET | Out-Null
```

The latest payload is also saved to `saved_payloads/` if saving is enabled.

1. Optional: live Python viz (no Unity required)

```powershell
python -m unity_hybrid.viz_rich --steps 600 --mode live --policy auction --planner graph --ids
```

## Project Layout

- `server_rest.py` — FastAPI app with endpoints to run sims and serve Unity payloads
- `unity_hybrid/` — Simulation package (config, city/graph, agents, dispatch, RL policy, visualization)
- `dqn_agent.py` — DQN implementation (dueling/double, optional prioritized replay)
- `train_dqn.py` — CLI to train DQN weights and track metrics
- `analytics/` — Post-run analysis tools
- `saved_payloads/` — JSON payloads saved by the server
- `dqn_weights/` — Stored weights, with `best/` subdirs and optional `live/`

## Architecture Overview

- City and routing: `unity_hybrid.city` builds either a manual road graph or a generated grid. It includes waypoint search (Dijkstra) with no-immediate-backtracking and U-turn discouragement. Bins are placed with curb positions and spacing constraints.
- Agents: `unity_hybrid.agents` defines `Truck` and `BinObj`. Trucks execute movement with lane offsetting, energy/load tracking, depot docking, and dwell after service.
- Dispatch policies: `unity_hybrid.dispatch`
  - `auction()`: greedy matching with urgency, coverage bias, repulsion, and anti-tailgating
  - `market()`: multi-round contract-net style bidding
- RL: `unity_hybrid.rl_policy.DQNManager` integrates with `dqn_agent.DQNAgent` when `POLICY='dqn'`. Observations are constructed from truck state and top-k candidate bins.
- Simulation loop: `unity_hybrid.sim.Simulation` runs per-step: fill bins, assign targets, move trucks with spacing and intersection arbitration, handle pickups/dumps/recharge, log frames/events, and optionally learn for DQN.
- Export to Unity: `unity_hybrid.export_unity` and server helpers build `SimData` containing agents, bins, events, metrics. Server version road-constrains tracks using event stitching and scheduling to avoid same-cell conflicts and edge swaps.
- Unity C#: `SimulationPlayer_PathObj.cs` loads SimData, spawns bins/trucks, animates with lane offsets and intersection handling. `CameraFollowCycler.cs` provides simple camera control; `TruckAvoidanceBubble.cs` nudges nearby trucks laterally.

## Running the REST API
Start the server:

```powershell
uvicorn server_rest:app --host 127.0.0.1 --port 8000 --reload
```

Open the homepage to adjust defaults and run a preview: <http://127.0.0.1:8000>

Key endpoints:

- `GET /health` — liveness probe
- `GET /config` — base configuration template
- `GET /defaults` — current server-side defaults
- `POST /defaults` — update defaults and auto-run; returns `result` and `meta`
- `GET /preview` — quick 60-step run using current defaults
- `POST /simulate` — run with overrides (honors body and query params)
- `GET /simulate` — same as above but with query params only
- `POST /simulate_async` — start background job; then poll `GET /job_status` and read `GET /last_result`
- `GET /schema` — JSON schema for Unity payload

Example POST (PowerShell):

```powershell
$body = @{ steps=1200; num_agents=4; num_waste_locations=36; policy='auction' } | ConvertTo-Json
Invoke-RestMethod -Uri "http://127.0.0.1:8000/simulate" -Method POST -ContentType 'application/json' -Body $body | Out-File -FilePath out.json -Encoding utf8
```

Saved payloads: By default, responses are written to `saved_payloads/`. Toggle with env vars:

- `SIM_SAVE_PAYLOAD=0` to disable
- `SIM_SAVE_DIR=...` to change folder

## Unity Integration

You can load either a local JSON file or fetch from the REST server.

Data schema highlights (`GET /schema`):

- `grid`: `{ width, height, depot }`
- `agents`: array with `{ id, start, pathObj, [pathDwell], distance, collected, capacity, [floatPath], [rawPlannedPath], [laneDebug] }`
- `bins`: array with `{ id, pos, initial, remaining }`
- `events`: `ASSIGN`, `SERVICE`, `DUMP`, `RECHARGE`, `OVERFLOW`, plus optional `INTERSECTION_WAIT`, `UTURN_BLOCK`
- `metrics`: `{ total_collected, avg_distance_per_agent, negotiation_messages, intersection_waits, uturn_blocks, steps }`

Unity usage (`SimulationPlayer_PathObj.cs`):

- Supports reading from disk or `http(s)` URLs
- Remote GET: `http://127.0.0.1:8000/simulate?steps=1200&num_agents=4&num_waste_locations=36`
- Remote POST with JSON body; also supports query-string overrides
- Lane offset application, intersection arbitration, dwell playback, optional path reconstruction, gizmos

### Unity Setup

Follow these steps to wire a Unity scene to this simulator. The scripts support Unity 2020.3+ (tested up to 2023.x). No extra Unity packages are required.

1. Create or open a Unity 3D project

- Recommended: Linear color space, default 3D template.

1. Import the C# scripts

- Copy these files into your Unity project under `Assets/Scripts/`:
  - `SimulationPlayer_PathObj.cs`
  - `CameraFollowCycler.cs`
  - `TruckAvoidanceBubble.cs`

1. Prepare prefabs and tags

- Create a `TruckPrefab`:
  - A simple cube/capsule works; forward should be +Z for natural rotation.
  - Add a tag named `Truck` and assign it to the prefab (used by `CameraFollowCycler`).
  - Optionally add `TruckAvoidanceBubble` to tune `radius`/`maxNudge`. If omitted, it is auto-added at runtime with defaults.
- Create a `BinPrefab`:
  - Any small mesh; orientation can be adjusted as needed. Spawning rotation defaults to `Quaternion.Euler(0,0,-90)` in code; tweak prefab rotation if needed.

1. Add the Simulation Player

- Create an empty GameObject named `SimulationController` and add `SimulationPlayer_PathObj`.
- Assign `TruckPrefab` and `BinPrefab` references.
- Important fields to set:
  - `cellSize`: Size in world units per grid cell (1.0 means 1 unit = 1 cell).
  - `worldOrigin`: World-space origin for grid (shift to align your map).
  - `stepDuration`: Seconds per simulation step (0.1–0.2 works well).
  - `smoothLerp`: Enable for smooth motion between steps.
  - `rotationSpeed`: Degrees/second yaw rotation toward travel direction.
  - Lane offsets:
    - `applyLaneOffsets`: Keep trucks from overlapping exactly.
    - Set `horizontalLaneOffset` and `verticalLaneOffset` (meters). If `horizontalLaneOffset` is 0, it can derive from vertical when `adaptHorizontalIfUnset` is true.
    - `trafficSide`: Choose Right or Left.
    - `clampOffsets` and per-axis maxes to avoid sidewalk spill.
    - `splitHorizontalWhenZero` and `horizontalZeroSplit` optionally split opposing horizontal flows even when offset is 0.
  - Centering corrections:
    - `autoSuggestCentering`: Prints one-time hints in Console.
    - `horizontalCenterCorrection`/`verticalCenterCorrection`: Fine-tune centering along the perpendicular axis.
  - Intersection control:
    - `enableIntersectionControl`: Unity-side virtual traffic lights.
    - If your payloads include dwell (`schemaVersion >= 2` with `pathDwell`) and you trust Python timing, set `usePythonIntersection = true` to avoid double holding.
  - Dwell playback:
    - `useDwellIfAvailable`: Consume `pathDwell` to hold at nodes.
  - Path fallback:
    - `rebuildStaticPathsFromEvents`: If an agent path is static/invalid, reconstruct from events to avoid stalls.
  - Visual aids:
    - `drawLaneGizmos` and colors/sizes for on-scene debugging.

1. Choose data source: Local file or Remote API

- Local file mode:
  - Create `Assets/StreamingAssets/` in your Unity project.
  - Drop a payload file there named `sim_run_pathObj.json` (or change `jsonFileName`).
  - In `SimulationPlayer_PathObj`, disable `useRemote`.
  - Tip (PowerShell, run from the Python repo):

    ```powershell
    # Copy newest payload into a Unity project's StreamingAssets
    $src = Get-ChildItem .\saved_payloads\*.json | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    Copy-Item $src.FullName -Destination "C:\Path\To\Your\UnityProject\Assets\StreamingAssets\sim_run_pathObj.json"
    ```

- Remote API mode (recommended for iteration):
  - Run the Python server: `uvicorn server_rest:app --host 127.0.0.1 --port 8000 --reload`.
  - In `SimulationPlayer_PathObj`:
    - Enable `useRemote` and `autoRequestOnStart`.
    - Set `remoteUrl` to `http://127.0.0.1:8000/simulate`.
    - Choose request type:
      - `useGetRequest = true` to pass overrides via query string.
      - or `useGetRequest = false` to POST a compact JSON body.
    - Optional overrides: `seedOverride`, `numAgentsOverride`, `numBinsOverride`, `stepsOverride`, `planner`, `truckSpeed`, `returnSpeedFactor`.

1. Add a camera follower (optional but handy)

- Add `CameraFollowCycler` to your main camera.
- Configure:
  - `truckTag = "Truck"` to auto-discover truck instances.
  - `autoDiscover = true`, `autoRefresh = true` to react to runtime spawns.
  - Tweak `offset`, `followSmooth`, `lookSmooth`.
- Controls:
  - Next/Prev target: `]` / `[` keys.
  - Toggle auto-cycle: Spacebar.
  - Works with both old and new Input Systems.

1. Press Play

- If using Remote mode, the scene will fetch and spawn automatically.
- If using Local mode, ensure the JSON exists in `StreamingAssets`.

Advanced tuning cheatsheet:

- Horizontal lane overlap on narrow roads:
  - Increase `horizontalLaneOffset`, or enable `splitHorizontalWhenZero` and set `horizontalZeroSplit` (e.g., 0.5).
- Trucks bleed off sidewalks on vertical/horizontal streets:
  - Use `clampOffsets` with `horizontalMaxOffset`/`verticalMaxOffset`.
- Systematic mis-centering of horizontal movers:
  - Check Console for `[CenterSuggest]` and set `horizontalCenterCorrection` accordingly.
- Unity-side collisions at intersections despite Python dwell:
  - Set `usePythonIntersection = true` and ensure `useDwellIfAvailable = true` (payload must be `schemaVersion >= 2`).
- Roads visually offset from grid center:
  - Adjust `worldOrigin` and `cellSize` to match your art scale. A plane of size `grid.width * cellSize` by `grid.height * cellSize` works as a ground.

## Local Visualization (Python)

`unity_hybrid/viz_rich.py` renders the simulation without Unity.

Examples:

```powershell
python -m unity_hybrid.viz_rich --steps 600 --mode live --policy auction --planner graph --ids
python -m unity_hybrid.viz_rich --steps 400 --mode playback --policy dqn --ids
```

## DQN Training

The RL policy is optional. To enable it in sim/API, set `POLICY='dqn'` and ensure PyTorch is installed.

Train weights:

```powershell
python train_dqn.py --episodes 20 --steps 2000 --agents 3 --bins 15 --weights-dir dqn_weights --planner graph --tensorboard
```

Outputs:

- Weights in `dqn_weights/` (auto-saved by agents; best copy in `dqn_weights/best/`)
- Metrics in `training_metrics.csv` and `train_metrics.jsonl`
- Optional TensorBoard logs under `runs/`

Sweeps:

```powershell
python train_dqn.py --sweep sweep.json --episodes 12 --steps 2000 --planner graph
```

Server auto-weights:

- When `POLICY='dqn'`, the server will scan `DQN_WEIGHTS_DIR` for a `best/` with `best.json` and copy `.pt` files into a `live/` folder for continued training, else it prepares a fresh directory.

Freeze DQN for deterministic evaluation (viz):

```powershell
python -m unity_hybrid.viz_rich --policy dqn --freeze-dqn --weights-dir dqn_weights
```

## Analytics

Analyze a saved payload and export metrics:

```powershell
python -m analytics.analyze_sim --latest --search-dir saved_payloads --pattern *.json --json metrics.json --csv agents.csv --bin-csv bins.csv
```
Or target a specific file:

```powershell
python -m analytics.analyze_sim --input saved_payloads\simulate_post_steps12000_agents4_bins36_YYYYmmdd_HHMMSS.json
```

## Configuration Reference (selected)

From `unity_hybrid/config.py` (defaults can be overridden via API or code):

- Core: `SEED`, `MAP_SIZE`, `N_TRUCKS`, `N_BINS`, `BIN_CAPACITY`, `TRUCK_SPEED_MPS`, `SIDEWALK_OFFSET_M`
- Dispatch: `OPPORTUNISTIC_FILL_FRAC`, `URGENCY_HORIZON_S`, `COVERAGE_BIAS`, `SERVICE_COOLDOWN_S`
- Routing: `ENABLE_LANES`, `LANE_OFFSET_M`, `TRAFFIC_SIDE`, `MIN_FOLLOW_GAP_STEPS`, `ANTI_TAILGATE_EXTRA_HOLD`
- RL: `POLICY`, `DQN_WEIGHTS_DIR`, `DQN_TRAIN_ENABLED`, `EPS_START`, `EPS_END`, `EPS_DECAY`, `DQN_K_CANDS`, `DQN_SAVE_EVERY_STEPS`, reward/cost shaping keys
- Planner: `planner` can be `graph` or `grid` (API params `planner=...`)

## File-by-File Guide

- `server_rest.py`: REST API; builds road-constrained paths from events, validates payloads, manages async jobs, persists defaults, and saves payloads.
- `unity_hybrid/config.py`: Central config dictionary with map, lanes, dispatch, RL, and routing parameters.
- `unity_hybrid/city.py`: Road network and routing; nearest waypoint, Dijkstra with U-turn avoidance; bin placement with spacing/margins.
- `unity_hybrid/agents.py`: `Truck` and `BinObj`; movement step, energy/load tracking, dwell, depot docking, lane offsetting.
- `unity_hybrid/dispatch.py`: `auction()` and `market()` assignment policies.
- `unity_hybrid/rl_policy.py`: `DQNManager` wrapper that constructs observations, chooses actions, stores transitions, and trains via `dqn_agent.DQNAgent`.
- `dqn_agent.py`: Dueling/Double DQN with optional prioritized replay; replay buffer; epsilon scheduling; soft target updates.
- `unity_hybrid/sim.py`: Simulation driver; per-step loop, intersection arbitration, reservation, events and frames logging, and RL hooks.
- `unity_hybrid/export_unity.py`: Standalone exporter that can write SimData and full logs to disk.
- `unity_hybrid/viz_rich.py`, `unity_hybrid/run_viz_custom.py`: Rich matplotlib-based live/playback visualization utilities.
- `Unity C#`: `SimulationPlayer_PathObj.cs`, `CameraFollowCycler.cs`, `TruckAvoidanceBubble.cs` implement the Unity side.
- `analytics/analyze_sim.py`: KPI extraction and CSV/JSON exporters.
- `train_dqn.py`: DQN training CLI, sweep support, metrics logging, best weights management.

## Troubleshooting

- API returns but Unity movement looks diagonal or off-road
  - Prefer server payloads (they build road-constrained paths from events and schedule conflicts). Ensure Unity’s `applyLaneOffsets` and `rebuildStaticPathsFromEvents` options match your data.
- No files in `saved_payloads/`
  - Call `GET /debug_force_save` and check response; verify `SIM_SAVE_PAYLOAD` and write permissions.
- DQN policy not active
  - Install PyTorch and set `POLICY='dqn'`. For viz, point to `--weights-dir` or allow server `DQN_WEIGHTS_DIR` auto-prep.
- Python 3.13 and AgentPy
  - `agentpy` is optional and only installed for Python < 3.13. The sim works without it.

## Development Notes

- Code style favors minimal external dependencies; keep changes focused and respect existing APIs.
- When extending the API schema, update `GET /schema` and Unity’s reader accordingly.
- For new RL features, prefer adding toggles to `config.py` and wiring through `DQNManager`.

## License

This repository contains academic coursework and supporting code.
