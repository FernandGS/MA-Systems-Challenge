# Waste Collection Multi‑Agent Simulation + Unity Export (Hybrid)

This repo provides a streamlined, self‑contained exporter in `unity_hybrid/` that generates Unity‑ready JSON. It preserves your original routing/dispatch logic and optionally integrates agentpy for scheduling and data collection.

## What to use now

- `unity_hybrid/export_unity.py` — Run this to generate `sim_run_pathObj.json` (Unity) and `full_log.json` (detailed log)
- `SimulationPlayer_PathObj.cs` — Unity MonoBehaviour that loads and animates the JSON

Optional (only if you want agentpy features)

- `unity_hybrid/ap_model.py` — AgentPy model wrappers used automatically if agentpy is installed

Legacy (safe to archive if you won’t use them)

- `ced/`, `unity_runner/`, `agen_simulation.py`, `server.py` — Older pipelines and wrappers

## Requirements

- Python 3.9+
- No mandatory dependencies. Optional: `agentpy` (to run via agentpy scheduler), `pillow` (for future grid mask loading)

PowerShell (optional venv)

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install agentpy  # optional
```

## Generate JSON for Unity

From the repo root:

```powershell
python .\unity_hybrid\export_unity.py --steps 300 --trucks 3 --bins 12
```

Flags

- `--steps` Number of simulation steps
- `--trucks` Number of trucks
- `--bins` Number of bins
- `--bin-cap` Bin capacity (default from config)
- `--planner graph|grid` Planner choice (graph is default; grid is stubbed for future mask support)

Outputs

- `sim_run_pathObj.json` — Unity SimData (grid, agents with pathObj, bins, events, metrics)
- `full_log.json` — Full frames+events log + cfg

## Unity usage

1) Copy `sim_run_pathObj.json` to your Unity project under `Assets/StreamingAssets/`
2) Add `SimulationPlayer_PathObj` (provided in `SimulationPlayer_PathObj.cs`) to a GameObject
3) Assign `TruckPrefab` and `BinPrefab` in the Inspector
4) Press Play to see trucks/bins animate

Notes

- The script smoothly interpolates position/rotation each step and prints basic KPIs.
- If incoming paths are static, it can rebuild them from events (toggle `rebuildStaticPathsFromEvents`).

## REST API (for Unity remote mode)

Run the server (PowerShell):

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
uvicorn server_rest:app --host 0.0.0.0 --port 8000 --reload
```

Endpoints

- GET `/health` → `{ ok: true }`
- GET `/config` → current default CONFIG
- GET `/defaults` → server-side defaults that are applied when params are omitted
- POST `/defaults` → set server-side defaults (JSON body; all fields optional)
- POST `/simulate` → returns Unity SimData JSON
- GET `/simulate` → same result using query params (handy for browsers)
- GET `/schema` → SimData JSON schema
- GET `/` → Simple index page with links and a form to call /simulate

Example POST body (all fields optional):

```json
{ "seed": 42, "num_agents": 4, "num_waste_locations": 15, "bin_capacity": 120, "steps": 1200, "planner": "graph", "truck_speed": 6.0 }
```

Example GET (browser):

```text
http://127.0.0.1:8000/simulate?steps=120&num_agents=3&num_waste_locations=10&truck_speed=2.0
```

Unity: set `useRemote = true` and `remoteUrl = http://127.0.0.1:8000/simulate` in `SimulationPlayer_PathObj`.

Tip: Use the index page “Set server defaults” card or POST `/defaults` so Unity can keep a constant URL without querystring. `/simulate` will apply saved defaults unless the request explicitly overrides them.

## How agentpy is used here (optional)

If `agentpy` is installed, the exporter will run via a thin agentpy model (`WasteSimModel`) that:

- Wraps each Truck/Bin as `ap.Agent` instances
- Uses `AgentList.step()` as the scheduler but delegates all behavior to your existing classes
- Records simple series (events_total) and reports totals at the end

Behavior of the simulation and the JSON output remains the same either way.

## JSON shape (for reference)

```text
{
  "grid": {"width":int, "height":int, "depot":[x,y]},
  "agents": [ {"id":int, "start":[x,y], "pathObj":[{"x":int,"y":int}...], "distance":int, "collected":int, "capacity":int }, ... ],
  "bins": [ {"id":int, "pos":[x,y], "initial":int, "remaining":int }, ... ],
  "events": [ {"t":int, "type":"ASSIGN|SERVICE|DUMP|RECHARGE|OVERFLOW", "agent"?:int, "bin"?:int, "amount"?:int }, ... ],
  "metrics": {"total_collected":int, "avg_distance_per_agent":float, "negotiation_messages":int, "steps":int }
}
```

## Cleaning up (optional)

To keep this repo lean for Unity export, you can delete or archive: `ced/`, `unity_runner/`, `agen_simulation.py`, `server.py`, `__pycache__/`, `sim_run_pathObj.json`, `full_log.json` (regenerate anytime).

---
Updated on 2025‑08‑31.

## Advanced DQN (Optional)

Set `POLICY="dqn"` to enable the deep RL dispatcher (`DQNManager`). Key config flags in `config.py`:

| Flag | Description |
|------|-------------|
| `DQN_DUELING` | Enable dueling network heads (value + advantage) |
| `DQN_DOUBLE` | Use Double DQN target selection |
| `DQN_PRIORITIZED` | Turn on prioritized replay buffer |
| `PRIORITY_ALPHA` | Priority exponent (0 = uniform) |
| `PRIORITY_BETA` | Initial importance-sampling beta |
| `PRIORITY_BETA_INC` | Increment per update until 1.0 |
| `DQN_K_CANDS` | Number of candidate bins encoded in state |
| `RL_REWARD_*` | Reward shaping terms |

Weights auto‑save to `DQN_WEIGHTS_DIR` every `DQN_SAVE_EVERY_STEPS` steps. One file per truck (independent learners).

### Training Script

`train_dqn.py` now supports:

- Evaluation episodes (`--eval-eps N`) with epsilon forced to 0 (greedy)
- Best checkpoint copying to `<weights>/best/`
- JSONL metrics logging (`--log-jsonl train_metrics.jsonl`)
- Moving average & early stopping (`--early-stop <pickup_rate>` per 1k steps, window `--ma-window`)

Example:

```powershell
python .\train_dqn.py --episodes 30 --steps 2500 --eval-eps 2 --early-stop 550 --ma-window 4 --weights-dir dqn_weights
```


Metrics JSON lines include: episode, pickups, overflows, pickup_rate, moving average, epsilon stats.

## Lane Separation & Anti‑Tailgating

To reduce visual overlap and unrealistic platooning:

| Config | Meaning |
|--------|---------|
| `ENABLE_LANES` | Apply lateral offset to each route waypoint so trucks keep to a side |
| `LANE_OFFSET_M` | Magnitude of lateral shift from road center |
| `MIN_FOLLOW_GAP_STEPS` | Minimum step gap before a second truck may start along an occupied first waypoint |
| `ANTI_TAILGATE_EXTRA_HOLD` | Extra enforced hold steps if a gap violation would occur |

When enabled, trucks get a small perpendicular offset; events `lane_assignment` record lane usage. If another truck just started moving from the same starting waypoint, a `tailgate_hold` event is emitted and the new truck waits (increasing `assign_hold_steps`).

## New Events

| Event | When |
|-------|------|
| `assign` | Truck assigned a bin or depot route |
| `pickup` | Trash collected from a bin (includes `amount`) |
| `drop` | Load dumped at depot |
| `recharge` | Energy restored at depot |
| `overflow` | Bin reached capacity during fill step |
| `lane_assignment` | Route created with lane offset applied |
| `tailgate_hold` | Assignment delayed to enforce follow gap |

These appear in `events` arrays for both exporter and training logs.

## Enabling / Disabling Features

- Turn off lanes or anti‑tailgating by setting `ENABLE_LANES=False` or reducing gap parameters to 0.
- Turn off advanced DQN features individually (e.g., `DQN_DUELING=False`).

---
Updated on 2025‑09‑07 (advanced DQN + lanes + anti‑tailgating + training tooling).
