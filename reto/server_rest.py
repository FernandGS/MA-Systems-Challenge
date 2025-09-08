import os
import sys
import json
import threading
import time
from typing import Optional, Dict, Any
from copy import deepcopy

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, Response
import base64
from pydantic import BaseModel

# Make unity_hybrid importable regardless of cwd
ROOT = os.path.dirname(os.path.abspath(__file__))
HYB = os.path.join(ROOT, 'unity_hybrid')
if HYB not in sys.path:
    sys.path.insert(0, HYB)

# Import hybrid modules as top-level modules from the unity_hybrid folder
from config import CONFIG  # type: ignore
from city import City  # type: ignore
from sim import Simulation  # type: ignore
from ap_model import WasteSimModel, ap  # type: ignore

app = FastAPI(title="Waste Collection Hybrid API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# (load occurs after helper functions are defined)

# Server-side defaults so clients can use a constant URL (e.g., /simulate)
# and still get updated parameters set from the index page or API.
DEFAULT_OVERRIDES: Dict[str, Any] = {}
DEFAULTS_FILE = os.path.join(ROOT, 'server_defaults.json')
LAST_RESULT: Dict[str, Any] | None = None
LAST_RESULT_META: Dict[str, Any] = {}

# Async job state (single job)
JOB_STATE: Dict[str, Any] = {
    'id': None,
    'status': 'idle',  # idle|running|done|error
    'started_at': None,
    'finished_at': None,
    'steps': None,
    'planner': None,
    'config_snapshot': None,
    'error': None
}
_JOB_LOCK = threading.Lock()
_JOB_COUNTER = 0

def _load_defaults_from_disk():
    global DEFAULT_OVERRIDES
    try:
        if os.path.exists(DEFAULTS_FILE):
            with open(DEFAULTS_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, dict):
                    DEFAULT_OVERRIDES = data
    except Exception:
        DEFAULT_OVERRIDES = {}

def _save_defaults_to_disk():
    try:
        with open(DEFAULTS_FILE, 'w', encoding='utf-8') as f:
            json.dump(DEFAULT_OVERRIDES, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

# Load defaults saved from previous runs (now that helpers exist)
_load_defaults_from_disk()


class SimRequest(BaseModel):
    seed: Optional[int] = None
    num_agents: Optional[int] = None
    num_waste_locations: Optional[int] = None
    bin_capacity: Optional[int] = None
    steps: Optional[int] = None
    planner: Optional[str] = "graph"  # or "grid"
    truck_speed: Optional[float] = None
    return_speed_factor: Optional[float] = None  # accepted but unused
    sidewalk_offset: Optional[float] = None
    opportunistic_fill_frac: Optional[float] = None
    urgency_horizon_s: Optional[int] = None
    coverage_bias: Optional[float] = None
    service_cooldown_s: Optional[float] = None
    policy: Optional[str] = None  # 'auction' | 'dqn'


class DefaultsRequest(BaseModel):
    seed: Optional[int] = None
    num_agents: Optional[int] = None
    num_waste_locations: Optional[int] = None
    bin_capacity: Optional[int] = None
    steps: Optional[int] = None
    planner: Optional[str] = None
    truck_speed: Optional[float] = None
    sidewalk_offset: Optional[float] = None
    opportunistic_fill_frac: Optional[float] = None
    urgency_horizon_s: Optional[int] = None
    coverage_bias: Optional[float] = None
    service_cooldown_s: Optional[float] = None
    policy: Optional[str] = None  # 'auction' | 'dqn'


# Local helpers (copied/adapted from exporter) ---------------------------------

def _build_agent_paths_from_events(city: 'City', sim: 'Simulation'):
    # Build bin id -> curb/pos map
    bin_pos = {}
    for b in sim.bins:
        # prefer curb (on-road), else bin.pos
        pt = getattr(b, 'curb', None)
        if pt is None:
            try:
                pt = (float(b.pos[0]), float(b.pos[1]))
            except Exception:
                pt = city.depot
        bin_pos[b.id] = pt

    # Collect per-truck ordered targets from events
    events_by_truck: Dict[str, list] = {}
    for ev in sim.events:
        if ev.get('type') not in ('assign', 'pickup', 'drop'):
            continue
        tid = ev.get('truck')
        if not tid:
            continue
        events_by_truck.setdefault(tid, []).append(ev)

    starts: Dict[str, list] = {}
    tracks: Dict[str, list] = {}

    def _road_expand(p0: Dict[str, int], p1: Dict[str, int]):
        # Expand along axis-aligned segment; fallback to Manhattan if needed
        out = []
        x0, y0 = int(p0['x']), int(p0['y'])
        x1, y1 = int(p1['x']), int(p1['y'])
        if x0 == x1 or y0 == y1:
            while x0 != x1:
                x0 += 1 if x0 < x1 else -1
                out.append({"x": x0, "y": y0})
            while y0 != y1:
                y0 += 1 if y0 < y1 else -1
                out.append({"x": x0, "y": y0})
            return out
        # not axis-aligned: Manhattan fallback
        return _manhattan_expand(p0, p1)

    for rank, t in enumerate(sim.trucks):
        tid = t.tid
        # Start exactly at depot (no lateral offset to avoid diagonal artifacts)
        start = [int(round(city.depot[0])), int(round(city.depot[1]))]
        starts[tid] = start
        cur = (float(city.depot[0]), float(city.depot[1]))
        # Small stagger to reduce pile-ups at the depot: delay later trucks by a few frames
        pad = max(0, min(rank, 5)) * 2
        path_cells = [{"x": start[0], "y": start[1]}] * (1 + pad)

        # Sort events chronologically
        evs = sorted(events_by_truck.get(tid, []), key=lambda e: e.get('t', 0))
        # Build target list: assigned bin curbs in order; ignore duplicates in a row
        targets: list = []
        for ev in evs:
            et = ev.get('type')
            if et == 'assign' or et == 'pickup':
                bid = ev.get('bin')
                if bid in bin_pos:
                    tgt = bin_pos[bid]
                    if not targets or targets[-1] != tgt:
                        targets.append(tgt)
            elif et == 'drop':
                # If a drop happens, ensure depot is next target if not already last
                if not targets or targets[-1] != city.depot:
                    targets.append(city.depot)

        # Maintain previous waypoint index to avoid immediate U-turns when chaining targets
        prev_wp_idx = None
        cur_wp_idx = city.nearest_waypoint_idx(cur)

        def _append_route(cur_pt, tgt_pt):
            nonlocal prev_wp_idx, cur_wp_idx, path_cells
            route = city.plan_route(cur_pt, tgt_pt, prev_idx=prev_wp_idx) or [cur_pt, tgt_pt]
            if not route or route[-1] != tgt_pt:
                route = route + [tgt_pt]
            # Densify along each segment on the road
            # Seed with last cell
            last_cell = path_cells[-1]
            for i in range(1, len(route)):
                a = route[i-1]
                b = route[i]
                a_cell = {"x": int(round(a[0])), "y": int(round(a[1]))}
                b_cell = {"x": int(round(b[0])), "y": int(round(b[1]))}
                # Ensure continuity from previous
                if a_cell != last_cell:
                    seg = _road_expand(last_cell, a_cell)
                    if seg:
                        path_cells.extend(seg)
                        last_cell = path_cells[-1]
                seg2 = _road_expand(last_cell, b_cell)
                if seg2:
                    path_cells.extend(seg2)
                    last_cell = path_cells[-1]
            # Update waypoint indices for U-turn prevention on next leg
            if len(route) >= 2:
                prev_wp_idx = city.nearest_waypoint_idx(route[-2])
                cur_wp_idx = city.nearest_waypoint_idx(route[-1])
            else:
                prev_wp_idx = cur_wp_idx
                cur_wp_idx = city.nearest_waypoint_idx(route[-1])
            return route[-1] if route else cur_pt

        # Stitch routes to all targets
        for tgt in targets:
            cur = _append_route(cur, tgt)

        # Ensure return to depot
        if (int(round(cur[0])) != start[0]) or (int(round(cur[1])) != start[1]):
            _append_route(cur, city.depot)

        tracks[tid] = path_cells

    # Server-side per-step occupancy scheduler to avoid same-cell conflicts and edge swaps
    def _schedule_tracks(tracks_in: Dict[str, list]) -> Dict[str, list]:
        # Convert to tuple lists for easier comparison
        tkeys = list(tracks_in.keys())
        tracks_t = {k: [ (int(p["x"]), int(p["y"])) for p in v ] for k, v in tracks_in.items()}
        # Tie-break order: by numeric id parsed from tid (e.g., 'T3' -> 3), fallback to index
        def _tid_ord(tid: str, idx: int) -> int:
            try:
                return int(''.join(ch for ch in str(tid) if ch.isdigit()))
            except Exception:
                return idx
        torder = {tid: _tid_ord(tid, i) for i, tid in enumerate(tkeys)}

        changed = True
        safety = 0
        while changed and safety < 2000:
            safety += 1
            changed = False
            # Equalize lengths by padding last cell
            max_len = max((len(v) for v in tracks_t.values()), default=0)
            for tid in tkeys:
                if len(tracks_t[tid]) == 0 and max_len > 0:
                    # if empty, seed with city.depot
                    d = (int(round(city.depot[0])), int(round(city.depot[1])))
                    tracks_t[tid] = [d]
                while len(tracks_t[tid]) < max_len:
                    tracks_t[tid].append(tracks_t[tid][-1])

            # Iterate steps and resolve conflicts
            for i in range(1, max_len):
                # Build occupancy at this step
                occ = {}
                for tid in tkeys:
                    cell = tracks_t[tid][i]
                    occ.setdefault(cell, []).append(tid)

                # Same-cell conflicts: allow only the smallest order to move; others hold
                for cell, tids in list(occ.items()):
                    if len(tids) <= 1:
                        continue
                    # Determine winners by order (lowest goes)
                    tids_sorted = sorted(tids, key=lambda x: (torder.get(x, 0), x))
                    winner = tids_sorted[0]
                    for loser in tids_sorted[1:]:
                        # Insert hold at i for loser (repeat previous cell)
                        prev = tracks_t[loser][i-1]
                        if tracks_t[loser][i] != prev:
                            tracks_t[loser].insert(i, prev)
                            changed = True
                    if changed:
                        break  # restart pass due to index shifts
                if changed:
                    break

                # Edge swap conflicts: A@u->v and B@v->u
                # Collect step transitions
                trans = {}
                for tid in tkeys:
                    trans[tid] = (tracks_t[tid][i-1], tracks_t[tid][i])
                # Check pairs
                for a in range(len(tkeys)):
                    for b in range(a+1, len(tkeys)):
                        ta, tb = tkeys[a], tkeys[b]
                        u1, v1 = trans[ta]
                        u2, v2 = trans[tb]
                        if u1 != v1 and u2 != v2 and u1 == v2 and v1 == u2:
                            # Hold the higher order id
                            hold_tid = ta if torder[ta] > torder[tb] else tb
                            prev = tracks_t[hold_tid][i-1]
                            if tracks_t[hold_tid][i] != prev:
                                tracks_t[hold_tid].insert(i, prev)
                                changed = True
                                break
                    if changed:
                        break
                if changed:
                    break
        # Convert back to dict lists
        out = {}
        for tid, arr in tracks_t.items():
            out[tid] = [{"x": int(x), "y": int(y)} for (x, y) in arr]
        return out

    tracks = _schedule_tracks(tracks)
    return starts, tracks


def _manhattan_expand(p0: Dict[str, int], p1: Dict[str, int]):
    """Generate 4-neighbor steps from p0 to p1 (x first, then y), excluding p0, including p1."""
    out = []
    x0, y0 = int(p0["x"]), int(p0["y"])
    x1, y1 = int(p1["x"]), int(p1["y"])
    # Step along x
    while x0 != x1:
        x0 += 1 if x0 < x1 else -1
        out.append({"x": x0, "y": y0})
    # Then along y
    while y0 != y1:
        y0 += 1 if y0 < y1 else -1
        out.append({"x": x0, "y": y0})
    return out


def _densify_tracks_manhattan(tracks: Dict[Any, list]):
    """Replace each track with a 4-neighbor staircase path to avoid diagonal jumps in Unity."""
    dense: Dict[Any, list] = {}
    for tid, pts in tracks.items():
        if not pts:
            dense[tid] = []
            continue
        d = [{"x": int(pts[0]["x"]), "y": int(pts[0]["y"])}]
        for i in range(1, len(pts)):
            p0 = d[-1]
            p1 = {"x": int(pts[i]["x"]), "y": int(pts[i]["y"])}
            if p0["x"] == p1["x"] and p0["y"] == p1["y"]:
                continue
            d.extend(_manhattan_expand(p0, p1))
        dense[tid] = d
    return dense


def _to_unity_payload(cfg: Dict[str, Any], city: City, sim: Simulation) -> Dict[str, Any]:
    # Build road-constrained paths from events to avoid off-road/diagonal motion
    starts, tracks = _build_agent_paths_from_events(city, sim)

    truck_ids = [t.tid for t in sim.trucks]
    truck_id_to_int = {tid: idx for idx, tid in enumerate(truck_ids)}

    initial_bins = {}
    if sim.frames:
        for b in sim.frames[0]["bins"]:
            initial_bins[b["id"]] = int(b["fill"])

    bins_out = []
    for b in sim.bins:
        bid_int = int(str(b.id).lstrip("b")) if isinstance(b.id, str) else int(b.id)
        bx, by = int(round(b.pos[0])), int(round(b.pos[1]))
        bins_out.append({
            "id": bid_int,
            "pos": [bx, by],
            "initial": initial_bins.get(b.id, int(b.fill)),
            "remaining": int(b.fill)
        })

    agents_out = []
    for tid in truck_ids:
        aid = truck_id_to_int[tid]
        start = starts.get(tid, [int(round(city.depot[0])), int(round(city.depot[1]))])
        path = tracks.get(tid, [])
        dist_sum = 0.0
        for i in range(1, len(path)):
            dx = path[i]["x"] - path[i-1]["x"]
            dy = path[i]["y"] - path[i-1]["y"]
            dist_sum += (dx*dx + dy*dy) ** 0.5
        collected = 0
        for ev in sim.events:
            if ev.get("type") == "pickup" and ev.get("truck") == tid:
                collected += int(ev.get("amount", 0))
        agents_out.append({
            "id": aid,
            "start": start,
            "pathObj": path,
            "distance": int(round(dist_sum)),
            "collected": collected,
            "capacity": int(cfg.get("TRUCK_CAPACITY", 300))
        })

    events_out = []
    for ev in sim.events:
        et = ev.get("type")
        if et not in ("assign", "pickup", "drop", "recharge", "overflow"):
            continue
        out = {"t": int(round(ev.get("t", 0)))}
        if et == "assign":
            out["type"] = "ASSIGN"
            out["agent"] = truck_id_to_int.get(ev.get("truck"), 0)
            bid = ev.get("bin")
            out["bin"] = int(str(bid).lstrip("b")) if isinstance(bid, str) else (int(bid) if bid is not None else 0)
        elif et == "pickup":
            out["type"] = "SERVICE"
            out["agent"] = truck_id_to_int.get(ev.get("truck"), 0)
            bid = ev.get("bin")
            out["bin"] = int(str(bid).lstrip("b")) if isinstance(bid, str) else (int(bid) if bid is not None else 0)
            out["amount"] = int(ev.get("amount", 0))
        elif et == "drop":
            out["type"] = "DUMP"
            out["agent"] = truck_id_to_int.get(ev.get("truck"), 0)
        elif et == "recharge":
            out["type"] = "RECHARGE"
            out["agent"] = truck_id_to_int.get(ev.get("truck"), 0)
        elif et == "overflow":
            out["type"] = "OVERFLOW"
            bid = ev.get("bin")
            out["bin"] = int(str(bid).lstrip("b")) if isinstance(bid, str) else (int(bid) if bid is not None else 0)
        events_out.append(out)

    total_collected = sum(e.get("amount", 0) for e in sim.events if e.get("type") == "pickup")
    avg_dist = 0.0
    if agents_out:
        avg_dist = sum(a["distance"] for a in agents_out) / float(len(agents_out))
    negotiation_msgs = sum(1 for e in events_out if e.get("type") == "ASSIGN")

    grid = {
        "width": int(cfg["MAP_SIZE"][0]),
        "height": int(cfg["MAP_SIZE"][1]),
        "depot": [int(round(city.depot[0])), int(round(city.depot[1]))]
    }

    return {
        "grid": grid,
        "agents": agents_out,
        "bins": bins_out,
        "events": events_out,
        "metrics": {
            "total_collected": int(total_collected),
            "avg_distance_per_agent": float(avg_dist),
            "negotiation_messages": int(negotiation_msgs),
            "steps": int(len(sim.frames))
        }
    }


def _run_simulation(cfg: Dict[str, Any], steps: int, planner: str):
    """Run the hybrid simulation, preferring agentpy if available, and return (city, sim)."""
    city = City(cfg)
    ap_ok = (ap is not None) and hasattr(ap, 'Parameters') and hasattr(ap, 'Model') and (WasteSimModel is not None)
    if ap_ok:
        try:
            params = ap.Parameters({'cfg': cfg, 'steps': steps, 'planner': planner})
            model = WasteSimModel(parameters=params)
            for _ in range(steps):
                model.step()
            return model.city, model.sim
        except Exception:
            pass
    sim = Simulation(cfg=cfg, city=city, planner=planner)
    sim.run(steps)
    return city, sim


def _run_simulation_background(job_id: int, cfg: Dict[str, Any], steps: int, planner: str):
    """Background thread for async simulation jobs."""
    global LAST_RESULT, LAST_RESULT_META, JOB_STATE
    try:
        city, sim = _run_simulation(cfg, steps, planner)
        payload = _to_unity_payload(cfg, city, sim)
        from time import time as _now
        with _JOB_LOCK:
            LAST_RESULT = payload
            LAST_RESULT_META = {
                'generated_at': _now(),
                'steps': steps,
                'planner': planner,
                'config': {k: cfg[k] for k in ['N_TRUCKS','N_BINS','BIN_CAPACITY','TRUCK_SPEED_MPS','POLICY'] if k in cfg}
            }
            if JOB_STATE.get('id') == job_id:
                JOB_STATE['status'] = 'done'
                JOB_STATE['finished_at'] = _now()
    except Exception as e:
        with _JOB_LOCK:
            if JOB_STATE.get('id') == job_id:
                JOB_STATE['status'] = 'error'
                JOB_STATE['error'] = str(e)
                JOB_STATE['finished_at'] = time.time()
               

def _apply_overrides(cfg: Dict[str, Any], *, seed=None, num_agents=None, num_waste_locations=None,
                     bin_capacity=None, truck_speed=None, sidewalk_offset=None,
                     opportunistic_fill_frac=None, urgency_horizon_s=None, coverage_bias=None,
                     service_cooldown_s=None, policy=None) -> None:
    if seed is not None:
        cfg["SEED"] = int(seed)
    if num_agents is not None:
        cfg["N_TRUCKS"] = int(num_agents)
    if num_waste_locations is not None:
        cfg["N_BINS"] = int(num_waste_locations)
    if bin_capacity is not None:
        cfg["BIN_CAPACITY"] = int(bin_capacity)
    if truck_speed is not None:
        cfg["TRUCK_SPEED_MPS"] = float(truck_speed)
    if sidewalk_offset is not None:
        cfg["SIDEWALK_OFFSET_M"] = float(sidewalk_offset)
    if opportunistic_fill_frac is not None:
        cfg["OPPORTUNISTIC_FILL_FRAC"] = float(opportunistic_fill_frac)
    if urgency_horizon_s is not None:
        cfg["URGENCY_HORIZON_S"] = int(urgency_horizon_s)
    if coverage_bias is not None:
        cfg["COVERAGE_BIAS"] = float(coverage_bias)
    if service_cooldown_s is not None:
        cfg["SERVICE_COOLDOWN_S"] = float(service_cooldown_s)
    if policy is not None:
        p = str(policy).strip().lower()
        if p in ("auction", "dqn"):
            cfg["POLICY"] = p


# Routes -----------------------------------------------------------------------

@app.get("/health")
def health():
    return {"ok": True}


@app.get("/preview")
def preview():
    """Run a fast short simulation (default 60 steps) using current defaults for quick Unity spawn preview."""
    from copy import deepcopy as _dc
    cfg = _dc(CONFIG)
    _apply_overrides(cfg,
                     seed=DEFAULT_OVERRIDES.get('seed'),
                     num_agents=DEFAULT_OVERRIDES.get('num_agents'),
                     num_waste_locations=DEFAULT_OVERRIDES.get('num_waste_locations'),
                     bin_capacity=DEFAULT_OVERRIDES.get('bin_capacity'),
                     truck_speed=DEFAULT_OVERRIDES.get('truck_speed'),
                     sidewalk_offset=DEFAULT_OVERRIDES.get('sidewalk_offset'),
                     opportunistic_fill_frac=DEFAULT_OVERRIDES.get('opportunistic_fill_frac'),
                     urgency_horizon_s=DEFAULT_OVERRIDES.get('urgency_horizon_s'),
                     coverage_bias=DEFAULT_OVERRIDES.get('coverage_bias'),
                     service_cooldown_s=DEFAULT_OVERRIDES.get('service_cooldown_s'),
                     policy=DEFAULT_OVERRIDES.get('policy'))
    steps = 60
    planner = DEFAULT_OVERRIDES.get('planner') or 'graph'
    city, sim = _run_simulation(cfg, steps, planner)
    return _to_unity_payload(cfg, city, sim)


@app.get("/config")
def get_config():
    return deepcopy(CONFIG)


@app.get("/defaults")
def get_defaults():
    return deepcopy(DEFAULT_OVERRIDES)


@app.post("/defaults")
def set_defaults(req: DefaultsRequest):
    if req.seed is not None:
        DEFAULT_OVERRIDES['seed'] = int(req.seed)
    if req.num_agents is not None:
        DEFAULT_OVERRIDES['num_agents'] = int(req.num_agents)
    if req.num_waste_locations is not None:
        DEFAULT_OVERRIDES['num_waste_locations'] = int(req.num_waste_locations)
    if req.bin_capacity is not None:
        DEFAULT_OVERRIDES['bin_capacity'] = int(req.bin_capacity)
    if req.steps is not None:
        DEFAULT_OVERRIDES['steps'] = int(req.steps)
    if req.planner is not None:
        DEFAULT_OVERRIDES['planner'] = str(req.planner)
    if req.truck_speed is not None:
        DEFAULT_OVERRIDES['truck_speed'] = float(req.truck_speed)
    if req.sidewalk_offset is not None:
        DEFAULT_OVERRIDES['sidewalk_offset'] = float(req.sidewalk_offset)
    if req.opportunistic_fill_frac is not None:
        DEFAULT_OVERRIDES['opportunistic_fill_frac'] = float(req.opportunistic_fill_frac)
    if req.urgency_horizon_s is not None:
        DEFAULT_OVERRIDES['urgency_horizon_s'] = int(req.urgency_horizon_s)
    if req.coverage_bias is not None:
        DEFAULT_OVERRIDES['coverage_bias'] = float(req.coverage_bias)
    if req.service_cooldown_s is not None:
        DEFAULT_OVERRIDES['service_cooldown_s'] = float(req.service_cooldown_s)
    if req.policy is not None:
        DEFAULT_OVERRIDES['policy'] = str(req.policy)
    _save_defaults_to_disk()
    # Auto-run a simulation with new defaults so Unity can fetch a ready snapshot
    try:
        cfg = deepcopy(CONFIG)
        _apply_overrides(cfg,
                         seed=DEFAULT_OVERRIDES.get('seed'),
                         num_agents=DEFAULT_OVERRIDES.get('num_agents'),
                         num_waste_locations=DEFAULT_OVERRIDES.get('num_waste_locations'),
                         bin_capacity=DEFAULT_OVERRIDES.get('bin_capacity'),
                         truck_speed=DEFAULT_OVERRIDES.get('truck_speed'),
                         sidewalk_offset=DEFAULT_OVERRIDES.get('sidewalk_offset'),
                         opportunistic_fill_frac=DEFAULT_OVERRIDES.get('opportunistic_fill_frac'),
                         urgency_horizon_s=DEFAULT_OVERRIDES.get('urgency_horizon_s'),
                         coverage_bias=DEFAULT_OVERRIDES.get('coverage_bias'),
                         service_cooldown_s=DEFAULT_OVERRIDES.get('service_cooldown_s'),
                         policy=DEFAULT_OVERRIDES.get('policy'))
        steps = int(DEFAULT_OVERRIDES.get('steps') or 600)
        planner = DEFAULT_OVERRIDES.get('planner') or 'graph'
        city, sim = _run_simulation(cfg, steps, planner)
        payload = _to_unity_payload(cfg, city, sim)
        from time import time as _now
        global LAST_RESULT, LAST_RESULT_META
        LAST_RESULT = payload
        LAST_RESULT_META = {
            'generated_at': _now(),
            'steps': steps,
            'planner': planner,
            'config': {k: cfg[k] for k in ['N_TRUCKS','N_BINS','BIN_CAPACITY','TRUCK_SPEED_MPS','POLICY'] if k in cfg}
        }
        return {"ok": True, "defaults": deepcopy(DEFAULT_OVERRIDES), "ready": True, "result": payload, "meta": LAST_RESULT_META}
    except Exception as e:
        return {"ok": False, "error": str(e), "defaults": deepcopy(DEFAULT_OVERRIDES), "ready": False}

@app.get("/last_result")
def last_result():
    if LAST_RESULT is None:
        return {"ready": False, "message": "No simulation generated yet. POST /defaults or /simulate first."}
    return {"ready": True, "result": LAST_RESULT, "meta": LAST_RESULT_META}


@app.post("/simulate_async")
def simulate_async(req: SimRequest):
    global _JOB_COUNTER, JOB_STATE
    with _JOB_LOCK:
        if JOB_STATE.get('status') == 'running':
            return {"accepted": False, "reason": "Job already running", "job": JOB_STATE}
        _JOB_COUNTER += 1
        job_id = _JOB_COUNTER
        cfg = deepcopy(CONFIG)
        _apply_overrides(cfg,
                         seed=DEFAULT_OVERRIDES.get('seed'),
                         num_agents=DEFAULT_OVERRIDES.get('num_agents'),
                         num_waste_locations=DEFAULT_OVERRIDES.get('num_waste_locations'),
                         bin_capacity=DEFAULT_OVERRIDES.get('bin_capacity'),
                         truck_speed=DEFAULT_OVERRIDES.get('truck_speed'),
                         sidewalk_offset=DEFAULT_OVERRIDES.get('sidewalk_offset'),
                         opportunistic_fill_frac=DEFAULT_OVERRIDES.get('opportunistic_fill_frac'),
                         urgency_horizon_s=DEFAULT_OVERRIDES.get('urgency_horizon_s'),
                         coverage_bias=DEFAULT_OVERRIDES.get('coverage_bias'),
                         service_cooldown_s=DEFAULT_OVERRIDES.get('service_cooldown_s'),
                         policy=DEFAULT_OVERRIDES.get('policy'))
        _apply_overrides(cfg,
                         seed=req.seed,
                         num_agents=req.num_agents,
                         num_waste_locations=req.num_waste_locations,
                         bin_capacity=req.bin_capacity,
                         truck_speed=req.truck_speed,
                         sidewalk_offset=req.sidewalk_offset,
                         opportunistic_fill_frac=req.opportunistic_fill_frac,
                         urgency_horizon_s=req.urgency_horizon_s,
                         coverage_bias=req.coverage_bias,
                         service_cooldown_s=req.service_cooldown_s,
                         policy=req.policy)
        steps = int(req.steps if req.steps is not None else (DEFAULT_OVERRIDES.get('steps') or 600))
        planner = req.planner if req.planner is not None else (DEFAULT_OVERRIDES.get('planner') or 'graph')
        JOB_STATE = {
            'id': job_id,
            'status': 'running',
            'started_at': time.time(),
            'finished_at': None,
            'steps': steps,
            'planner': planner,
            'config_snapshot': {k: cfg.get(k) for k in ['N_TRUCKS','N_BINS','BIN_CAPACITY','TRUCK_SPEED_MPS','POLICY','SEED']},
            'error': None
        }
    th = threading.Thread(target=_run_simulation_background, args=(job_id, cfg, steps, planner), daemon=True)
    th.start()
    return {"accepted": True, "job": JOB_STATE}


@app.get("/job_status")
def job_status():
    with _JOB_LOCK:
        return deepcopy(JOB_STATE)


@app.post("/simulate")
def simulate(req: SimRequest, request: Request):
    cfg = deepcopy(CONFIG)
    # 0) Apply server defaults first
    _apply_overrides(cfg,
                     seed=DEFAULT_OVERRIDES.get('seed'),
                     num_agents=DEFAULT_OVERRIDES.get('num_agents'),
                     num_waste_locations=DEFAULT_OVERRIDES.get('num_waste_locations'),
                     bin_capacity=DEFAULT_OVERRIDES.get('bin_capacity'),
                     truck_speed=DEFAULT_OVERRIDES.get('truck_speed'),
                     sidewalk_offset=DEFAULT_OVERRIDES.get('sidewalk_offset'),
                     opportunistic_fill_frac=DEFAULT_OVERRIDES.get('opportunistic_fill_frac'),
                     urgency_horizon_s=DEFAULT_OVERRIDES.get('urgency_horizon_s'),
                     coverage_bias=DEFAULT_OVERRIDES.get('coverage_bias'),
                     service_cooldown_s=DEFAULT_OVERRIDES.get('service_cooldown_s'),
                     policy=DEFAULT_OVERRIDES.get('policy'))

    # 1) Apply JSON body overrides
    _apply_overrides(cfg,
                     seed=req.seed,
                     num_agents=req.num_agents,
                     num_waste_locations=req.num_waste_locations,
                     bin_capacity=req.bin_capacity,
                     truck_speed=req.truck_speed,
                     sidewalk_offset=req.sidewalk_offset,
                     opportunistic_fill_frac=req.opportunistic_fill_frac,
                     urgency_horizon_s=req.urgency_horizon_s,
                     coverage_bias=req.coverage_bias,
                     service_cooldown_s=req.service_cooldown_s,
                     policy=req.policy)

    # 2) Also honor query params on POST (handy if client appends ?seed=... etc.)
    qp = request.query_params
    _apply_overrides(cfg,
                     seed=qp.get('seed'),
                     num_agents=qp.get('num_agents'),
                     num_waste_locations=qp.get('num_waste_locations'),
                     bin_capacity=qp.get('bin_capacity'),
                     truck_speed=qp.get('truck_speed'),
                     sidewalk_offset=qp.get('sidewalk_offset'),
                     opportunistic_fill_frac=qp.get('opportunistic_fill_frac'),
                     urgency_horizon_s=qp.get('urgency_horizon_s'),
                     coverage_bias=qp.get('coverage_bias'),
                     service_cooldown_s=qp.get('service_cooldown_s'),
                     policy=qp.get('policy'))

    steps = int((qp.get('steps') if 'steps' in qp else (req.steps if req.steps is not None else (DEFAULT_OVERRIDES.get('steps') or 600))))
    planner = (qp.get('planner') if 'planner' in qp else (req.planner if req.planner is not None else (DEFAULT_OVERRIDES.get('planner') or 'graph')))

    city, sim = _run_simulation(cfg, steps, planner)
    payload = _to_unity_payload(cfg, city, sim)
    return payload


@app.get("/simulate")
def simulate_get(
    seed: Optional[int] = None,
    num_agents: Optional[int] = None,
    num_waste_locations: Optional[int] = None,
    bin_capacity: Optional[int] = None,
    steps: Optional[int] = None,
    planner: Optional[str] = "graph",
    truck_speed: Optional[float] = None,
    sidewalk_offset: Optional[float] = None,
    opportunistic_fill_frac: Optional[float] = None,
    urgency_horizon_s: Optional[int] = None,
    coverage_bias: Optional[float] = None,
    service_cooldown_s: Optional[float] = None,
    policy: Optional[str] = None,
):
    cfg = deepcopy(CONFIG)
    # Apply server defaults first
    _apply_overrides(cfg,
                     seed=DEFAULT_OVERRIDES.get('seed'),
                     num_agents=DEFAULT_OVERRIDES.get('num_agents'),
                     num_waste_locations=DEFAULT_OVERRIDES.get('num_waste_locations'),
                     bin_capacity=DEFAULT_OVERRIDES.get('bin_capacity'),
                     truck_speed=DEFAULT_OVERRIDES.get('truck_speed'),
                     sidewalk_offset=DEFAULT_OVERRIDES.get('sidewalk_offset'),
                     opportunistic_fill_frac=DEFAULT_OVERRIDES.get('opportunistic_fill_frac'),
                     urgency_horizon_s=DEFAULT_OVERRIDES.get('urgency_horizon_s'),
                     service_cooldown_s=DEFAULT_OVERRIDES.get('service_cooldown_s'),
                     coverage_bias=DEFAULT_OVERRIDES.get('coverage_bias'))
    if seed is not None:
        cfg["SEED"] = int(seed)
    if num_agents is not None:
        cfg["N_TRUCKS"] = int(num_agents)
    if num_waste_locations is not None:
        cfg["N_BINS"] = int(num_waste_locations)
    if bin_capacity is not None:
        cfg["BIN_CAPACITY"] = int(bin_capacity)
    if truck_speed is not None:
        cfg["TRUCK_SPEED_MPS"] = float(truck_speed)
    if sidewalk_offset is not None:
        cfg["SIDEWALK_OFFSET_M"] = float(sidewalk_offset)
    if opportunistic_fill_frac is not None:
        cfg["OPPORTUNISTIC_FILL_FRAC"] = float(opportunistic_fill_frac)
    if urgency_horizon_s is not None:
        cfg["URGENCY_HORIZON_S"] = int(urgency_horizon_s)
    if coverage_bias is not None:
        cfg["COVERAGE_BIAS"] = float(coverage_bias)
    if service_cooldown_s is not None:
        cfg["SERVICE_COOLDOWN_S"] = float(service_cooldown_s)
    if policy is not None:
        p = str(policy).strip().lower()
        if p in ("auction", "dqn"):
            cfg["POLICY"] = p

    # Steps/planner fallback to server defaults too
    eff_steps = int(steps if steps is not None else (DEFAULT_OVERRIDES.get('steps') or 600))
    eff_planner = planner or DEFAULT_OVERRIDES.get('planner') or "graph"
    city, sim = _run_simulation(cfg, eff_steps, eff_planner)
    return _to_unity_payload(cfg, city, sim)


@app.get("/schema")
def schema():
    """Return the JSON schema for the Unity SimData payload."""
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "SimData",
        "type": "object",
        "properties": {
            "grid": {
                "type": "object",
                "properties": {
                    "width": {"type": "integer"},
                    "height": {"type": "integer"},
                    "depot": {"type": "array", "items": {"type": "integer"}, "minItems": 2, "maxItems": 2}
                },
                "required": ["width", "height", "depot"]
            },
            "agents": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "integer"},
                        "start": {"type": "array", "items": {"type": "integer"}, "minItems": 2, "maxItems": 2},
                        "pathObj": {"type": "array", "items": {"type": "object", "properties": {"x": {"type": "integer"}, "y": {"type": "integer"}}, "required": ["x", "y"]}},
                        "distance": {"type": "integer"},
                        "collected": {"type": "integer"},
                        "capacity": {"type": "integer"}
                    },
                    "required": ["id", "start", "pathObj", "distance", "collected", "capacity"]
                }
            },
            "bins": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "integer"},
                        "pos": {"type": "array", "items": {"type": "integer"}, "minItems": 2, "maxItems": 2},
                        "initial": {"type": "integer"},
                        "remaining": {"type": "integer"}
                    },
                    "required": ["id", "pos", "initial", "remaining"]
                }
            },
            "events": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "t": {"type": "integer"},
                        "type": {"type": "string", "enum": ["ASSIGN", "SERVICE", "DUMP", "RECHARGE", "OVERFLOW"]},
                        "agent": {"type": "integer"},
                        "bin": {"type": "integer"},
                        "amount": {"type": "integer"}
                    },
                    "required": ["t", "type"]
                }
            },
            "metrics": {
                "type": "object",
                "properties": {
                    "total_collected": {"type": "integer"},
                    "avg_distance_per_agent": {"type": "number"},
                    "negotiation_messages": {"type": "integer"},
                    "steps": {"type": "integer"}
                },
                "required": ["total_collected", "avg_distance_per_agent", "negotiation_messages", "steps"]
            }
        },
        "required": ["grid", "agents", "bins", "events", "metrics"]
    }


@app.get("/", response_class=HTMLResponse)
def index():
    return """
<!doctype html>
<html>
    <head>
        <meta charset=\"utf-8\" />
        <title>Waste Collection Hybrid API</title>
    <link rel=\"icon\" href=\"/favicon.ico\" />
        <style>
            body{font-family:system-ui,-apple-system,Segoe UI,Roboto,Arial,sans-serif;max-width:920px;margin:2rem auto;padding:0 1rem;}
            code, pre{background:#f6f8fa;padding:2px 4px;border-radius:4px}
            .card{border:1px solid #e5e7eb;border-radius:8px;padding:1rem;margin:1rem 0}
            label{display:block;margin:.4rem 0 .2rem}
            input,select{padding:.4rem .5rem;border:1px solid #d1d5db;border-radius:4px;width:12rem}
            button{padding:.5rem .9rem;border:1px solid #2563eb;background:#2563eb;color:#fff;border-radius:6px;cursor:pointer}
            button:hover{background:#1d4ed8}
            .row{display:flex;gap:1rem;flex-wrap:wrap}
        </style>
    </head>
    <body>
        <h1>Waste Collection Hybrid API</h1>
        <p>Quick links:</p>
        <ul>
            <li><a href=\"/health\">/health</a></li>
            <li><a href=\"/config\">/config</a></li>
            <li><a href=\"/schema\">/schema</a></li>
            <li><a href=\"/preview\">/preview</a> (fast 60-step preview)</li>
            <li><a href=\"/simulate?steps=120&num_agents=3&num_waste_locations=10\">/simulate?steps=120&amp;num_agents=3&amp;num_waste_locations=10</a></li>
        </ul>
        <div class=\"card\">
            <h2>Set server defaults</h2>
            <p>These defaults are applied whenever requests omit a parameter. Unity can call a constant URL (<code>/simulate</code>) and your saved defaults will be used.</p>
            <form method=\"POST\" action=\"/defaults\" onsubmit=\"return submitDefaults(event)\">
                <div class=\"row\">
                    <div><label>seed</label><input id=\"d_seed\" type=\"number\" name=\"seed\" placeholder=\"42\"></div>
                    <div><label>num_agents</label><input id=\"d_agents\" type=\"number\" name=\"num_agents\" placeholder=\"3\"></div>
                    <div><label>num_waste_locations</label><input id=\"d_bins\" type=\"number\" name=\"num_waste_locations\" placeholder=\"12\"></div>
                    <div><label>bin_capacity</label><input id=\"d_cap\" type=\"number\" name=\"bin_capacity\" placeholder=\"100\"></div>
                    <div><label>steps</label><input id=\"d_steps\" type=\"number\" name=\"steps\" placeholder=\"600\"></div>
                    <div><label>planner</label>
                        <select id=\"d_planner\" name=\"planner\">
                            <option value=\"\">(keep)</option>
                            <option value=\"graph\">graph</option>
                            <option value=\"grid\">grid</option>
                        </select>
                    </div>
                    <div><label>truck_speed</label><input id=\"d_speed\" type=\"number\" step=\"0.1\" name=\"truck_speed\" placeholder=\"2.0\"></div>
                    <div><label>sidewalk_offset (m)</label><input id=\"d_side\" type=\"number\" step=\"0.1\" name=\"sidewalk_offset\" placeholder=\"2.0\"></div>
                    <div><label>opportunistic_fill_frac</label><input id=\"d_off\" type=\"number\" step=\"0.05\" min=\"0\" max=\"1\" name=\"opportunistic_fill_frac\" placeholder=\"0.60\"></div>
                    <div><label>coverage_bias</label><input id=\"d_cov\" type=\"number\" step=\"0.05\" min=\"0\" max=\"1\" name=\"coverage_bias\" placeholder=\"0.50\"></div>
                    <div><label>urgency_horizon_s</label><input id=\"d_urg\" type=\"number\" name=\"urgency_horizon_s\" placeholder=\"120\"></div>
                    <div><label>service_cooldown_s</label><input id=\"d_cool\" type=\"number\" step=\"1\" min=\"0\" name=\"service_cooldown_s\" placeholder=\"300\"></div>
                    <div><label>policy</label>
                        <select id=\"d_policy\" name=\"policy\">\n                            <option value=\"\">(keep)</option>\n                            <option value=\"auction\">auction</option>\n                            <option value=\"dqn\">dqn</option>\n                        </select>
                    </div>
                </div>
                <p><button type=\"submit\">Save defaults</button></p>
            </form>
            <pre id=\"defaults_out\"></pre>
            <div id=\"defaults_run_badge\" style=\"margin-top:.5rem;padding:.3rem .6rem;display:inline-block;border:1px solid #ccc;border-radius:4px;font-size:.75rem;background:#f8f8f8;\">Idle</div>
        </div>
        <div class=\"card\">
            <h2>Async simulation run</h2>
            <p>Start a background run; poll status (Unity: GET /job_status, then /last_result).</p>
            <form onsubmit=\"return startAsync(event)\">
                <div class=\"row\">
                    <div><label>steps (override)</label><input id=\"a_steps\" type=\"number\" placeholder=\"(keep)\" /></div>
                    <div><label>planner</label>
                        <select id=\"a_planner\">
                            <option value=\"\">(keep)</option>
                            <option value=\"graph\">graph</option>
                            <option value=\"grid\">grid</option>
                        </select>
                    </div>
                </div>
                <p><button type=\"submit\">Start async run</button></p>
            </form>
            <pre id=\"async_status\">(idle)</pre>
            <div id=\"run_done_badge\" style=\"margin-top:.5rem;padding:.3rem .6rem;display:inline-block;border:1px solid #ccc;border-radius:4px;font-size:.85rem;background:#f8f8f8;\">No run yet</div>
        </div>
        <script>
        async function loadDefaults(){
            try{
                const r = await fetch('/defaults');
                const d = await r.json();
                document.getElementById('defaults_out').textContent = JSON.stringify(d, null, 2);
            }catch(e){ console.error(e); }
        }
        async function submitDefaults(ev){
            ev.preventDefault();
            const body = {
                seed: valNum('d_seed'),
                num_agents: valNum('d_agents'),
                num_waste_locations: valNum('d_bins'),
                bin_capacity: valNum('d_cap'),
                steps: valNum('d_steps'),
                planner: (document.getElementById('d_planner').value || undefined),
                truck_speed: valNum('d_speed', true),
                    sidewalk_offset: valNum('d_side', true),
                opportunistic_fill_frac: valNum('d_off', true),
                    urgency_horizon_s: valNum('d_urg'),
                    coverage_bias: valNum('d_cov', true),
                    service_cooldown_s: valNum('d_cool'),
                    policy: (document.getElementById('d_policy').value || undefined)
            };
            Object.keys(body).forEach(k=> body[k]===undefined && delete body[k]);
            const db = document.getElementById('defaults_run_badge');
            if(db){ db.textContent='Running...'; db.style.background='#fff7ed'; db.style.borderColor='#fb923c'; }
            const r = await fetch('/defaults', {method:'POST', headers:{'content-type':'application/json'}, body: JSON.stringify(body)});
            const d = await r.json();
            document.getElementById('defaults_out').textContent = JSON.stringify(d, null, 2);
            if(db){ if(d.ok){ db.textContent='Run complete'; db.style.background='#d1fae5'; db.style.borderColor='#10b981'; } else { db.textContent='Error'; db.style.background='#fee2e2'; db.style.borderColor='#ef4444'; } }
            return false;
        }
        function valNum(id, isFloat){
            const v = document.getElementById(id).value.trim();
            if(!v) return undefined;
            return isFloat ? parseFloat(v) : parseInt(v,10);
        }
        async function pollJob(){
            try{ const r = await fetch('/job_status'); const j = await r.json(); document.getElementById('async_status').textContent = JSON.stringify(j,null,2); if(j.status==='done'||j.status==='error'){ const lr = await fetch('/last_result'); const lrj = await lr.json(); if(lrj.meta) document.getElementById('async_status').textContent = JSON.stringify({job:j,last_result:lrj.meta},null,2); const badge=document.getElementById('run_done_badge'); if(badge){ if(j.status==='done'){ badge.textContent='Run complete @ '+ new Date(j.finished_at*1000).toLocaleTimeString(); badge.style.background='#d1fae5'; badge.style.borderColor='#10b981'; } else { badge.textContent='Run error'; badge.style.background='#fee2e2'; badge.style.borderColor='#ef4444'; } } } else { const badge=document.getElementById('run_done_badge'); if(badge){ badge.textContent='Running...'; badge.style.background='#fff7ed'; badge.style.borderColor='#fb923c'; } } }catch(e){}
        }
        setInterval(pollJob,1000);
        async function startAsync(ev){ ev.preventDefault(); const body={}; const s=valNum('a_steps'); if(s!==undefined) body.steps=s; const planner=document.getElementById('a_planner').value; if(planner) body.planner=planner; const r=await fetch('/simulate_async',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify(body)}); const j=await r.json(); document.getElementById('async_status').textContent=JSON.stringify(j,null,2); return false; }
        loadDefaults();
        </script>
    </body>
    </html>
        """


# Tiny inline favicon (16x16 PNG, blue square)
_FAVICON_BASE64 = b'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMB/at5fjoAAAAASUVORK5CYII='


@app.get("/favicon.ico")
def favicon():
    data = base64.b64decode(_FAVICON_BASE64)
    return Response(content=data, media_type="image/png")
