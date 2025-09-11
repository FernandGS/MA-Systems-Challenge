import sys, os, json, math, argparse
from config import CONFIG
from city import City
from sim import Simulation
from ap_model import WasteSimModel, ap


def build_agent_paths(frames, enable_dwell=True):
    """Return (starts, pathObj, pathDwell, fullFrames)

    If enable_dwell is False, pathObj is the full per-frame sequence (no compression)
    and pathDwell will be None so older Unity consumers behave exactly as before.
    """
    tracks_full = {}
    starts = {}
    for i, fr in enumerate(frames):
        for t in fr["trucks"]:
            tid = t["id"]
            if i == 0:
                starts[tid] = [int(round(t["x"])), int(round(t["y"]))]
            tracks_full.setdefault(tid, []).append({
                "x": int(round(t["x"])),
                "y": int(round(t["y"]))
            })
    if not enable_dwell:
        # Legacy mode: no compression, no dwell metadata
        return starts, tracks_full, None, tracks_full
    # Compress consecutive duplicates into dwell counts
    comp_tracks = {}
    dwell = {}
    for tid, pts in tracks_full.items():
        c = []
        d = []
        last = None
        for p in pts:
            if last is None or p["x"] != last["x"] or p["y"] != last["y"]:
                c.append(p)
                d.append(1)
                last = p
            else:
                d[-1] += 1
        comp_tracks[tid] = c
        dwell[tid] = d
    return starts, comp_tracks, dwell, tracks_full


def to_unity_simdata(cfg: dict, city: City, sim: Simulation, enable_dwell=True):
    # Build per-agent paths and starts (conditionally compressed)
    starts, tracks, dwell, tracks_full = build_agent_paths(sim.frames, enable_dwell=enable_dwell)

    # Map string ids (e.g., 'T0', 'b3') to ints for Unity
    truck_ids = [t.tid for t in sim.trucks]
    truck_id_to_int = {tid: idx for idx, tid in enumerate(truck_ids)}

    # Bin initial/remaining
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

    # Agents with pathObj
    agents_out = []
    for tid in truck_ids:
        aid = truck_id_to_int[tid]
        start = starts.get(tid, [int(round(city.depot[0])), int(round(city.depot[1]))])
        path = tracks.get(tid, [])
        path_dwell = dwell.get(tid, []) if dwell is not None else None
        # Retrieve truck object for float + debug
        truck_obj = next((t for t in sim.trucks if t.tid == tid), None)
        float_path = []
        lane_dbg = []
        raw_path = []
        if truck_obj is not None:
            # Use full uncompressed per-frame path for floatPath
            for p in tracks_full.get(tid, []):
                float_path.append({"x": float(p["x"]), "y": float(p["y"])})
            lane_dbg = getattr(truck_obj, 'debug_lane', []) or []
            if hasattr(truck_obj, 'raw_route_debug'):
                raw_path = [{"x": float(p[0]), "y": float(p[1])} for p in getattr(truck_obj, 'raw_route_debug')]
        dist_sum = 0.0
        for i in range(1, len(path)):
            dx = path[i]["x"] - path[i-1]["x"]
            dy = path[i]["y"] - path[i-1]["y"]
            dist_sum += math.hypot(dx, dy)
        collected = 0
        for ev in sim.events:
            if ev.get("type") == "pickup" and ev.get("truck") == tid:
                collected += int(ev.get("amount", 0))
        agent_entry = {
            "id": aid,
            "start": start,
            "pathObj": path,          # compressed or full spatial path
            "floatPath": float_path,   # full per-frame coordinates
            "rawPlannedPath": raw_path,
            "laneDebug": lane_dbg,
            "distance": int(round(dist_sum)),
            "collected": collected,
            "capacity": int(cfg.get("TRUCK_CAPACITY", 300))
        }
        if enable_dwell and path_dwell is not None:
            agent_entry["pathDwell"] = path_dwell
        agents_out.append(agent_entry)

    # Events mapping to Unity schema
    events_out = []
    for ev in sim.events:
        et = ev.get("type")
        if et not in ("assign", "pickup", "drop", "recharge", "overflow", "intersection_wait", "uturn_block"):
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
        elif et == "intersection_wait":
            out["type"] = "INTERSECTION_WAIT"
            out["agent"] = truck_id_to_int.get(ev.get("truck"), 0)
        elif et == "uturn_block":
            out["type"] = "UTURN_BLOCK"
            out["agent"] = truck_id_to_int.get(ev.get("truck"), 0)
        events_out.append(out)

    # Metrics
    total_collected = sum(e.get("amount", 0) for e in sim.events if e.get("type") == "pickup")
    avg_dist = 0.0
    if agents_out:
        avg_dist = sum(a["distance"] for a in agents_out) / float(len(agents_out))
    negotiation_msgs = sum(1 for e in events_out if e.get("type") == "ASSIGN")
    waits = sum(1 for e in events_out if e.get("type") == "INTERSECTION_WAIT")
    uturn_blocks = sum(1 for e in events_out if e.get("type") == "UTURN_BLOCK")

    grid = {
        "width": int(cfg["MAP_SIZE"][0]),
        "height": int(cfg["MAP_SIZE"][1]),
        "depot": [int(round(city.depot[0])), int(round(city.depot[1]))]
    }

    sim_obj = {
        "grid": grid,
        "agents": agents_out,
        "bins": bins_out,
        "events": events_out,
        "metrics": {
            "total_collected": int(total_collected),
            "avg_distance_per_agent": float(avg_dist),
            "negotiation_messages": int(negotiation_msgs),
            "intersection_waits": int(waits),
            "uturn_blocks": int(uturn_blocks),
            "steps": int(len(sim.frames))
        }
    }
    # Backward compatible schema versioning
    sim_obj["schemaVersion"] = 2 if enable_dwell else 1
    sim_obj["hasDwell"] = bool(enable_dwell and dwell is not None)
    return sim_obj


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--trucks", type=int, default=3)
    parser.add_argument("--bins", type=int, default=15)
    parser.add_argument("--bin-cap", type=int, default=None)
    parser.add_argument("--planner", choices=["graph", "grid"], default="graph")
    parser.add_argument("--bin-spacing", type=float, default=None, help="override MIN_BIN_SEP_M / BIN_MIN_SPACING for placement")
    parser.add_argument("--out-json", default="sim_run_pathObj.json")
    parser.add_argument("--out-log", default="full_log.json")
    parser.add_argument("--no-dwell", action="store_true", help="disable dwell/path compression for backward compatibility")
    args = parser.parse_args()

    cfg = CONFIG.copy()
    cfg["N_TRUCKS"] = args.trucks
    cfg["N_BINS"] = args.bins
    if args.bin_cap is not None:
        cfg["BIN_CAPACITY"] = int(args.bin_cap)
    if args.bin_spacing is not None:
        # Use canonical key; City will also respect legacy alias if needed
        cfg["MIN_BIN_SEP_M"] = float(args.bin_spacing)

    city = City(cfg)
    # Prefer agentpy if available, but keep identical logic by delegating to Simulation
    ap_ok = (ap is not None) and hasattr(ap, 'Parameters') and hasattr(ap, 'Model') and (WasteSimModel is not None)
    if ap_ok:
        try:
            params = ap.Parameters({
                'cfg': cfg,
                'steps': args.steps,
                'planner': args.planner,
            })
            model = WasteSimModel(parameters=params)
            for _ in range(args.steps):
                model.step()
            sim = model.sim
            city = model.city
        except Exception:
            # Fallback to direct simulation if agentpy behaves unexpectedly
            sim = Simulation(cfg=cfg, city=city, planner=args.planner)
            sim.run(args.steps)
    else:
        sim = Simulation(cfg=cfg, city=city, planner=args.planner)
        sim.run(args.steps)

    enable_dwell = not args.no_dwell
    simdata = to_unity_simdata(cfg, city, sim, enable_dwell=enable_dwell)

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(simdata, f, ensure_ascii=False, indent=2)
    with open(args.out_log, "w", encoding="utf-8") as f:
        json.dump({"cfg": cfg, "frames": sim.frames, "events": sim.events}, f, ensure_ascii=False, indent=2)

    print(f"Wrote {args.out_json} and {args.out_log}")


if __name__ == "__main__":
    main()
