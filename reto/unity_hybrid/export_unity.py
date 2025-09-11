import sys, os, json, math, argparse
try:  # package context
    from .sim import Simulation  # type: ignore
    from .config import CONFIG  # type: ignore
    from .city import City  # type: ignore
except ImportError:  # script fallback
    from sim import Simulation  # type: ignore
    from config import CONFIG  # type: ignore
    from city import City  # type: ignore
from ap_model import WasteSimModel, ap


def _project_to_roads(city: City, x: float, y: float) -> tuple[float, float]:
    import math
    best_p = (x, y)
    best_d = float('inf')
    for r in city.roads:
        (x1,y1),(x2,y2) = r.polyline
        vx, vy = x2-x1, y2-y1
        wx, wy = x - x1, y - y1
        L2 = vx*vx + vy*vy
        t = 0.0 if L2 < 1e-9 else max(0.0, min(1.0, (wx*vx + wy*vy) / L2))
        px, py = x1 + t*vx, y1 + t*vy
        d = math.hypot(px-x, py-y)
        if d < best_d:
            best_d = d; best_p = (px, py)
    return best_p

def build_agent_paths(frames, city: City):
    # Collapse per-truck positions into pathObj arrays for Unity, keyed by truck id
    tracks = {}
    starts = {}
    for i, fr in enumerate(frames):
        for t in fr["trucks"]:
            tid = t["id"]
            if i == 0:
                sx, sy = _project_to_roads(city, float(t["x"]), float(t["y"]))
                starts[tid] = [int(round(sx)) , int(round(sy))]
            px, py = _project_to_roads(city, float(t["x"]), float(t["y"]))
            tracks.setdefault(tid, []).append({
                "x": int(round(px)),
                "y": int(round(py))
            })
    return starts, tracks


def to_unity_simdata(cfg: dict, city: City, sim: Simulation):
    # Build per-agent paths and starts
    starts, tracks = build_agent_paths(sim.frames, city)

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
        # distance and collected
        dist_sum = 0.0
        for i in range(1, len(path)):
            dx = path[i]["x"] - path[i-1]["x"]
            dy = path[i]["y"] - path[i-1]["y"]
            dist_sum += math.hypot(dx, dy)
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

    # Events mapping to Unity schema
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

    # Metrics
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--trucks", type=int, default=3)
    parser.add_argument("--bins", type=int, default=15)
    parser.add_argument("--bin-cap", type=int, default=None)
    parser.add_argument("--planner", choices=["graph", "grid"], default="graph")
    parser.add_argument("--out-json", default="sim_run_pathObj.json")
    parser.add_argument("--out-log", default="full_log.json")
    args = parser.parse_args()

    cfg = CONFIG.copy()
    cfg["N_TRUCKS"] = args.trucks
    cfg["N_BINS"] = args.bins
    if args.bin_cap is not None:
        cfg["BIN_CAPACITY"] = int(args.bin_cap)

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

    simdata = to_unity_simdata(cfg, city, sim)

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(simdata, f, ensure_ascii=False, indent=2)
    with open(args.out_log, "w", encoding="utf-8") as f:
        json.dump({"cfg": cfg, "frames": sim.frames, "events": sim.events}, f, ensure_ascii=False, indent=2)

    print(f"Wrote {args.out_json} and {args.out_log}")


if __name__ == "__main__":
    main()
