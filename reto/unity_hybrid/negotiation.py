from __future__ import annotations
from typing import List, Dict, Callable, Tuple, Optional
import math

Point = Tuple[float, float]

def _euclid(a: Point, b: Point) -> float:
    return math.hypot(a[0]-b[0], a[1]-b[1])

def _path_length(path: List[Point]) -> float:
    if not path or len(path) < 2:
        return 0.0
    return sum(_euclid(path[i-1], path[i]) for i in range(1, len(path)))

def _can_accept(truck, cfg) -> bool:
    """Return True if truck is eligible to receive a new assignment.
    We skip trucks that:
      - are already en‑route (have route_pts remaining)
      - have an assigned bin
      - are currently heading to a target (e.g., depot) to dump/recharge
      - are at or over capacity
    """
    if getattr(truck, 'assigned_bin', None):
        return False
    if getattr(truck, 'route_pts', None):
        # Has an active route to finish
        if len(truck.route_pts) - getattr(truck, 'route_i', 0) > 0:
            return False
    if getattr(truck, 'target', None) is not None:
        return False
    if truck.load >= cfg.get("TRUCK_CAPACITY", 300):
        return False
    # Post-depot dwell requirement
    if hasattr(truck, 'ready_after_depot') and not truck.ready_after_depot():
        return False
    return True

def negotiate(bins, trucks, t_now: float, cfg: Dict, plan_route: Callable[[Point, Point], List[Point]]):
    """
    Contract-net: eligible bins announce requests, trucks bid with cost; lowest bid wins.
    Returns a list of assignment events: {"truck": tid, "bin": bid}
    """
    # 1) Bins that should request service
    requests: List[Tuple[str, object]] = []
    # Track bins already targeted by an active truck to avoid duplicate assignment
    active_bins = {getattr(t, 'assigned_bin') for t in trucks if getattr(t, 'assigned_bin', None)}
    for b in bins:
        cap = getattr(b, 'capacity', cfg.get('BIN_CAPACITY', 100))
        last_t = getattr(b, 'last_service_t', -1e9)
        cooldown = float(cfg.get('SERVICE_COOLDOWN_S', 120.0))
        near_full = float(cfg.get('NEAR_FULL_FRAC', 0.9))
        horizon = float(cfg.get('URGENCY_HORIZON_S', 120.0))
        # Request if near full or overdue
        if b.id in active_bins:
            continue  # already being serviced / en-route
        if (b.fill >= near_full * cap) or ((t_now - last_t) >= horizon and b.fill > 0):
            requests.append((b.id, b))

    if not requests:
        return []

    # 2) For each request, collect bids from trucks
    assignments = []
    reserved_trucks: set[str] = set()
    for bid, bin_obj in requests:
        best = None
        curb = getattr(bin_obj, 'curb', None) or getattr(bin_obj, 'pos')
        for trk in trucks:
            if trk.tid in reserved_trucks:
                continue
            if not _can_accept(trk, cfg):
                continue
            # Travel cost estimate
            r = plan_route(trk.pos, curb)
            dist_m = _path_length(r)
            # Energy/reserve soft penalty
            reserve_m = float(cfg.get('ENERGY_RESERVE_M', 30.0))
            energy_left_m = float(trk.energy / max(cfg.get('ENERGY_PER_M', 0.05), 1e-6))
            reserve_pen = 0.0
            if energy_left_m < reserve_m + dist_m:
                reserve_pen = 1e4  # strongly discourage
            # Load soft penalty
            load_frac = trk.load / max(cfg.get('TRUCK_CAPACITY', 300), 1)
            load_pen = 100.0 * load_frac
            bid_cost = dist_m + reserve_pen + load_pen
            if (best is None) or (bid_cost < best[0]):
                best = (bid_cost, trk)
        if best is not None:
            _, winner = best
            assignments.append({"truck": winner.tid, "bin": bid})
            reserved_trucks.add(winner.tid)

    return assignments
