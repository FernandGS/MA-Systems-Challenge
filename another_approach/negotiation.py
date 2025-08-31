# negotiation.py
# Auction-based negotiation between bins and trucks.

from typing import List
from agents import Truck, BinObj, dist

def compute_due_time(bin: BinObj, inflow_rate: float, now: float) -> float:
    """Estimate overflow time for a bin given expected inflow."""
    remain = max(0, bin.capacity - bin.fill)
    eta = remain / max(1e-9, inflow_rate)
    return now + eta

def bid_cost(truck: Truck, bin: BinObj, cfg: dict) -> float:
    """Monetary cost to service a bin."""
    d = dist(truck.pos, bin.pos)
    t_travel = d / cfg["TRUCK_SPEED_MPS"]
    cost = (cfg["WAGE_PER_HOUR"] / 3600.0) * t_travel
    cost += cfg["ENERGY_EUR_PER_UNIT"] * d * cfg["ENERGY_PER_M"]
    cost += cfg["MAINT_EUR_PER_KM"] * (d / 1000.0)
    return cost

def auction(bins: List[BinObj], trucks: List[Truck], now: float, cfg: dict, plan_route):
    """
    Assign bins to trucks with unique matching per tick:
    - Exclude bins already claimed by some truck (avoid dog-piling).
    - Prioritize full/urgent bins, then by fill fraction.
    - Greedy global matching on travel cost.
    - Fallback to nearest remaining non-empty bin.
    """
    inflow_rate = (cfg["BIN_FILL_PER_STEP"][1] + cfg["BIN_FILL_PER_STEP"][0]) / 2.0
    horizon = cfg.get("URGENCY_HORIZON_S", 100)
    thr = cfg.get("OPPORTUNISTIC_FILL_FRAC", 0.60)

    # trucks that are truly idle
    idle = [t for t in trucks if not t.assigned_bin and not (t.route_pts and t.route_i < len(t.route_pts))]
    if not idle:
        return

    # bins already claimed by any truck (persisting across ticks)
    already_claimed = {t.assigned_bin for t in trucks if t.assigned_bin is not None}

    # candidate bins: non-empty and not already claimed
    cand = [b for b in bins if b.fill > 0 and b.id not in already_claimed]
    if not cand:
        return

    # priority: urgent/full first; then by fill fraction
    def bin_priority(b: BinObj):
        due = compute_due_time(b, inflow_rate, now)
        urgent = 1 if ((due - now) < horizon) or (b.fill >= b.capacity) else 0
        fill_frac = b.fill / max(1, b.capacity)
        return (urgent, fill_frac)  # sort desc

    cand.sort(key=bin_priority, reverse=True)
    urgent_bins = [b for b in cand if bin_priority(b)[0] == 1]
    nonurgent_bins = [b for b in cand if bin_priority(b)[0] == 0 and (b.fill / b.capacity) >= thr]

    assigned_bins = set()

    def greedy_match(bin_list, idle_trucks):
        """Greedy min-distance matching; returns remaining idle trucks."""
        free_trucks = idle_trucks[:]  # copy
        remaining_bins = [b for b in bin_list if b.id not in assigned_bins]
        while free_trucks and remaining_bins:
            best_t, best_b, best_cost = None, None, float("inf")
            for t in free_trucks:
                for b in remaining_bins:
                    c = dist(t.pos, b.pos)
                    if c < best_cost:
                        best_t, best_b, best_cost = t, b, c
            # assign
            curb = getattr(best_b, "curb", best_b.pos)
            route = plan_route(best_t.pos, curb)
            best_t.assign_target(route, best_b.id, curb)
            assigned_bins.add(best_b.id)
            free_trucks.remove(best_t)
            remaining_bins = [b for b in remaining_bins if b.id != best_b.id]
        return free_trucks  # <— just return the list (no set ops)

    # 1) match urgent/full bins
    idle = greedy_match(urgent_bins, idle)
    if not idle:
        return

    # 2) opportunistic match (high fill bins)
    idle = greedy_match(nonurgent_bins, idle)
    if not idle:
        return

    # 3) fallback: any still-idle truck -> nearest remaining non-empty bin
    remaining = [b for b in bins if b.fill > 0 and b.id not in assigned_bins and b.id not in already_claimed]
    for t in idle:
        if not remaining:
            break
        b0 = min(remaining, key=lambda bb: dist(t.pos, bb.pos))
        curb = getattr(b0, "curb", b0.pos)
        route = plan_route(t.pos, curb)
        t.assign_target(route, b0.id, curb)
        assigned_bins.add(b0.id)
        remaining = [b for b in remaining if b.id != b0.id]
