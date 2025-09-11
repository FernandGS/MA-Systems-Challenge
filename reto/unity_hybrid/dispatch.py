from typing import List, Callable
from .agents import Truck, BinObj, dist


def compute_due_time(bin: BinObj, inflow_rate: float, now: float) -> float:
    remain = max(0, bin.capacity - bin.fill)
    eta = remain / max(1e-9, inflow_rate)
    return now + eta


def auction(bins: List[BinObj], trucks: List[Truck], now: float, cfg: dict, plan_route: Callable):
    inflow_rate = (cfg["BIN_FILL_PER_STEP"][1] + cfg["BIN_FILL_PER_STEP"][0]) / 2.0
    horizon = cfg.get("URGENCY_HORIZON_S", 100)
    thr = cfg.get("OPPORTUNISTIC_FILL_FRAC", 0.60)
    cooldown = float(cfg.get("SERVICE_COOLDOWN_S", 300.0))

    idle = [
        t for t in trucks
        if not t.assigned_bin and not t.route_pts and t.assign_hold_steps == 0
    ]
    if not idle:
        return []

    already_claimed = {t.assigned_bin for t in trucks if t.assigned_bin is not None}

    # Filter candidates: have trash, not already claimed, and not recently serviced
    cand = [
        b for b in bins
        if b.fill > 0
        and b.id not in already_claimed
        and (now - getattr(b, 'last_service_t', -1e9)) >= cooldown
    ]
    if not cand:
        return []

    def bin_priority(b: BinObj):
        due = compute_due_time(b, inflow_rate, now)
        urgent = 1 if ((due - now) < horizon) or (b.fill >= b.capacity) else 0
        fill_frac = b.fill / max(1, b.capacity)
        return (urgent, fill_frac)

    cand.sort(key=bin_priority, reverse=True)
    urgent_bins = [b for b in cand if bin_priority(b)[0] == 1]
    nonurgent_bins = [b for b in cand if bin_priority(b)[0] == 0 and (b.fill / b.capacity) >= thr]

    assigned_events = []

    def greedy_match(bin_list, idle_trucks, apply_cov: bool):
        nonlocal assigned_events
        free_trucks = idle_trucks[:]
        remaining_bins = [b for b in bin_list if b.id not in already_claimed]
        while free_trucks and remaining_bins:
            best_t, best_b, best_score = None, None, 1e18
            cov = float(cfg.get("COVERAGE_BIAS", 0.0))
            # Apply coverage bias optionally to spread trucks: prefer farther bins as cov increases
            # Score = (1-cov)*distance + cov*(maxDist - distance)
            for t in free_trucks:
                # precompute maxDist from this truck to candidate bins
                dists = [dist(t.pos, b.pos) for b in remaining_bins]
                if not dists:
                    continue
                maxd = max(dists)
                for b, d in zip(remaining_bins, dists):
                    if apply_cov and cov > 0:
                        score = (1.0 - cov) * d + cov * (maxd - d)
                    else:
                        score = d
                    if score < best_score:
                        best_t, best_b, best_score = t, b, score
            curb = getattr(best_b, "curb", best_b.pos)
            route = plan_route(best_t.pos, curb)
            # append precise curb point to minimize cutting corners
            if not route or route[-1] != curb:
                route = route + [curb]
            best_t.assign_target(route, best_b.id, curb)
            assigned_events.append({"type": "assign", "truck": best_t.tid, "bin": best_b.id})
            free_trucks.remove(best_t)
            remaining_bins = [b for b in remaining_bins if b.id != best_b.id]
        return free_trucks

    idle = greedy_match(urgent_bins, idle, apply_cov=True)
    if idle:
        idle = greedy_match(nonurgent_bins, idle, apply_cov=True)

    # fallback nearest
    remaining = [
        b for b in bins
        if b.fill > 0
        and b.id not in already_claimed
        and b.id not in {e['bin'] for e in assigned_events}
        and (now - getattr(b, 'last_service_t', -1e9)) >= cooldown
    ]
    for t in idle:
        if not remaining:
            break
        # coverage-biased choice among remaining
        cov = float(cfg.get("COVERAGE_BIAS", 0.0))
        if cov > 0 and len(remaining) > 1:
            dists = [dist(t.pos, bb.pos) for bb in remaining]
            maxd = max(dists)
            scores = [( (1.0 - cov) * d + cov * (maxd - d), i) for i, d in enumerate(dists)]
            _, idx = min(scores)
            b0 = remaining[idx]
        else:
            b0 = min(remaining, key=lambda bb: dist(t.pos, bb.pos))
        curb = getattr(b0, "curb", b0.pos)
        route = plan_route(t.pos, curb)
        if not route or route[-1] != curb:
            route = route + [curb]
        t.assign_target(route, b0.id, curb)
        assigned_events.append({"type": "assign", "truck": t.tid, "bin": b0.id})
        remaining = [b for b in remaining if b.id != b0.id]

    return assigned_events
