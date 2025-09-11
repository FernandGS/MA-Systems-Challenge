from typing import List, Callable, Dict, Tuple
from agents import Truck, BinObj, dist


def compute_due_time(bin: BinObj, inflow_rate: float, now: float) -> float:
    remain = max(0, bin.capacity - bin.fill)
    eta = remain / max(1e-9, inflow_rate)
    return now + eta


def auction(bins: List[BinObj], trucks: List[Truck], now: float, cfg: dict, plan_route: Callable):
    inflow_rate = (cfg["BIN_FILL_PER_STEP"][1] + cfg["BIN_FILL_PER_STEP"][0]) / 2.0
    horizon = cfg.get("URGENCY_HORIZON_S", 100)
    thr = cfg.get("OPPORTUNISTIC_FILL_FRAC", 0.60)
    cooldown = float(cfg.get("SERVICE_COOLDOWN_S", 300.0))
    cooldown_min = float(cfg.get("SERVICE_COOLDOWN_MIN_S", cooldown * 0.25))

    idle = [
        t for t in trucks
        if not t.assigned_bin and not t.route_pts and t.assign_hold_steps == 0
    ]
    if not idle:
        return []

    already_claimed = {t.assigned_bin for t in trucks if t.assigned_bin is not None}

    # Filter candidates: have trash, not already claimed, and not recently serviced
    cand = []
    for b in bins:
        if b.id in already_claimed:
            continue
        if b.fill <= 0:
            continue
        time_since = (now - getattr(b, 'last_service_t', -1e9))
        fill_frac = b.fill / max(1, b.capacity)
        due = compute_due_time(b, inflow_rate, now)
        urgent = ((due - now) < horizon) or (b.fill >= b.capacity)
        # Allow urgent or high-fill bins to bypass cooldown partially
        if urgent or fill_frac >= max(thr, 0.6):
            if time_since >= cooldown_min:
                cand.append(b)
        else:
            if time_since >= cooldown:
                cand.append(b)
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

    min_gap = int(cfg.get("MIN_FOLLOW_GAP_STEPS", 0))
    extra_hold = int(cfg.get("ANTI_TAILGATE_EXTRA_HOLD", 0))
    occ = {(round(t.pos[0],1), round(t.pos[1],1)) for t in trucks if t.route_pts}

    def greedy_match(bin_list, idle_trucks, apply_cov: bool):
        nonlocal assigned_events
        free_trucks = idle_trucks[:]
        remaining_bins = [b for b in bin_list if b.id not in already_claimed]
        while free_trucks and remaining_bins:
            best_t, best_b, best_score = None, None, 1e18
            cov = float(cfg.get("COVERAGE_BIAS", 0.0))
            repel_w = float(cfg.get("PLATOON_REPEL_WEIGHT", 0.0))
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
                    # Repulsion term: penalize bins near other trucks
                    if repel_w > 0:
                        repel = 0.0
                        for ot in trucks:
                            if ot is t:
                                continue
                            dd = dist(ot.pos, b.pos)
                            repel += 1.0 / max(0.5, dd)  # bounded
                        score += repel_w * repel
                    if score < best_score:
                        best_t, best_b, best_score = t, b, score
            curb = getattr(best_b, "curb", best_b.pos)
            route = plan_route(best_t.pos, curb)
            # append precise curb point to minimize cutting corners
            if not route or route[-1] != curb:
                route = route + [curb]
            # Anti-tailgating: if first waypoint already occupied by moving truck, delay assignment
            if min_gap > 0 and route:
                first_wp = route[0]
                if (round(first_wp[0],1), round(first_wp[1],1)) in occ:
                    best_t.assign_hold_steps = max(best_t.assign_hold_steps, min_gap + extra_hold)
                    assigned_events.append({"type": "tailgate_hold", "truck": best_t.tid, "bin": best_b.id})
            best_t.assign_target(route, best_b.id, curb)
            assigned_events.append({"type": "assign", "truck": best_t.tid, "bin": best_b.id})
            free_trucks.remove(best_t)
            remaining_bins = [b for b in remaining_bins if b.id != best_b.id]
        return free_trucks

    idle = greedy_match(urgent_bins, idle, apply_cov=True)
    if idle:
        idle = greedy_match(nonurgent_bins, idle, apply_cov=True)

    # fallback nearest
    assigned_ids = {e['bin'] for e in assigned_events}
    remaining = []
    for b in bins:
        if b.id in already_claimed or b.id in assigned_ids:
            continue
        if b.fill <= 0:
            continue
        time_since = (now - getattr(b, 'last_service_t', -1e9))
        fill_frac = b.fill / max(1, b.capacity)
        due = compute_due_time(b, inflow_rate, now)
        urgent = ((due - now) < horizon) or (b.fill >= b.capacity)
        if urgent or fill_frac >= max(thr, 0.6):
            if time_since >= cooldown_min:
                remaining.append(b)
        else:
            if time_since >= cooldown:
                remaining.append(b)
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
        if min_gap > 0 and route:
            first_wp = route[0]
            if (round(first_wp[0],1), round(first_wp[1],1)) in occ:
                t.assign_hold_steps = max(t.assign_hold_steps, min_gap + extra_hold)
                assigned_events.append({"type": "tailgate_hold", "truck": t.tid, "bin": b0.id})
        t.assign_target(route, b0.id, curb)
        assigned_events.append({"type": "assign", "truck": t.tid, "bin": b0.id})
        remaining = [b for b in remaining if b.id != b0.id]

    return assigned_events


def market(bins: List[BinObj], trucks: List[Truck], now: float, cfg: dict, plan_route: Callable):
    """Multi-round Contract-Net-style negotiation between idle trucks and candidate bins.

    - Bins broadcast requests if eligible (urgency/cooldown/claimed filters like auction)
    - Idle trucks propose to their best bin (score = value(bin) - beta*distance)
    - Each bin accepts the highest score; ties broken by lower distance then truck id
    - Multiple rounds allow unassigned trucks to try next-best bins
    """
    inflow_rate = (cfg["BIN_FILL_PER_STEP"][1] + cfg["BIN_FILL_PER_STEP"][0]) / 2.0
    horizon = cfg.get("URGENCY_HORIZON_S", 100)
    thr = cfg.get("OPPORTUNISTIC_FILL_FRAC", 0.60)
    cooldown = float(cfg.get("SERVICE_COOLDOWN_S", 300.0))
    cooldown_min = float(cfg.get("SERVICE_COOLDOWN_MIN_S", cooldown * 0.25))

    max_rounds = int(cfg.get("MARKET_ROUNDS", 3))
    beta_cost = float(cfg.get("MARKET_BETA_COST", 1.0))
    val_urgent = float(cfg.get("MARKET_VALUE_URGENT", 1.0))
    w_fill = float(cfg.get("MARKET_VALUE_FILL_W", 0.5))

    # Idle trucks only
    idle: List[Truck] = [
        t for t in trucks
        if not t.assigned_bin and not t.route_pts and t.assign_hold_steps == 0
    ]
    if not idle:
        return []

    already_claimed = {t.assigned_bin for t in trucks if t.assigned_bin is not None}

    # Eligible bins
    elig: List[BinObj] = []
    for b in bins:
        if b.id in already_claimed:
            continue
        if b.fill <= 0:
            continue
        time_since = (now - getattr(b, 'last_service_t', -1e9))
        fill_frac = b.fill / max(1, b.capacity)
        due = compute_due_time(b, inflow_rate, now)
        urgent = ((due - now) < horizon) or (b.fill >= b.capacity)
        if urgent or fill_frac >= max(thr, 0.6):
            if time_since >= cooldown_min:
                elig.append(b)
        else:
            if time_since >= cooldown:
                elig.append(b)
    if not elig:
        return []

    # Precompute bin values
    def bin_value(b: BinObj) -> float:
        due = compute_due_time(b, inflow_rate, now)
        urgent = 1.0 if ((due - now) < horizon) or (b.fill >= b.capacity) else 0.0
        fill_frac = b.fill / max(1, b.capacity)
        return val_urgent * urgent + w_fill * fill_frac

    assigned_events: List[Dict] = []

    # Set of bins still open for proposals
    open_bins: Dict[int, BinObj] = {b.id: b for b in elig}
    assigned_trucks: set[str] = set()

    for _round in range(max_rounds):
        # Proposals: bin_id -> list of (score, -dist, truck)
        proposals: Dict[int, List[Tuple[float, float, Truck]]] = {}
        for t in idle:
            if t.tid in assigned_trucks:
                continue
            best_bid = None
            best_score = -1e18
            best_dist = 1e18
            for b in open_bins.values():
                # compute distance cost
                curb = getattr(b, "curb", b.pos)
                d = dist(t.pos, curb)
                score = bin_value(b) - beta_cost * d
                if score > best_score or (isinstance(score, float) and abs(score - best_score) < 1e-9 and d < best_dist):
                    best_score = score
                    best_dist = d
                    best_bid = b
            if best_bid is not None:
                proposals.setdefault(best_bid.id, []).append((best_score, -best_dist, t))

        if not proposals:
            break

        # Accept highest score per bin
        winners: Dict[int, Truck] = {}
        for bid, plist in proposals.items():
            plist.sort(reverse=True)
            winners[bid] = plist[0][2]

        # Assign and remove taken bins
        for bid, t in winners.items():
            if t.tid in assigned_trucks:
                continue
            b = open_bins.get(bid)
            if b is None:
                continue
            curb = getattr(b, "curb", b.pos)
            route = plan_route(t.pos, curb)
            if not route or route[-1] != curb:
                route = route + [curb]
            t.assign_target(route, b.id, curb)
            assigned_events.append({"type": "assign", "truck": t.tid, "bin": b.id})
            assigned_trucks.add(t.tid)
            open_bins.pop(bid, None)

        # Stop if no more capacity
        if len(open_bins) == 0:
            break
        if len(assigned_trucks) == len(idle):
            break

    return assigned_events
