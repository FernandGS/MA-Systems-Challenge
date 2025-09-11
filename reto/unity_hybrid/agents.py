from dataclasses import dataclass, field
from typing import Tuple, Dict, List, Optional, Callable
import math

Point = Tuple[float, float]


def dist(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


@dataclass
class BinObj:
    id: str
    pos: Point
    capacity: int
    fill: int = 0
    curb: Optional[Point] = None
    last_service_t: float = -1e9

    def step_fill(self, lo: int, hi: int, rnd) -> bool:
        if self.fill < self.capacity:
            self.fill = min(self.capacity, self.fill + rnd.randint(lo, hi))
        return self.fill >= self.capacity


@dataclass
class Truck:
    tid: str
    pos: Point
    cfg: dict
    energy: float
    load: int = 0

    # motion state
    route_pts: List[Point] = field(default_factory=list)
    route_i: int = 0
    target: Optional[Point] = None
    assigned_bin: Optional[str] = None
    state: str = "idle"

    # bookkeeping
    km_total: float = 0.0
    kwh_total: float = 0.0
    costs_eur: Dict[str, float] = field(default_factory=lambda: {
        "wage": 0.0, "energy": 0.0, "maint": 0.0
    })

    # anti-churn
    route_freeze_steps: int = 0
    assign_hold_steps: int = 0
    go_depot_lock_steps: int = 0
    stops_since_depot: int = 0

    # dynamics
    _speed: float = 0.0
    _service_timer: float = 0.0
    _dump_timer: float = 0.0
    _recharge_rate: float = 0.0
    idle_steps: int = 0
    _at_depot_since: float = -1.0

    def _smooth_and_lane_shift(self, pts: List[Point]) -> List[Point]:
        if len(pts) < 2:
            return pts
        lane_offset = float(self.cfg.get("LANE_OFFSET_M", 2.0))
        turn_pull = float(self.cfg.get("TURN_PULL_M", 2.5))
        curve_pts = int(self.cfg.get("CURVE_INTERP_POINTS", 1))
        out: List[Point] = [pts[0]]
        # Evaluate each segment, add lane offsets for vertical motion, add curve smoothing at corners
        for i in range(1, len(pts)):
            prev = pts[i-1]; cur = pts[i]
            dx, dy = cur[0]-prev[0], cur[1]-prev[1]
            seg_len = math.hypot(dx, dy)
            if seg_len < 1e-6:
                continue
            is_vertical = abs(dx) < 1e-3 and abs(dy) > 1e-3
            lane = 0.0
            if is_vertical:
                if dy > 0: lane = lane_offset
                elif dy < 0: lane = -lane_offset
            # Apply lane offset only on vertical portion
            target_pt = (cur[0] + lane, cur[1]) if is_vertical else cur
            # Corner smoothing: if we have a next point (forming a corner), blend via simple quadratic Bezier
            if 0 < i < len(pts)-1:
                nxt = pts[i+1]
                ndx, ndy = nxt[0]-cur[0], nxt[1]-cur[1]
                if (abs(dx)>1e-3 and abs(ndx)<1e-3) or (abs(dy)>1e-3 and abs(ndy)<1e-3) or (dx*ndx + dy*ndy < 0):
                    # A change in direction; build pre and post approach points
                    pull_a = min(turn_pull, seg_len*0.5)
                    nxt_len = math.hypot(ndx, ndy)
                    pull_b = min(turn_pull, nxt_len*0.5)
                    ax = prev[0] + dx/seg_len * (seg_len - pull_a)
                    ay = prev[1] + dy/seg_len * (seg_len - pull_a)
                    bx = cur[0] + (ndx/max(1e-6,nxt_len)) * pull_b
                    by = cur[1] + (ndy/max(1e-6,nxt_len)) * pull_b
                    A = (ax, ay)
                    B = (cur[0], cur[1])
                    C = (bx, by)
                    # Append approach point if different
                    if dist(out[-1], A) > 1e-6:
                        out.append(A)
                    # Bezier interpolation
                    for k in range(1, curve_pts+1):
                        t = k/(curve_pts+1)
                        # Quadratic Bezier A->B->C
                        x1 = (1-t)*A[0] + t*B[0]
                        y1 = (1-t)*A[1] + t*B[1]
                        x2 = (1-t)*B[0] + t*C[0]
                        y2 = (1-t)*B[1] + t*C[1]
                        x = (1-t)*x1 + t*x2
                        y = (1-t)*y1 + t*y2
                        out.append((x,y))
                    # Add departure point
                    out.append(C)
                    continue
            out.append(target_pt)
        # De-duplicate near-identical consecutive points
        dedup = [out[0]]
        for p in out[1:]:
            if dist(dedup[-1], p) > 1e-3:
                dedup.append(p)
        return dedup

    def ready_after_depot(self) -> bool:
        """Return True if truck has satisfied post-depot dwell time and is free to receive tasks."""
        if self._at_depot_since < 0:
            return True
        dwell_req = float(self.cfg.get("DEPOT_MIN_DWELL_S", 0.0))
        return self._at_depot_since >= dwell_req

    def assign_target(self, route_pts: List[Point], bin_id: Optional[str], final_target: Optional[Point]):
        # Clean + smooth path + lane shifts
        cleaned = []
        for p in route_pts:
            if not cleaned or dist(cleaned[-1], p) > 1e-3:
                cleaned.append(p)
        self.route_pts = self._smooth_and_lane_shift(cleaned)
        self.route_i = 0
        self.assigned_bin = bin_id
        self.target = final_target
        self.state = "moving"
        self.route_freeze_steps = int(self.cfg.get("ROUTE_FREEZE_STEPS", 6))
        if bin_id is not None:
            self.assign_hold_steps = int(self.cfg.get("ASSIGN_HOLD_STEPS", 10))
        if final_target is not None and bin_id is None:
            self.go_depot_lock_steps = max(self.go_depot_lock_steps, int(self.cfg.get("DEPOT_LOCK_STEPS", 8)))

    def _desired_speed(self, next_pt: Point, after_pt: Optional[Point]) -> float:
        base = float(self.cfg.get("TRUCK_SPEED_MPS", 2.0))
        if after_pt is None:
            return base
        # If sharp turn ahead, slow down
        v1x, v1y = next_pt[0]-self.pos[0], next_pt[1]-self.pos[1]
        v2x, v2y = after_pt[0]-next_pt[0], after_pt[1]-next_pt[1]
        n1 = math.hypot(v1x, v1y); n2 = math.hypot(v2x, v2y)
        if n1 < 1e-6 or n2 < 1e-6:
            return base
        dot = (v1x*v2x + v1y*v2y)/(n1*n2)
        dot = max(-1.0, min(1.0, dot))
        angle = math.acos(dot)  # 0 straight, pi reversal
        if angle > math.radians(30):
            slow = float(self.cfg.get("CURVE_SLOW_FRAC", 0.5))
            return max(base*float(self.cfg.get("MIN_SPEED_FRACTION",0.4)), base*slow)
        return base

    def _move_towards(self, target: Point, after_pt: Optional[Point], dt: float) -> float:
        dx, dy = target[0] - self.pos[0], target[1] - self.pos[1]
        d = math.hypot(dx, dy)
        if d < 1e-6:
            return 0.0
        desired = self._desired_speed(target, after_pt)
        accel = float(self.cfg.get("ACCEL_MPS2", 1.0))
        # accelerate / decelerate
        if self._speed < desired:
            self._speed = min(desired, self._speed + accel*dt)
        else:
            self._speed = max(desired, self._speed - accel*dt)
        step = min(d, self._speed * dt)
        nx = self.pos[0] + dx/d * step
        ny = self.pos[1] + dy/d * step
        self.pos = (nx, ny)
        self.km_total += step/1000.0
        e_used = step * self.cfg.get("ENERGY_PER_M",0.06)
        self.energy -= e_used
        self.kwh_total += e_used
        self.costs_eur["energy"] += e_used * self.cfg.get("ENERGY_EUR_PER_UNIT",0.3)
        self.costs_eur["maint"] += (step/1000.0) * self.cfg.get("MAINT_EUR_PER_KM",0.06)
        return e_used

    def _move_along_route(self, dt: float):
        if not self.route_pts or self.route_i >= len(self.route_pts):
            self.state = "idle"
            return
        # Current waypoint and lookahead for corner speed adjustment
        tgt = self.route_pts[self.route_i]
        after = self.route_pts[self.route_i+1] if self.route_i+1 < len(self.route_pts) else None
        self._move_towards(tgt, after, dt)
        # Waypoint arrival check
        if dist(self.pos, tgt) < 0.4:
            self.route_i += 1
            if self.route_i >= len(self.route_pts):
                # Route finished
                self.state = "idle"
                # Clear target so dispatcher sees truck as free (prevents depot target from blocking reassignment)
                self.target = None
                # Reset route list to signal availability
                self.route_pts = []
                self.route_i = 0

    def step(self, dt: float, bins: List[BinObj], depot: Point, plan_route: Callable[[Point, Point], List[Point]]):
        # wage per tick
        self.costs_eur["wage"] += (self.cfg["WAGE_PER_HOUR"] / 3600.0) * dt
        
        # tick windows
        if self.route_freeze_steps > 0: self.route_freeze_steps -= 1
        if self.assign_hold_steps > 0:  self.assign_hold_steps  -= 1
        if self.go_depot_lock_steps > 0: self.go_depot_lock_steps -= 1

        # at depot: dump/recharge with timing
        if dist(self.pos, depot) < 1.0:
            self.pos = depot
            # If auto redeploy, suppress dwell timing
            if not self.cfg.get("AUTO_REDEPLOY_FROM_DEPOT", False):
                if self._at_depot_since < 0:
                    self._at_depot_since = 0.0
                else:
                    self._at_depot_since += dt
            else:
                self._at_depot_since = -1.0
            # Dump phase
            if self.load > 0:
                if self.cfg.get("INSTANT_DUMP", False):
                    # Immediate unload
                    yield {"type": "drop", "truck": self.tid, "amount": self.load}
                    self.load = 0
                    self.stops_since_depot = 0
                    self._dump_timer = 0.0
                else:
                    if self._dump_timer <= 0:
                        self._dump_timer = float(self.cfg.get("DUMP_TIME_S",5.0))
                    self._dump_timer -= dt
                    if self._dump_timer <= 0:
                        yield {"type": "drop", "truck": self.tid, "amount": self.load}
                        self.load = 0
                        self.stops_since_depot = 0
            else:
                self._dump_timer = 0.0
            # Recharge phase (gradual)
            if self.energy < self.cfg["ENERGY_MAX"]:
                rate = float(self.cfg.get("RECHARGE_RATE_PER_S", 20.0))
                self.energy = min(self.cfg["ENERGY_MAX"], self.energy + rate*dt)
                if abs(self.energy - self.cfg["ENERGY_MAX"]) < 1e-3:
                    yield {"type": "recharge", "truck": self.tid}

        # service if at assigned bin
        thr = self.cfg.get("APPROACH_RADIUS_M", 3.0)
        curb_allow = float(self.cfg.get("SIDEWALK_OFFSET_M", 2.0))
        thr = max(thr, curb_allow - 0.1)
        if self.assigned_bin:
            b = next((bb for bb in bins if bb.id == self.assigned_bin), None)
            at_bin = False
            if b:
                # Check both offset position and true bin/curb position
                if b.curb is not None and dist(self.pos, b.curb) < thr:
                    at_bin = True
                elif dist(self.pos, b.pos) < thr:
                    at_bin = True
                # If truck is offset but within threshold of bin/curb, snap to bin/curb for pickup
                if at_bin:
                    if b.curb is not None:
                        self.pos = b.curb
                    else:
                        self.pos = b.pos
            if b and at_bin and b.fill > 0:
                # Time-based service: drain gradually over SERVICE_TIME_S
                if self._service_timer <= 0:
                    self._service_timer = float(self.cfg.get("SERVICE_TIME_S", 4.0))
                # per-step amount proportional to remaining time
                dt_local = dt
                frac = min(1.0, dt_local / max(1e-6, self._service_timer))
                capacity_left = self.cfg["TRUCK_CAPACITY"] - self.load
                possible = min(capacity_left, b.fill)
                take = max(0, int(round(possible * frac)))
                self._service_timer -= dt_local
                if take > 0:
                    self.load += take; b.fill -= take; self.stops_since_depot += 1
                    yield {"type": "pickup", "truck": self.tid, "bin": b.id, "amount": take}
                if self._service_timer <= 0 or b.fill == 0 or self.load >= self.cfg["TRUCK_CAPACITY"]:
                    self._service_timer = 0.0
                if self.load >= self.cfg["TRUCK_CAPACITY"]:
                    self.assigned_bin = None
                    route = plan_route(self.pos, depot)
                    if not route or route[-1] != depot:
                        route = route + [depot]
                    self.assign_target(route, None, depot)
                elif b.fill == 0:
                    self.assigned_bin = None
                    self.target = None
                    self.route_pts = []
                    self.route_i = 0

        # auto-go-depot if carrying and not currently routed
        if (self.load > 0) and (not self.route_pts) and (self.target is None):
            near_full = self.load >= self.cfg.get("NEAR_FULL_FRAC",0.9) * self.cfg["TRUCK_CAPACITY"]
            low_energy = self.energy <= self.cfg.get("ENERGY_PER_M",0.06) * self.cfg.get("ENERGY_RESERVE_M",30.0)
            if near_full or low_energy or self.go_depot_lock_steps>0:
                self.target = depot
                route = plan_route(self.pos, depot)
                if not route or route[-1] != depot:
                    route = route + [depot]
                self.assign_target(route, None, depot)

        # Reset depot dwell marker if we leave depot vicinity
        if dist(self.pos, depot) >= 1.0 and self._at_depot_since >= 0:
            self._at_depot_since = -1.0

        # move
        if self.route_pts:
            self._move_along_route(dt)
        elif self.target is not None:
            route = plan_route(self.pos, self.target)
            if not route or route[-1] != self.target:
                route = route + [self.target]
            self.assign_target(route, self.assigned_bin, self.target)
            self._move_along_route(dt)

        # Track idleness for exploration feature
        if self.state == "idle" and not self.assigned_bin and self.target is None and not self.route_pts:
            self.idle_steps += 1
        else:
            self.idle_steps = 0
