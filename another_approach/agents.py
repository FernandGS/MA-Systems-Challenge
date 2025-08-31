# agents.py
# Trucks and Bins. Handles physics, service, cost accounting, and RL reward stubs.

from dataclasses import dataclass, field
from typing import Tuple, Dict, List, Optional
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

    def step_fill(self, lo: int, hi: int, rnd) -> bool:
        """Fill this bin randomly. Return True if overflow happened."""
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
    v: float = 0.0
    heading: float = 0.0

    target: Optional[Point] = None
    assigned_bin: Optional[str] = None
    state: str = "idle"

    # bookkeeping
    km_total: float = 0.0
    kwh_total: float = 0.0
    costs_eur: Dict[str, float] = field(default_factory=lambda: {
        "wage": 0.0, "energy": 0.0, "maint": 0.0
    })
    route_pts: List[Point] = field(default_factory=list)
    route_i: int = 0

    # anti-churn / batching
    route_freeze_steps: int = 0
    assign_hold_steps: int = 0
    go_depot_lock_steps: int = 0
    stops_since_depot: int = 0

    # ---------------------------------------------------------------------
    # Utilities
    # ---------------------------------------------------------------------
    def _can_return_to_depot(self, depot: Point) -> bool:
        meters_left = self.energy / max(1e-9, self.cfg["ENERGY_PER_M"])
        return meters_left >= (dist(self.pos, depot) + self.cfg["ENERGY_RESERVE_M"])

    def assign_target(self, route_pts, bin_id, final_target):
        self.route_pts = [p for i, p in enumerate(route_pts)
                          if i == 0 or dist(route_pts[i - 1], p) > 1e-3]
        if self.route_pts and dist(self.pos, self.route_pts[0]) < 0.5:
            self.route_i = 1
        else:
            self.route_i = 0
        self.assigned_bin = bin_id
        self.target = final_target
        self.state = "moving"

        # freeze + hold windows to avoid churn
        self.route_freeze_steps = int(self.cfg.get("ROUTE_FREEZE_STEPS", 6))
        if bin_id is not None:
            self.assign_hold_steps = int(self.cfg.get("ASSIGN_HOLD_STEPS", 10))
        if final_target is not None and bin_id is None:
            self.go_depot_lock_steps = max(self.go_depot_lock_steps, int(self.cfg.get("DEPOT_LOCK_STEPS", 8)))

    def _move_along_route(self, dt: float) -> float:
        if not self.route_pts or self.route_i >= len(self.route_pts):
            self.state = "idle"
            return 0.0
        tgt = self.route_pts[self.route_i]
        c = self._move_towards(tgt, dt)
        if dist(self.pos, tgt) < 0.5:
            self.route_i += 1
            if self.route_i >= len(self.route_pts):
                self.state = "idle"
        else:
            self.state = "moving"
        return c

    def _move_towards(self, target: Point, dt: float) -> float:
        dx, dy = target[0] - self.pos[0], target[1] - self.pos[1]
        d = math.hypot(dx, dy)
        if d < 1e-6:
            return 0.0
        step = min(d, self.cfg["TRUCK_SPEED_MPS"] * dt)
        nx = self.pos[0] + dx / d * step
        ny = self.pos[1] + dy / d * step
        self.pos = (nx, ny)

        self.km_total += step / 1000.0
        e_used = step * self.cfg["ENERGY_PER_M"]
        self.energy -= e_used
        self.kwh_total += e_used

        energy_cost = e_used * self.cfg["ENERGY_EUR_PER_UNIT"]
        maint_cost = (step / 1000.0) * self.cfg["MAINT_EUR_PER_KM"]
        self.costs_eur["energy"] += energy_cost
        self.costs_eur["maint"] += maint_cost

        return energy_cost + maint_cost

    # ---------------------------------------------------------------------
    # RL-friendly discrete action interface (unchanged)
    # ---------------------------------------------------------------------
    def apply_action(self, action: int, bins: List[BinObj], depot: Point, cfg: dict) -> float:
        dt = cfg["DT"]
        reward = 0.0

        # counters tick (RL path)
        if self.route_freeze_steps > 0: self.route_freeze_steps -= 1
        if self.assign_hold_steps > 0:  self.assign_hold_steps  -= 1
        if self.go_depot_lock_steps > 0: self.go_depot_lock_steps -= 1

        wage_cost = (cfg["WAGE_PER_HOUR"] / 3600.0) * dt
        self.costs_eur["wage"] += wage_cost
        reward -= wage_cost

        thr = cfg.get("APPROACH_RADIUS_M", 3.0)

        if dist(self.pos, depot) < 1.0:
            if self.load > 0:
                dumped = self.load
                self.load = 0
                reward += 0.03 * dumped
                if self.stops_since_depot >= 2:
                    reward += 0.3
                self.stops_since_depot = 0
            if action == 3 and self.energy < cfg["ENERGY_MAX"]:
                self.energy = cfg["ENERGY_MAX"]
                reward += 0.1

        if self.assigned_bin:
            b = next((bb for bb in bins if bb.id == self.assigned_bin), None)
            if b and dist(self.pos, b.pos) < thr and b.fill > 0 and self.load < cfg["TRUCK_CAPACITY"]:
                take = min(cfg["TRUCK_CAPACITY"] - self.load, b.fill)
                if take > 0:
                    self.load += take
                    b.fill -= take
                    reward += 0.02 * take
                    self.stops_since_depot += 1

            if b:
                if self.load >= cfg["TRUCK_CAPACITY"]:
                    self.assigned_bin = None
                    self.target = depot
                    route = cfg.get("plan_route_fn")(self.pos, self.target) if "plan_route_fn" in cfg else [self.pos, self.target]
                    self.assign_target(route, None, self.target)
                elif b.fill == 0:
                    self.assigned_bin = None
                    self.target = None
                    self.route_pts = []
                    self.route_i = 0

        if (self.load > 0) and (not self.route_pts) and (self.target is None):
            must_recharge = not self._can_return_to_depot(depot)
            near_full_frac = float(cfg.get("NEAR_FULL_FRAC", 0.9))
            near_full = self.load >= near_full_frac * cfg["TRUCK_CAPACITY"]
            if near_full or must_recharge or self.go_depot_lock_steps > 0:
                self.target = depot
                route = cfg.get("plan_route_fn")(self.pos, self.target) if "plan_route_fn" in cfg else [self.pos, self.target]
                self.assign_target(route, None, self.target)

        if action != 4:
            if self.route_pts:
                c = self._move_along_route(dt)
                reward -= c
            elif self.target is not None:
                route = cfg.get("plan_route_fn")(self.pos, self.target) if "plan_route_fn" in cfg else [self.pos, self.target]
                self.assign_target(route, self.assigned_bin, self.target)
                c = self._move_along_route(dt)
                reward -= c
        else:
            self.state = "idle"

        if self.energy <= 0 and dist(self.pos, depot) > 1.0:
            reward -= cfg.get("OUTAGE_PENALTY_EUR", 1000.0)

        if self.assigned_bin:
            b = next((bb for bb in bins if bb.id == self.assigned_bin), None)
            if b:
                reward += 0.001 * max(0.0, 1.0 - dist(self.pos, b.pos) / 200.0)

        return reward

    # ---------------------------------------------------------------------
    # Baseline step (negotiation + simple sim)
    # ---------------------------------------------------------------------
    def step(self, dt: float, bins: List[BinObj], depot: Point, plan_route) -> List[dict]:
        """Perform one step in baseline sim. Returns events."""
        events = []

        # >>> FIX: tick down anti-churn counters also in baseline <<<
        if self.route_freeze_steps > 0: self.route_freeze_steps -= 1
        if self.assign_hold_steps > 0:  self.assign_hold_steps  -= 1
        if self.go_depot_lock_steps > 0: self.go_depot_lock_steps -= 1
        # ---------------------------------------------------------------

        # depot logic
        if dist(self.pos, depot) < 1.0:
            if self.load > 0:
                events.append({"type": "drop", "truck": self.tid, "amount": self.load})
                self.load = 0
                self.stops_since_depot = 0
            if self.energy < self.cfg["ENERGY_MAX"]:
                self.energy = self.cfg["ENERGY_MAX"]
                events.append({"type": "recharge", "truck": self.tid})

        # servicing bin
        thr = self.cfg.get("APPROACH_RADIUS_M", 3.0)
        for b in bins:
            if self.assigned_bin == b.id and dist(self.pos, b.pos) < thr and b.fill > 0:
                take = min(self.cfg["TRUCK_CAPACITY"] - self.load, b.fill)
                if take > 0:
                    self.load += take
                    b.fill -= take
                    self.stops_since_depot += 1
                    events.append({"type": "pickup", "truck": self.tid, "bin": b.id, "amount": take})

                if self.load >= self.cfg["TRUCK_CAPACITY"]:
                    self.assigned_bin = None
                    route = plan_route(self.pos, depot)
                    self.assign_target(route, None, depot)
                elif b.fill == 0:
                    self.assigned_bin = None
                    self.target = None
                    self.route_pts = []
                    self.route_i = 0
                return events

        # motion toward target (only if target exists)
        if self.route_pts:
            self._move_along_route(dt)
        elif self.target:
            route = plan_route(self.pos, self.target)
            self.assign_target(route, self.assigned_bin, self.target)
            self._move_along_route(dt)

        return events
