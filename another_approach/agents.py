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

    # ---------------------------------------------------------------------
    # Utilities
    # ---------------------------------------------------------------------
    def _can_return_to_depot(self, depot: Point) -> bool:
        meters_left = self.energy / max(1e-9, self.cfg["ENERGY_PER_M"])
        return meters_left >= (dist(self.pos, depot) + self.cfg["ENERGY_RESERVE_M"])

    def assign_target(self, route_pts, bin_id, final_target):
        # de-dupe route
        self.route_pts = [p for i, p in enumerate(route_pts)
                          if i == 0 or dist(route_pts[i - 1], p) > 1e-3]
        # drop leading point if it's basically current position
        if self.route_pts and dist(self.pos, self.route_pts[0]) < 0.5:
            self.route_i = 1
        else:
            self.route_i = 0
        self.assigned_bin = bin_id
        self.target = final_target
        self.state = "moving"

    def _move_along_route(self, dt: float) -> float:
        """Follow route_pts[route_i:] along roads. Returns incremental € cost (energy+maint)."""
        if not self.route_pts or self.route_i >= len(self.route_pts):
            self.state = "idle"
            return 0.0

        # current waypoint
        tgt = self.route_pts[self.route_i]
        c = self._move_towards(tgt, dt)

        # if close enough, advance to next
        if dist(self.pos, tgt) < 0.5:
            self.route_i += 1
            if self.route_i >= len(self.route_pts):
                self.state = "idle"  # arrived at final
        else:
            self.state = "moving"
        return c

    # ---------------------------------------------------------------------
    # Physics + cost bookkeeping
    # ---------------------------------------------------------------------
    def _move_towards(self, target: Point, dt: float) -> float:
        """Move toward target. Returns incremental cost in euros (energy + maintenance)."""
        dx, dy = target[0] - self.pos[0], target[1] - self.pos[1]
        d = math.hypot(dx, dy)
        if d < 1e-6:
            return 0.0
        step = min(d, self.cfg["TRUCK_SPEED_MPS"] * dt)
        nx = self.pos[0] + dx / d * step
        ny = self.pos[1] + dy / d * step
        self.pos = (nx, ny)

        # bookkeeping
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
    # RL-friendly discrete action interface
    # ---------------------------------------------------------------------
    def apply_action(self, action: int, bins: List[BinObj], depot: Point, cfg: dict) -> float:
        """
        Discrete action step with automatic service + automatic movement when a route exists.
        - Auto pickup when near assigned bin.
        - Auto unload at depot.
        - If carrying load and no target/route, auto-route to depot.
        - Movement proceeds every step if a route exists, unless action==WAIT (4).
        """
        dt = cfg["DT"]
        reward = 0.0

        # Wage tick for this truck
        wage_cost = (cfg["WAGE_PER_HOUR"] / 3600.0) * dt
        self.costs_eur["wage"] += wage_cost
        reward -= wage_cost

        thr = cfg.get("APPROACH_RADIUS_M", 3.0)

        # --- At depot: auto unload; recharge only on explicit action ---
        if dist(self.pos, depot) < 1.0:
            if self.load > 0:
                self.load = 0
                reward += 1.0  # unload shaping
            if action == 3 and self.energy < cfg["ENERGY_MAX"]:
                self.energy = cfg["ENERGY_MAX"]
                reward += 0.5

        # --- At assigned bin: auto pickup (cannot be blocked) ---
        if self.assigned_bin:
            b = next((bb for bb in bins if bb.id == self.assigned_bin), None)
            if b and dist(self.pos, b.pos) < thr and b.fill > 0 and self.load < cfg["TRUCK_CAPACITY"]:
                take = min(cfg["TRUCK_CAPACITY"] - self.load, b.fill)
                if take > 0:
                    self.load += take
                    b.fill -= take
                    reward += 0.1 * take

            # Done with this bin? (empty bin or truck full) -> head to depot
            if b and (b.fill == 0 or self.load >= cfg["TRUCK_CAPACITY"]):
                self.assigned_bin = None
                self.target = depot
                route = cfg.get("plan_route_fn")(self.pos, self.target) if "plan_route_fn" in cfg else [self.pos, self.target]
                self.assign_target(route, None, self.target)

        # --- If carrying load but no plan, auto plan to depot ---
        if self.load > 0 and not self.route_pts and self.target is None:
            self.target = depot
            route = cfg.get("plan_route_fn")(self.pos, self.target) if "plan_route_fn" in cfg else [self.pos, self.target]
            self.assign_target(route, None, self.target)

        # --- Automatic movement when a route exists (unless WAIT) ---
        # This is the key fix so trucks actually leave the depot / continue along routes.
        if action != 4:  # not WAIT
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

        # --- Out-of-energy penalty if stranded ---
        if self.energy <= 0 and dist(self.pos, depot) > 1.0:
            reward -= cfg.get("OUTAGE_PENALTY_EUR", 1000.0)

        # --- Small shaping toward assigned bin (optional) ---
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

        # depot logic
        if dist(self.pos, depot) < 1.0:
            if self.load > 0:
                events.append({"type": "drop", "truck": self.tid, "amount": self.load})
                self.load = 0
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
                    events.append({"type": "pickup", "truck": self.tid, "bin": b.id, "amount": take})
                if self.load >= self.cfg["TRUCK_CAPACITY"] or b.fill == 0:
                    self.assigned_bin = None
                    route = plan_route(self.pos, depot)
                    self.assign_target(route, None, depot)
                return events

        # motion toward target (only if target exists)
        if self.route_pts:
            self._move_along_route(dt)
        elif self.target:
            route = plan_route(self.pos, self.target)
            self.assign_target(route, self.assigned_bin, self.target)
            self._move_along_route(dt)

        return events
