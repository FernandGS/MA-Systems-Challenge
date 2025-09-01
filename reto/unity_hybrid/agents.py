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
    # last simulation time when this bin was serviced; used for cooldown in dispatch
    last_service_t: float = -1.0

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

    def assign_target(self, route_pts: List[Point], bin_id: Optional[str], final_target: Optional[Point]):
        # de-dup small segments
        cleaned = []
        for p in route_pts:
            if not cleaned or (dist(cleaned[-1], p) > 1e-3):
                cleaned.append(p)
        self.route_pts = cleaned
        self.route_i = 0
        self.assigned_bin = bin_id
        self.target = final_target
        self.state = "moving"
        self.route_freeze_steps = int(self.cfg.get("ROUTE_FREEZE_STEPS", 6))
        if bin_id is not None:
            self.assign_hold_steps = int(self.cfg.get("ASSIGN_HOLD_STEPS", 10))
        if final_target is not None and bin_id is None:
            self.go_depot_lock_steps = max(self.go_depot_lock_steps, int(self.cfg.get("DEPOT_LOCK_STEPS", 8)))

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
        self.costs_eur["energy"] += e_used * self.cfg["ENERGY_EUR_PER_UNIT"]
        self.costs_eur["maint"] += (step / 1000.0) * self.cfg["MAINT_EUR_PER_KM"]
        return self.costs_eur["energy"] + self.costs_eur["maint"]

    def _move_along_route(self, dt: float):
        if not self.route_pts or self.route_i >= len(self.route_pts):
            self.state = "idle"; return
        tgt = self.route_pts[self.route_i]
        self._move_towards(tgt, dt)
        if dist(self.pos, tgt) < 0.4:
            self.route_i += 1
            if self.route_i >= len(self.route_pts):
                self.state = "idle"

    def step(self, dt: float, bins: List[BinObj], depot: Point, plan_route: Callable[[Point, Point], List[Point]]):
        # wage per tick
        self.costs_eur["wage"] += (self.cfg["WAGE_PER_HOUR"] / 3600.0) * dt
        
        # tick windows
        if self.route_freeze_steps > 0: self.route_freeze_steps -= 1
        if self.assign_hold_steps > 0:  self.assign_hold_steps  -= 1
        if self.go_depot_lock_steps > 0: self.go_depot_lock_steps -= 1

        # at depot: dump/recharge
        if dist(self.pos, depot) < 1.0:
            if self.load > 0:
                yield {"type": "drop", "truck": self.tid, "amount": self.load}
                self.load = 0
                self.stops_since_depot = 0
            if self.energy < self.cfg["ENERGY_MAX"]:
                self.energy = self.cfg["ENERGY_MAX"]
                yield {"type": "recharge", "truck": self.tid}
            # snap to exact depot to keep on-road
            self.pos = depot

        # service if at assigned bin
        thr = self.cfg.get("APPROACH_RADIUS_M", 3.0)
        # Ensure threshold isn't smaller than curb offset so trucks can service from the curb
        curb_allow = float(self.cfg.get("SIDEWALK_OFFSET_M", 2.0))
        thr = max(thr, curb_allow - 0.1)
        if self.assigned_bin:
            b = next((bb for bb in bins if bb.id == self.assigned_bin), None)
            # Consider truck "at bin" if within threshold of either curb or bin position
            at_bin = False
            if b:
                if b.curb is not None and dist(self.pos, b.curb) < thr:
                    at_bin = True
                elif dist(self.pos, b.pos) < thr:
                    at_bin = True
            if b and at_bin and b.fill > 0:
                # snap to curb to avoid off-road departure
                if b.curb is not None:
                    self.pos = b.curb
                take = min(self.cfg["TRUCK_CAPACITY"] - self.load, b.fill)
                if take > 0:
                    self.load += take; b.fill -= take; self.stops_since_depot += 1
                    # mark service time to help dispatch avoid immediate re-selection
                    try:
                        # current time is injected by Simulation.step before appending
                        current_t = None
                        yield_ev = {"type": "pickup", "truck": self.tid, "bin": b.id, "amount": take}
                        yield yield_ev
                        # If Simulation sets ev['t'] after yield, it can't update here; sim will set last_service_t
                    finally:
                        pass
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

        # move
        if self.route_pts:
            self._move_along_route(dt)
        elif self.target is not None:
            route = plan_route(self.pos, self.target)
            if not route or route[-1] != self.target:
                route = route + [self.target]
            self.assign_target(route, self.assigned_bin, self.target)
            self._move_along_route(dt)
