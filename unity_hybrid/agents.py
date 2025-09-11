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
    # per-tick movement block (set by Simulation to avoid overlaps)
    block_move: bool = False

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

    # U-turn tracking
    last_pos: Point = (0.0, 0.0)
    last_move_dir: Point = (0.0, 0.0)
    # dwell counter to pause movement after servicing a bin
    pickup_dwell_steps: int = 0

    def assign_target(self, route_pts: List[Point], bin_id: Optional[str], final_target: Optional[Point]):
        # de-dup small segments
        cleaned: List[Point] = []
        for p in route_pts:
            if not cleaned or (dist(cleaned[-1], p) > 1e-3):
                cleaned.append(p)
        # Preserve original (un-offset) version for debug
        raw_cleaned = cleaned[:]
        lane_debug: List[dict] = []
        # Optional lane offset: keep to the right/left of the road based on travel direction
        if self.cfg.get("ENABLE_LANES", False) and len(cleaned) >= 2:
            off = float(self.cfg.get("LANE_OFFSET_M", 1.0))
            traffic_side = str(self.cfg.get("TRAFFIC_SIDE", "right")).lower()
            # +1 => offset to the right of travel, -1 => offset to the left of travel
            side_sign = 1.0 if traffic_side != "left" else -1.0
            lane_pts: List[Point] = []
            for i, p in enumerate(cleaned):
                # Direction vector from neighbor
                if i == 0:
                    nxt = cleaned[i+1]
                    dx, dy = nxt[0]-p[0], nxt[1]-p[1]
                else:
                    prev = cleaned[i-1]
                    dx, dy = p[0]-prev[0], p[1]-prev[1]
                l = math.hypot(dx, dy)
                if l < 1e-9:
                    lane_pts.append(p)
                    continue
                # Right-hand perpendicular to (dx,dy) is (dy, -dx)
                rx, ry = side_sign * dy, side_sign * (-dx)
                rl = math.hypot(rx, ry)
                ox, oy = (rx/rl) * off, (ry/rl) * off
                lane_debug.append({
                    "i": i,
                    "base_x": p[0], "base_y": p[1],
                    "dx": dx/l if l>0 else 0.0, "dy": dy/l if l>0 else 0.0,
                    "perp_x": (rx/rl) if rl>0 else 0.0, "perp_y": (ry/rl) if rl>0 else 0.0,
                    "off": off, "side_sign": side_sign,
                    "ox": ox, "oy": oy,
                    "final_x": p[0] + ox, "final_y": p[1] + oy
                })
                lane_pts.append((p[0] + ox, p[1] + oy))
            cleaned = lane_pts
        # Attach debug lane info to truck for exporter
        try:
            self.debug_lane = lane_debug  # type: ignore
            self.raw_route_debug = raw_cleaned  # original points before offset
        except Exception:
            pass
        # Ensure the very last waypoint is the exact final target (bin curb or depot)
        # so service/drop triggers using precise position.
        if final_target is not None:
            if not cleaned:
                cleaned = [final_target]
            else:
                if dist(cleaned[-1], final_target) > 1e-6:
                    cleaned.append(final_target)
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
        # If movement is blocked for this tick, do not advance position, but still accrue non-motion costs in step()
        if self.block_move or self.pickup_dwell_steps > 0:
            return 0.0
        dx, dy = target[0] - self.pos[0], target[1] - self.pos[1]
        d = math.hypot(dx, dy)
        if d < 1e-6:
            return 0.0
        step = min(d, self.cfg["TRUCK_SPEED_MPS"] * dt)
        nx = self.pos[0] + dx / d * step
        ny = self.pos[1] + dy / d * step
        self.last_pos = self.pos  # record previous before updating
        self.pos = (nx, ny)
        # store last movement direction (normalized) for U-turn detection
        if step > 0:
            self.last_move_dir = (dx / d, dy / d)
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
        # Prevent pure mid-road U-turn: if next waypoint equals previous waypoint and a config forbids, skip one step
        if self.route_i+1 < len(self.route_pts) and self.route_i>0:
            cur_wp = self.route_pts[self.route_i]
            prev_wp = self.route_pts[self.route_i-1]
            next_wp = self.route_pts[self.route_i+1]
            if (abs(prev_wp[0]-next_wp[0])<1e-6 and abs(prev_wp[1]-next_wp[1])<1e-6) and self.cfg.get("FORBID_UTURN_IF_ALTERNATIVE", True):
                # Detected immediate backtrack; fast-forward to avoid oscillation
                self.route_i += 1
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
        # decrement service dwell if active
        if self.pickup_dwell_steps > 0:
            self.pickup_dwell_steps -= 1

        # Depot docking: allow drop/recharge when intentionally arriving to depot or carrying load
        near_dep = dist(self.pos, depot) <= float(self.cfg.get("DEPOT_APPROACH_RADIUS_M", 1.0))
        arriving_to_depot = (self.target == depot) or (self.route_pts and self.route_pts[-1] == depot)
        if near_dep and (arriving_to_depot or self.load > 0):
            did_service = False
            if self.load > 0:
                yield {"type": "drop", "truck": self.tid, "amount": self.load}
                self.load = 0
                self.stops_since_depot = 0
                did_service = True
            if self.energy < self.cfg["ENERGY_MAX"]:
                self.energy = self.cfg["ENERGY_MAX"]
                yield {"type": "recharge", "truck": self.tid}
                did_service = True
            if did_service or arriving_to_depot:
                # Snap near depot, but use a small, deterministic per-truck offset to avoid stacking
                try:
                    numeric_id = int(''.join(ch for ch in self.tid if ch.isdigit()))
                except Exception:
                    numeric_id = 0
                n_trucks = int(self.cfg.get("N_TRUCKS", 4))
                angle = (numeric_id % max(1, n_trucks)) * (2.0 * math.pi / max(1, n_trucks))
                radius = float(self.cfg.get("DEPOT_QUEUE_RADIUS_M", 0.8))
                self.pos = (depot[0] + radius * math.cos(angle), depot[1] + radius * math.sin(angle))
                # Ready for next assignment
                self.target = None
                self.route_pts = []
                self.route_i = 0
                self.state = "idle"
                # Clear holds/locks so dispatcher can pick this truck immediately
                self.assign_hold_steps = 0
                self.route_freeze_steps = 0
                self.go_depot_lock_steps = 0
                self.assigned_bin = None

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
                # Snap near curb with a tiny per-truck offset ring to avoid stacking
                if b.curb is not None:
                    try:
                        numeric_id = int(''.join(ch for ch in self.tid if ch.isdigit()))
                    except Exception:
                        numeric_id = 0
                    n_trucks = int(self.cfg.get("N_TRUCKS", 4))
                    angle = (numeric_id % max(1, n_trucks)) * (2.0 * math.pi / max(1, n_trucks))
                    radius = float(self.cfg.get("CURB_QUEUE_RADIUS_M", 0.6))
                    self.pos = (b.curb[0] + radius * math.cos(angle), b.curb[1] + radius * math.sin(angle))
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
                    # Start a brief dwell to create a visible stop
                    dwell_ticks = int(self.cfg.get("PICKUP_DWELL_TICKS", 0))
                    if dwell_ticks > 0:
                        self.pickup_dwell_steps = dwell_ticks
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
        # If carrying and very close to depot but not moving (e.g., blocked), force a short nudge towards depot
        if self.load > 0 and dist(self.pos, depot) <= float(self.cfg.get("DEPOT_FREE_RADIUS_M", 0.0)) and not self.route_pts:
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
