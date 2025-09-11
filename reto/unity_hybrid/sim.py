from typing import List, Dict
import random, json
from .agents import Truck, BinObj
from .dispatch import auction
from .negotiation import negotiate
from .rl_tabular import QManager
try:
    from rl_policy import DQNManager
except Exception:
    DQNManager = None  # type: ignore

class Simulation:
    def __init__(self, cfg, city, planner="graph", grid_passable=None):
        self.cfg = cfg
        self.city = city
        self.t = 0.0
        self.planner = planner
        # If planner is grid and no mask passed, rasterize from city roads
        if planner == "grid" and grid_passable is None:
            try:
                from .grid_planner import city_to_grid_mask
                N = int(cfg.get("GRID_SIZE", 150))
                self.grid_passable = city_to_grid_mask(city, N)
            except Exception:
                self.grid_passable = grid_passable
        else:
            self.grid_passable = grid_passable

        # Bins
        self.bins: List[BinObj] = [
            BinObj(b["id"], b["pos"], b["capacity"], b["fill"], b.get("curb")) for b in city.bins
        ]

        # Trucks
        self.trucks: List[Truck] = []
        for i in range(cfg["N_TRUCKS"]):
            t = Truck(tid=f"T{i}", pos=city.depot, cfg=cfg, energy=cfg["ENERGY_MAX"]) 
            self.trucks.append(t)

        # Logs
        self.frames: List[Dict] = []
        self.events: List[Dict] = []
        # RL manager if using DQN (graceful fallback if unavailable)
        self.rl: object | None = None
        if self.cfg.get("POLICY", "auction") == "dqn":
            # Try DQN first, fallback to tabular
            if DQNManager is not None:
                try:
                    self.rl = DQNManager(self.cfg)
                except Exception:
                    self.rl = QManager(self.cfg)
            else:
                self.rl = QManager(self.cfg)

    def _rnd(self):
        return random.Random(int(self.t) ^ self.cfg["SEED"])

    def _fill_bins(self):
        lo, hi = self.cfg["BIN_FILL_PER_STEP"]
        rnd = self._rnd()
        overflows = 0
        for b in self.bins:
            before = b.fill
            if b.step_fill(lo, hi, rnd) and before < b.capacity:
                overflows += 1
                self.events.append({"t": self.t, "type": "overflow", "bin": b.id})
        return overflows

    def _plan_route(self, start, goal):
        if self.planner == "grid" and self.grid_passable is not None:
            # grid A* fallback: use simple Manhattan if A* fails
            from .grid_planner import astar, manhattan_path
            s = (int(round(start[0])), int(round(start[1])))
            g = (int(round(goal[0])), int(round(goal[1])))
            path = astar(s, g, self.grid_passable)
            if not path:
                path = manhattan_path(s, g)
            # convert to float points
            return [(float(x), float(y)) for (x,y) in path]
        return self.city.plan_route(start, goal)

    def step(self):
        dt = self.cfg["DT"]

        # 1. Bin fill + penalties
        new_ov = self._fill_bins()
        if new_ov > 0:
            # accumulate penalties on events consumer; exporter will tally
            pass

        # 2. Assignment policy (auction or DQN)
        assigns = []
        if self.cfg.get("POLICY", "auction") == "dqn" and self.rl is not None:
            # Hybrid: negotiation produces candidate requests; RL chooses among options
            # First get negotiation suggestions
            nego = negotiate(self.bins, self.trucks, self.t, self.cfg, self._plan_route)
            # RL layer can add/override decisions as actions
            rl_assigns = self.rl.select_and_assign(self.city, self.bins, self.trucks, self.t, self._plan_route)
            # Merge: prefer RL for trucks it mentions, otherwise use negotiation
            taken = {ev["truck"] for ev in rl_assigns}
            assigns = rl_assigns + [ev for ev in nego if ev["truck"] not in taken]
        else:
            # Auction replaced by negotiation for clarity
            assigns = negotiate(self.bins, self.trucks, self.t, self.cfg, self._plan_route)
        for ev in assigns:
            tid = ev.get("truck")
            bid = ev.get("bin")
            trk = next((tt for tt in self.trucks if tt.tid == tid), None)
            if trk is None:
                continue
            # Skip if truck is already moving toward something to avoid constant reassign churn
            if trk.assigned_bin or trk.route_pts or trk.target is not None:
                continue
            if bid is None:
                curb = self.city.depot
                route = self._plan_route(trk.pos, curb)
                if not route or route[-1] != curb:
                    route = route + [curb]
                trk.assign_target(route, None, curb)
            else:
                b = next((bb for bb in self.bins if str(bb.id) == str(bid)), None)
                if b is None:
                    continue
                curb = getattr(b, 'curb', None) or b.pos
                route = self._plan_route(trk.pos, curb)
                if not route or route[-1] != curb:
                    route = route + [curb]
                trk.assign_target(route, b.id, curb)
            self.events.append({"t": self.t, "type": "assign", "truck": tid, "bin": bid})

        # 3. Trucks step
        step_events = []
        for t in self.trucks:
            t.prev_pos = t.pos
            t.prev_load = t.load
            for ev in t.step(dt, self.bins, self.city.depot, self._plan_route):
                ev["t"] = self.t
                # update bin cooldown timestamp on pickup
                if ev.get("type") == "pickup":
                    bid = ev.get("bin")
                    b = next((bb for bb in self.bins if bb.id == bid), None)
                    if b is not None:
                        b.last_service_t = self.t
                step_events.append(ev)
            # distance traveled for reward
            try:
                from math import hypot
                t.last_step_dist = hypot(t.pos[0]-t.prev_pos[0], t.pos[1]-t.prev_pos[1])
            except Exception:
                t.last_step_dist = 0.0

        # 3b. Simple collision avoidance/separation: push trucks apart if too close
        safe = float(self.cfg.get("SAFE_DISTANCE_M", 3.0))
        from math import hypot
        n = len(self.trucks)
        for i in range(n):
            for j in range(i+1, n):
                ti, tj = self.trucks[i], self.trucks[j]
                dx = tj.pos[0]-ti.pos[0]
                dy = tj.pos[1]-ti.pos[1]
                d = hypot(dx, dy)
                if d < 1e-6:
                    # exact overlap: nudge arbitrarily
                    nx, ny = 1.0, 0.0
                else:
                    nx, ny = dx/d, dy/d
                if d < safe:
                    # move each half the overlap distance away from the other
                    overlap = (safe - d) * 0.5
                    ti.pos = (ti.pos[0] - nx * overlap, ti.pos[1] - ny * overlap)
                    tj.pos = (tj.pos[0] + nx * overlap, tj.pos[1] + ny * overlap)
        self.events.extend(step_events)
        # RL learning at end of step
        if self.cfg.get("POLICY", "auction") == "dqn" and self.rl is not None:
            # If DQNManager implements these, call; otherwise tabular manager does
            if hasattr(self.rl, 'end_step_and_learn'):
                self.rl.end_step_and_learn(self.city, self.bins, self.trucks, self.t, step_events)

        # 4. Log frame
        frame = {
            "t": self.t,
            "trucks": [
                {
                    "id": t.tid, "x": t.pos[0], "y": t.pos[1],
                    "energy": t.energy, "load": t.load, "state": t.state,
                    "target": (None if t.target is None else {"x": t.target[0], "y": t.target[1]}),
                } for t in self.trucks
            ],
            "bins": [
                {"id": b.id, "x": b.pos[0], "y": b.pos[1], "fill": b.fill, "cap": b.capacity}
                for b in self.bins
            ],
        }
        self.frames.append(frame)

        # 5. Advance time
        self.t += dt

    def run(self, steps: int):
        for _ in range(steps):
            self.step()
