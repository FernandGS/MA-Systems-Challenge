from typing import List, Dict
import random, json
from agents import Truck, BinObj
from dispatch import auction
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
        if self.cfg.get("POLICY", "auction") == "dqn" and DQNManager is not None:
            try:
                self.rl = DQNManager(self.cfg)
            except Exception:
                # If DQN dependencies (torch/dqn_agent.py) are missing, fall back to auction
                self.rl = None
                self.cfg["POLICY"] = "auction"

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
            self.rl.start_step(self.trucks)
            assigns = self.rl.select_and_assign(self.city, self.bins, self.trucks, self.t, self._plan_route)
        else:
            assigns = auction(self.bins, self.trucks, self.t, self.cfg, self._plan_route)
        for ev in assigns:
            self.events.append({"t": self.t, "type": "assign", "truck": ev["truck"], "bin": ev["bin"]})
        # Lane assignment events (detect lateral offset on first segment)
        if self.cfg.get("ENABLE_LANES", False):
            for t in self.trucks:
                if t.route_pts and len(t.route_pts) >= 2 and any(ev.get("truck") == t.tid for ev in assigns):
                    p0, p1 = t.route_pts[0], t.route_pts[1]
                    dx, dy = p1[0]-p0[0], p1[1]-p0[1]
                    base_len = (dx*dx + dy*dy) ** 0.5
                    if base_len > 0:
                        # magnitude of offset relative to axis (heuristic: lane offset already applied)
                        self.events.append({"t": self.t, "type": "lane_assignment", "truck": t.tid})

        # 3. Trucks step
        step_events = []
        for t in self.trucks:
            for ev in t.step(dt, self.bins, self.city.depot, self._plan_route):
                ev["t"] = self.t
                # update bin cooldown timestamp on pickup
                if ev.get("type") == "pickup":
                    bid = ev.get("bin")
                    b = next((bb for bb in self.bins if bb.id == bid), None)
                    if b is not None:
                        b.last_service_t = self.t
                step_events.append(ev)
        self.events.extend(step_events)
        # RL learning at end of step
        if self.cfg.get("POLICY", "auction") == "dqn" and self.rl is not None:
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
