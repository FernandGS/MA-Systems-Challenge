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
            # Prevent assigning same bin twice in same step
            if bid is not None:
                if any(t.assigned_bin == bid for t in self.trucks):
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

        # 2b. Exploration: send idle trucks on patrol routes
        if self.cfg.get("EXPLORATION_ENABLED", False):
            # Skip if DQN policy and exploration disabled for DQN
            if not (self.cfg.get("POLICY") == "dqn" and not self.cfg.get("EXPLORATION_ALLOW_WITH_DQN", False)):
                import random, math
                idle_thresh = int(self.cfg.get("EXPLORATION_IDLE_THRESHOLD_STEPS", 3))
                prob = float(self.cfg.get("EXPLORATION_PROB", 0.5))
                min_dist = float(self.cfg.get("EXPLORATION_MIN_DIST", 20.0))
                prefer_farthest = bool(self.cfg.get("EXPLORATION_PREFER_FARTHEST", True))
                reserve_mult = float(self.cfg.get("EXPLORATION_ENERGY_RESERVE_MULT", 1.2))
                energy_per_m = float(self.cfg.get("ENERGY_PER_M", 0.06))
                reserve_m = float(self.cfg.get("ENERGY_RESERVE_M", 30.0)) * reserve_mult
                waypoints = self.city.waypoints
                for trk in self.trucks:
                    if trk.assigned_bin or trk.route_pts or trk.target is not None:
                        continue
                    if trk.idle_steps < idle_thresh:
                        continue
                    # Energy check: require enough distance margin
                    est_range_m = trk.energy / max(1e-9, energy_per_m)
                    if est_range_m < reserve_m:
                        continue
                    if random.random() > prob:
                        continue
                    # Candidate waypoints beyond current min distance
                    cx, cy = trk.pos
                    candidates = []
                    for w in waypoints:
                        d = math.hypot(w[0]-cx, w[1]-cy)
                        if d >= min_dist:
                            candidates.append((d, w))
                    if not candidates:
                        continue
                    if prefer_farthest:
                        candidates.sort(reverse=True, key=lambda x: x[0])
                        target_wp = candidates[0][1]
                    else:
                        target_wp = random.choice(candidates)[1]
                    route = self._plan_route(trk.pos, target_wp)
                    if not route or route[-1] != target_wp:
                        route = route + [target_wp]
                    trk.assign_target(route, None, target_wp)
                    self.events.append({"t": self.t, "type": "explore", "truck": trk.tid, "bin": None})

        # 3. Trucks step
        step_events: List[Dict] = []
        dump_trucks = set()
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
                if ev.get("type") == "drop":
                    dump_trucks.add(t.tid)
                step_events.append(ev)
            # distance traveled for reward
            try:
                from math import hypot
                t.last_step_dist = hypot(t.pos[0]-t.prev_pos[0], t.pos[1]-t.prev_pos[1])
            except Exception:
                t.last_step_dist = 0.0

        # 3b. Separation: enforce minimum distance except during warmup/cooldown or at depot (if allowed)
        safe = float(self.cfg.get("SAFE_DISTANCE_M", 3.0))
        warmup = int(self.cfg.get("SEPARATION_WARMUP_STEPS", 0))
        cooldown = int(self.cfg.get("SEPARATION_COOLDOWN_STEPS", 0))
        skip_at_depot = bool(self.cfg.get("SEPARATION_SKIP_AT_DEPOT", True))
        remaining_steps = float('inf')  # unknown total; if run() passes fixed steps we can't know here, so use cooldown only if frames length hints total
        # If caller uses run(steps), we can approximate target total by last frame index + projected remaining
        # (We skip implementing dynamic detection; cooldown applies only if configured as 0 -> disabled)
        apply_sep = True
        if self.t < warmup:
            apply_sep = False
        # cooldown disabled unless explicitly set and we somehow detect near end (not available here)
        if apply_sep and safe > 0:
            from math import hypot
            n = len(self.trucks)
            for i in range(n):
                for j in range(i+1, n):
                    ti, tj = self.trucks[i], self.trucks[j]
                    dx = tj.pos[0]-ti.pos[0]
                    dy = tj.pos[1]-ti.pos[1]
                    d = hypot(dx, dy)
                    if d <= 0:
                        # exact overlap: random tiny jitter based on ids
                        h = (hash(ti.tid) & 7) - 3
                        k = (hash(tj.tid) & 7) - 3
                        dx, dy = (h or 1)*0.01, (k or 1)*0.01
                        d = hypot(dx, dy)
                    nx, ny = dx/d, dy/d
                    # Optionally skip separation if both near depot
                    if skip_at_depot:
                        depot = self.city.depot
                        dep_thr = safe * 0.75
                        if (hypot(ti.pos[0]-depot[0], ti.pos[1]-depot[1]) < dep_thr and
                            hypot(tj.pos[0]-depot[0], tj.pos[1]-depot[1]) < dep_thr):
                            continue
                    if d < safe:
                        overlap = safe - d
                        # push proportionally; each truck moves half
                        shift = overlap * 0.5
                        ti.pos = (ti.pos[0] - nx * shift, ti.pos[1] - ny * shift)
                        tj.pos = (tj.pos[0] + nx * shift, tj.pos[1] + ny * shift)
        # Immediate post-dump assignment (optional)
        if self.cfg.get("IMMEDIATE_POST_DUMP_ASSIGN", False) and dump_trucks:
            # Build list of candidate bins (any non-empty) not already claimed
            already = {tr.assigned_bin for tr in self.trucks if tr.assigned_bin}
            # Avoid reassigning if truck already got a route in same step somehow
            for t in (tt for tt in self.trucks if tt.tid in dump_trucks and not tt.assigned_bin and not tt.route_pts and tt.load == 0):
                # choose fullest bin (break ties by distance)
                cands = [b for b in self.bins if b.fill > 0 and b.id not in already and all(tt.assigned_bin != b.id for tt in self.trucks)]
                if not cands:
                    continue
                cands.sort(key=lambda b: (-b.fill, ( (t.pos[0]-b.pos[0])**2 + (t.pos[1]-b.pos[1])**2 )))
                b0 = cands[0]
                curb = getattr(b0, 'curb', b0.pos)
                route = self._plan_route(t.pos, curb)
                if not route or route[-1] != curb:
                    route = route + [curb]
                t.assign_target(route, b0.id, curb)
                self.events.append({"t": self.t, "type": "assign", "truck": t.tid, "bin": b0.id, "immediate": True})
                already.add(b0.id)
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
