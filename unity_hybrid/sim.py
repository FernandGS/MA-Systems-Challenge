from typing import List, Dict, Set
import random, json, math
from agents import Truck, BinObj
from dispatch import auction, market
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
        # Discrete cell occupancy history to enforce headway/cooldown
        self._cell_last_entered: Dict[tuple, int] = {}

        # Bins
        self.bins: List[BinObj] = [
            BinObj(b["id"], b["pos"], b["capacity"], b["fill"], b.get("curb")) for b in city.bins
        ]

        # Trucks (with optional staggered activation)
        self.trucks: List[Truck] = []
        stagger = float(cfg.get("TRUCK_STAGGER_SEC", 0.0))
        for i in range(cfg["N_TRUCKS"]):
            angle = (i % max(1, cfg["N_TRUCKS"])) * (2.0 * math.pi / max(1, cfg["N_TRUCKS"]))
            spawn_r = float(cfg.get("SPAWN_RING_RADIUS_M", 1.2))
            sx = city.depot[0] + spawn_r * math.cos(angle)
            sy = city.depot[1] + spawn_r * math.sin(angle)
            t = Truck(tid=f"T{i}", pos=(sx, sy), cfg=cfg, energy=cfg["ENERGY_MAX"]) 
            t.spawn_time = i * stagger  # type: ignore[attr-defined]
            t.inactive = True if stagger > 0 and i > 0 else False  # type: ignore[attr-defined]
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
        active_trucks = [t for t in self.trucks if not getattr(t, 'inactive', False)]
        if self.cfg.get("POLICY", "dqn") == "dqn" and self.rl is not None:
            self.rl.start_step(self.trucks)
            assigns = self.rl.select_and_assign(self.city, self.bins, active_trucks, self.t, self._plan_route)
        else:
            pol = self.cfg.get("POLICY", "auction")
            if pol == "market":
                assigns = market(self.bins, active_trucks, self.t, self.cfg, self._plan_route)
            else:
                assigns = auction(self.bins, active_trucks, self.t, self.cfg, self._plan_route)
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

        # 3. Strict anti-overlap via cell reservation + cooldown and swap prevention
        # Reset movement blocks
        for t in self.trucks:
            t.block_move = False
            # Activate if its spawn time has arrived
            if getattr(t, 'inactive', False) and self.t >= getattr(t, 'spawn_time', 0.0):
                t.inactive = False  # type: ignore[attr-defined]
                self.events.append({"t": self.t, "type": "spawn", "truck": t.tid})
        # Compute intended next positions (greedy lookahead using current target)
        intended: Dict[str, tuple] = {}
        for t in self.trucks:
            if getattr(t, 'inactive', False):
                intended[t.tid] = t.pos
                continue
            # Hold position if truck is in pickup dwell pause
            if getattr(t, 'pickup_dwell_steps', 0) > 0:
                intended[t.tid] = t.pos
                continue
            # Determine immediate target for this tick without mutating state
            if t.route_pts and t.route_i < len(t.route_pts):
                tgt = t.route_pts[t.route_i]
            elif t.target is not None:
                # plan a very short route preview
                route = self._plan_route(t.pos, t.target)
                tgt = route[0] if route else t.target
            else:
                intended[t.tid] = t.pos
                continue
            # Predict movement length
            dx, dy = tgt[0] - t.pos[0], tgt[1] - t.pos[1]
            d = math.hypot(dx, dy)
            if d < 1e-6:
                intended[t.tid] = t.pos
            else:
                step = min(d, self.cfg["TRUCK_SPEED_MPS"] * dt)
                intended[t.tid] = (t.pos[0] + dx/d * step, t.pos[1] + dy/d * step)

        # Track trucks held by intersection control this tick (used to force block_move)
        intersection_held_ids: Set[str] = set()

        # Intersection arbitration (virtual traffic lights) BEFORE cell reservation
        if self.cfg.get("INTERSECTION_CONTROL", False) and len(intended) > 1:
            # Initialize tracking structure for wait streaks if absent
            if not hasattr(self, '_intersection_wait_streak'):
                self._intersection_wait_streak = {}
            if not hasattr(self, '_intersection_wait_prev'):
                self._intersection_wait_prev = set()
            axis_eps = float(self.cfg.get("INTERSECTION_AXIS_EPS", 0.05))
            mode = str(self.cfg.get("INTERSECTION_MODE", "alternate")).lower()
            # Toggle state for alternate mode
            if not hasattr(self, "_intersection_phase"):
                self._intersection_phase = True  # type: ignore
            else:
                if mode == "alternate":
                    self._intersection_phase = not getattr(self, "_intersection_phase")  # type: ignore
            # Collect move descriptors
            moves = []  # (tid, from_pos, to_pos, horizontal, vertical)
            for t in self.trucks:
                a = t.tid
                p0 = t.pos
                p1 = intended.get(a, p0)
                dx = p1[0]-p0[0]; dy = p1[1]-p0[1]
                horiz = abs(dx) >= abs(dy)
                moving = (abs(dx) > axis_eps) or (abs(dy) > axis_eps)
                if moving:
                    moves.append((a, p0, p1, horiz, not horiz))

            # Precompute nearest intersection (waypoint) each move is heading toward (projection heuristic)
            approach_window = float(self.cfg.get("INTERSECTION_APPROACH_WINDOW_M", 0.0))
            headway_m = float(self.cfg.get("INTERSECTION_APPROACH_HEADWAY_M", 0.0))
            wp_list = getattr(self.city, 'waypoints', []) or []
            move_targets = {}
            if approach_window > 0 and wp_list:
                for (tid, f0, t0, hflag, vflag) in moves:
                    # Ray from f0 to t0; find waypoint whose perpendicular distance to segment is small and lies ahead
                    vx, vy = t0[0]-f0[0], t0[1]-f0[1]
                    vlen2 = vx*vx + vy*vy + 1e-9
                    best_wp = None; best_d = 1e9
                    for wx, wy in wp_list:
                        # vector from start to waypoint
                        rx, ry = wx - f0[0], wy - f0[1]
                        tproj = (rx*vx + ry*vy)/vlen2
                        if tproj < 0 or tproj > 1.5:  # only look slightly ahead of current intended move
                            continue
                        # perpendicular distance
                        px = f0[0] + vx*tproj
                        py = f0[1] + vy*tproj
                        dist = math.hypot(wx-px, wy-py)
                        ahead_dist = math.hypot(wx-f0[0], wy-f0[1])
                        if dist <= approach_window and ahead_dist < best_d:
                            best_d = ahead_dist
                            best_wp = (wx, wy)
                    if best_wp:
                        move_targets[tid] = (best_wp, best_d)
            # Simple O(n^2) conflict detection: if segments approach the same intersection area
            held: Dict[str, bool] = {}
            angle_thresh = math.radians(float(self.cfg.get("INTERSECTION_ANGLE_DEG", 80.0)))
            cos_thresh = math.cos(angle_thresh)
            lookahead_steps = int(self.cfg.get("INTERSECTION_LOOKAHEAD_STEPS", 1))
            proximity_m = float(self.cfg.get("INTERSECTION_PROXIMITY_M", 0.0))
            speed_cfg = float(self.cfg.get("TRUCK_SPEED_MPS", 1.0))
            dt_cfg = float(self.cfg.get("DT", 1.0))
            extend_dist = lookahead_steps * speed_cfg * dt_cfg

            def _extend(p0, p1):
                dx, dy = p1[0]-p0[0], p1[1]-p0[1]
                d = math.hypot(dx, dy)
                if d < 1e-9:
                    return (p0, p1)
                f = (d + extend_dist)/d
                return (p0, (p0[0] + dx*f, p0[1] + dy*f))

            def _orient(a,b,c):
                return (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])

            def _segments_intersect(a1,a2,b1,b2):
                o1 = _orient(a1,a2,b1)
                o2 = _orient(a1,a2,b2)
                o3 = _orient(b1,b2,a1)
                o4 = _orient(b1,b2,a2)
                if (o1==0 and o2==0 and o3==0 and o4==0):
                    def _bbox(p,q,r,s):
                        return (max(min(p[0],q[0]), min(r[0],s[0])) <= min(max(p[0],q[0]), max(r[0],s[0])) + 1e-6 and
                                max(min(p[1],q[1]), min(r[1],s[1])) <= min(max(p[1],q[1]), max(r[1],s[1])) + 1e-6)
                    return _bbox(a1,a2,b1,b2)
                return (o1*o2 <= 0) and (o3*o4 <= 0)

            def _closest_dist(a1,a2,b1,b2):
                if _segments_intersect(a1,a2,b1,b2):
                    return 0.0
                def _proj(p, s1, s2):
                    vx, vy = s2[0]-s1[0], s2[1]-s1[1]
                    l2 = vx*vx+vy*vy
                    if l2<1e-12: return s1
                    t = max(0.0, min(1.0, ((p[0]-s1[0])*vx + (p[1]-s1[1])*vy)/l2))
                    return (s1[0]+vx*t, s1[1]+vy*t)
                ptsA = [a1,a2]
                ptsB = [b1,b2]
                dmin = 1e9
                for p in ptsA:
                    q = _proj(p,b1,b2); dmin = min(dmin, math.hypot(p[0]-q[0], p[1]-q[1]))
                for p in ptsB:
                    q = _proj(p,a1,a2); dmin = min(dmin, math.hypot(p[0]-q[0], p[1]-q[1]))
                return dmin

            for i in range(len(moves)):
                for j in range(i+1, len(moves)):
                    a1, f1, t1, h1, v1 = moves[i]
                    a2, f2, t2, h2, v2 = moves[j]
                    verbose = self.cfg.get("INTERSECTION_DEBUG_VERBOSE", False)
                    v1x, v1y = t1[0]-f1[0], t1[1]-f1[1]
                    v2x, v2y = t2[0]-f2[0], t2[1]-f2[1]
                    denom = (math.hypot(v1x, v1y) * math.hypot(v2x, v2y))
                    treat_conflict = False
                    # Skip early if they are strongly collinear (same direction) to avoid tailgating being treated as intersection
                    if denom>1e-6:
                        cosdir = (v1x*v2x + v1y*v2y)/denom
                        if cosdir > 0.985:
                            # Only keep as potential if lateral offset very small AND rear truck within small gap of front entering an actual waypoint zone.
                            # Compute lateral distance of a2 start to a1 direction.
                            vx, vy = v1x, v1y
                            norm = math.hypot(vx, vy)+1e-9
                            ux, uy = vx/norm, vy/norm
                            # vector from f1 to f2
                            rx, ry = f2[0]-f1[0], f2[1]-f1[1]
                            # lateral component magnitude
                            lat = abs(rx*-uy + ry*ux)
                            along = rx*ux + ry*uy
                            if lat < 0.6 and 0 < along < (approach_window if approach_window>0 else 4.0):
                                # treat as following; skip intersection arbitration
                                if self.cfg.get("INTERSECTION_DEBUG_VERBOSE", False):
                                    self.events.append({"t": self.t, "type": "intersection_dbg", "a": a1, "b": a2, "why": "skip_follow", "lat": round(lat,2), "along": round(along,2)})
                                continue
                            # else allow evaluation (they might be converging slight offset)
                    if h1 != h2:
                        treat_conflict = True
                    elif denom > 1e-6:
                        cosang = (v1x*v2x + v1y*v2y) / denom
                        if abs(cosang) < cos_thresh:
                            treat_conflict = True
                            if abs(v1x) >= abs(v1y):
                                h1 = True; h2 = False
                            else:
                                h1 = False; h2 = True
                    angle_used = 'axis' if h1!=h2 else ('angle' if treat_conflict else 'none')
                    # Extend and test predictive intersection
                    ef1, et1 = _extend(f1, t1)
                    ef2, et2 = _extend(f2, t2)
                    seg_cross = _segments_intersect(ef1, et1, ef2, et2)
                    prox_cross = False
                    if not seg_cross and proximity_m > 0.0:
                        # Only treat proximity if both trucks are within an expanded approach window of some shared waypoint (reduces false side-by-side suppression)
                        cd = _closest_dist(ef1, et1, ef2, et2)
                        if cd <= proximity_m:
                            # find nearest waypoint to midpoint of closest endpoints
                            mx = ( (f1[0]+t1[0]+f2[0]+t2[0]) * 0.25 )
                            my = ( (f1[1]+t1[1]+f2[1]+t2[1]) * 0.25 )
                            near_wp = False
                            if wp_list:
                                for wx, wy in wp_list:
                                    if math.hypot(wx-mx, wy-my) <= (approach_window*1.5 if approach_window>0 else 3.0):
                                        near_wp = True; break
                            if near_wp:
                                prox_cross = True
                    # Distance between trucks (current) used to suppress far false positives
                    cur_sep = math.hypot(f1[0]-f2[0], f1[1]-f2[1])
                    # Suppress if they are far apart and neither segment intersects nor near
                    far_suppress = False
                    if not (seg_cross or prox_cross) and cur_sep > (approach_window*1.25 if approach_window>0 else 8.0):
                        far_suppress = True
                    # Early approach via waypoint projection
                    if not treat_conflict and not seg_cross and not prox_cross:
                        if a1 in move_targets and a2 in move_targets:
                            wp1, d1 = move_targets[a1]; wp2, d2 = move_targets[a2]
                            if wp1 == wp2 and (d1 <= approach_window or d2 <= approach_window):
                                if d1 <= d2:
                                    loser = a2
                                else:
                                    loser = a1
                                held[loser] = True
                                if headway_m > 0:
                                    speed_l = max(1e-6, self.cfg.get('TRUCK_SPEED_MPS', 1.0))
                                    extra = int(math.ceil(headway_m / (speed_l * self.cfg.get('DT',1.0))))
                                    if extra > 0:
                                        if not hasattr(self, '_intersection_holds'):
                                            self._intersection_holds = {}
                                        self._intersection_holds[loser] = (self.t + extra, extra)
                                continue
                        if not (seg_cross or prox_cross):
                            continue
                    if far_suppress:
                        if verbose:
                            self.events.append({"t": self.t, "type": "intersection_dbg", "a": a1, "b": a2, "why": "far_suppress", "sep": round(cur_sep,2)})
                        continue
                    if not (treat_conflict or seg_cross or prox_cross):
                        continue
                    if verbose:
                        self.events.append({
                            "t": self.t,
                            "type": "intersection_dbg",
                            "a": a1, "b": a2,
                            "reason": angle_used,
                            "seg": seg_cross, "prox": prox_cross,
                            "cur_sep": round(cur_sep,2),
                            "f1": (round(f1[0],2), round(f1[1],2)),
                            "t1": (round(t1[0],2), round(t1[1],2)),
                            "f2": (round(f2[0],2), round(f2[1],2)),
                            "t2": (round(t2[0],2), round(t2[1],2)),
                        })
                    # Fallback to axis-based winner selection if not already decided
                    winner_h = None
                    turn_priority = self.cfg.get("INTERSECTION_TURN_PRIORITY", False)
                    def is_turn(aid, from_p, to_p):
                        for tt in self.trucks:
                            if tt.tid == aid and tt.route_pts:
                                if len(tt.route_pts) >= 2:
                                    vv1x = to_p[0] - from_p[0]; vv1y = to_p[1] - from_p[1]
                                    vv2x = tt.route_pts[1][0] - tt.route_pts[0][0]; vv2y = tt.route_pts[1][1] - tt.route_pts[0][1]
                                    den2 = (math.hypot(vv1x,vv1y)*math.hypot(vv2x,vv2y)+1e-6)
                                    if den2 > 0:
                                        dp2 = (vv1x*vv2x+vv1y*vv2y)/den2
                                        return abs(dp2) < 0.2
                        return False
                    a1_turn = is_turn(a1, f1, t1)
                    a2_turn = is_turn(a2, f2, t2)
                    multi_hold = int(self.cfg.get("INTERSECTION_MULTI_HOLD", 0))
                    if turn_priority and (a1_turn != a2_turn):
                        winner_h = h1 if (not a1_turn and a2_turn) else h2
                    if winner_h is None:
                        if mode == "horiz_priority":
                            winner_h = True
                        elif mode == "vert_priority":
                            winner_h = False
                        elif mode == "low_id":
                            def _idnum(tid):
                                num = ''.join(ch for ch in tid if ch.isdigit())
                                return int(num) if num else 0
                            n1, n2 = _idnum(a1), _idnum(a2)
                            winner_h = h1 if (n1 <= n2) else h2
                        else:
                            winner_h = bool(getattr(self, "_intersection_phase"))
                    a1_is_horizontal = h1
                    if not hasattr(self, "_intersection_holds"):
                        self._intersection_holds = {}
                    holds = self._intersection_holds
                    def remaining_hold(tid):
                        rec = holds.get(tid)
                        if rec is None or rec[0] < self.t:
                            return 0
                        return rec[1]
                    r1 = remaining_hold(a1)
                    r2 = remaining_hold(a2)
                    if r1>0 and r2==0:
                        held[a1]=True; continue
                    if r2>0 and r1==0:
                        held[a2]=True; continue
                    if r1>0 and r2>0:
                        if self.cfg.get("INTERSECTION_EXCLUSIVE_HOLD", True):
                            # Choose a single winner to proceed: reuse winner_h axis or fallback to low id
                            # Compute winner horizontally as before
                            # Release the one whose hold expires sooner or lower id if equal
                            # Determine remaining durations using holds record
                            def _remain(tid):
                                rec = holds.get(tid)
                                if rec is None or rec[0] < self.t:
                                    return 0
                                return rec[0]-self.t
                            rem1 = _remain(a1)
                            rem2 = _remain(a2)
                            proceed = a1
                            if rem2 < rem1:
                                proceed = a2
                            elif rem2 == rem1 and a2 < a1:
                                proceed = a2
                            # Keep loser held only
                            other = a2 if proceed == a1 else a1
                            held[proceed] = False
                            held[other] = True
                            continue
                        else:
                            held[a1]=True; held[a2]=True; continue
                    loser = a2 if a1_is_horizontal == winner_h else a1
                    held[loser] = True
                    if multi_hold>1:
                        holds[loser] = (self.t + multi_hold - 1, multi_hold - 1)
                    else:
                        # ensure any previous losing hold is cleared quickly by setting expiry to current time
                        if loser in holds:
                            holds[loser] = (self.t, 0)
                    winner_h = None
                    turn_priority = self.cfg.get("INTERSECTION_TURN_PRIORITY", False)
                    def is_turn(aid, from_p, to_p):
                        for tt in self.trucks:
                            if tt.tid == aid and tt.route_pts:
                                if len(tt.route_pts) >= 2:
                                    v1x = to_p[0] - from_p[0]; v1y = to_p[1] - from_p[1]
                                    v2x = tt.route_pts[1][0] - tt.route_pts[0][0]; v2y = tt.route_pts[1][1] - tt.route_pts[0][1]
                                    denom = (math.hypot(v1x,v1y)*math.hypot(v2x,v2y)+1e-6)
                                    if denom > 0:
                                        dp = (v1x*v2x+v1y*v2y)/denom
                                        return abs(dp) < 0.2
                        return False
                    a1_turn = is_turn(a1, f1, t1)
                    a2_turn = is_turn(a2, f2, t2)
                    multi_hold = int(self.cfg.get("INTERSECTION_MULTI_HOLD", 0))
                    if turn_priority and (a1_turn != a2_turn):
                        winner_h = h1 if (not a1_turn and a2_turn) else h2
                    if winner_h is None:
                        if mode == "horiz_priority":
                            winner_h = True
                        elif mode == "vert_priority":
                            winner_h = False
                        elif mode == "low_id":
                            def _idnum(tid):
                                num = ''.join(ch for ch in tid if ch.isdigit())
                                return int(num) if num else 0
                            n1, n2 = _idnum(a1), _idnum(a2)
                            winner_h = h1 if (n1 <= n2) else h2
                        else:
                            winner_h = bool(getattr(self, "_intersection_phase"))
                    a1_is_horizontal = h1
                    if not hasattr(self, "_intersection_holds"):
                        self._intersection_holds = {}
                    holds = self._intersection_holds
                    def remaining_hold(tid):
                        rec = holds.get(tid)
                        if rec is None or rec[0] < self.t:
                            return 0
                        return rec[1]
                    r1 = remaining_hold(a1)
                    r2 = remaining_hold(a2)
                    if r1>0 and r2==0:
                        held[a1]=True; continue
                    if r2>0 and r1==0:
                        held[a2]=True; continue
                    if r1>0 and r2>0:
                        if self.cfg.get("INTERSECTION_EXCLUSIVE_HOLD", True):
                            def _remain(tid):
                                rec = holds.get(tid)
                                if rec is None or rec[0] < self.t:
                                    return 0
                                return rec[0]-self.t
                            rem1 = _remain(a1); rem2 = _remain(a2)
                            proceed = a1
                            if rem2 < rem1:
                                proceed = a2
                            elif rem2 == rem1 and a2 < a1:
                                proceed = a2
                            other = a2 if proceed == a1 else a1
                            held[proceed] = False
                            held[other] = True
                            continue
                        else:
                            held[a1]=True; held[a2]=True; continue
                    loser = a2 if a1_is_horizontal == winner_h else a1
                    held[loser] = True
                    if multi_hold>1:
                        holds[loser] = (self.t + multi_hold - 1, multi_hold - 1)
            if held:
                # Enforce max consecutive wait limit per truck BEFORE applying
                max_wait = int(self.cfg.get("INTERSECTION_MAX_WAIT_TICKS", 0))
                if max_wait > 0:
                    to_release = []
                    for tid,hval in held.items():
                        if hval:
                            streak = self._intersection_wait_streak.get(tid,0)
                            if streak >= max_wait:  # already waited max_wait ticks consecutively
                                to_release.append(tid)
                    for tid in to_release:
                        held[tid] = False  # allow movement
                        # also clear any explicit multi-hold record
                        if hasattr(self, '_intersection_holds') and tid in self._intersection_holds:
                            self._intersection_holds.pop(tid, None)
                for t in self.trucks:
                    if held.get(t.tid):
                        intended[t.tid] = t.pos
                        intersection_held_ids.add(t.tid)
                        if self.cfg.get("INTERSECTION_LOG", False):
                            self.events.append({"t": self.t, "type": "intersection_wait", "truck": t.tid})
            # After determining this tick's holds update streak counters (postponed until after applying)
            new_prev = set()
            for t in self.trucks:
                if t.tid in intersection_held_ids:
                    prev = self._intersection_wait_streak.get(t.tid,0)
                    self._intersection_wait_streak[t.tid] = prev + 1
                    new_prev.add(t.tid)
                else:
                    # reset on successful move (or no hold)
                    if self._intersection_wait_streak.get(t.tid,0) != 0:
                        self._intersection_wait_streak[t.tid] = 0
            self._intersection_wait_prev = new_prev

    # Cell quantization helpers
        cell_size = float(self.cfg.get("TRUCK_CELL_SIZE_M", self.cfg.get("MIN_TRUCK_SPACING_M", 1.5)))
        follow_gap = int(self.cfg.get("MIN_FOLLOW_GAP_STEPS", 1))
        def cell_of(p: tuple) -> tuple:
            return (int(math.floor(p[0] / cell_size)), int(math.floor(p[1] / cell_size)))

        # Build current and next cells
        current_cell: Dict[str, tuple] = {}
        next_cell: Dict[str, tuple] = {}
        for t in self.trucks:
            current_cell[t.tid] = cell_of(t.pos)
            nxt = intended.get(t.tid, t.pos)
            next_cell[t.tid] = cell_of(nxt)

        # Build reverse index: which trucks are in each current cell
        cell_to_tids: Dict[tuple, List[str]] = {}
        for tid, c in current_cell.items():
            cell_to_tids.setdefault(c, []).append(tid)

        # Reservation process (deterministic by truck id)
        step_idx = int(self.t // dt)
        reserved: Dict[tuple, str] = {}  # cell -> tid
        # Precompute swap intentions
        wants: Dict[str, tuple] = {tid: next_cell[tid] for tid in next_cell}
        stuck: Dict[str, bool] = {t.tid: False for t in self.trucks}
        # Consider current occupancy for cooldown even if vacated this tick
        occupied_now = {cell: tids[:] for cell, tids in cell_to_tids.items()}
        # If multiple trucks currently share a cell, only allow the lowest-id to attempt moving first
        for cell, tids in occupied_now.items():
            if len(tids) > 1:
                for tid in sorted(tids)[1:]:
                    stuck[tid] = True
                    reserved.setdefault(cell, tid)
        for t in sorted(self.trucks, key=lambda x: x.tid):
            if getattr(t, 'inactive', False):
                continue
            tid = t.tid
            cur_c = current_cell[tid]
            nxt_c = next_cell[tid]
            # If not changing cell, allow
            if nxt_c == cur_c:
                continue
            # Cooldown: avoid entering a cell too soon after someone just occupied it
            last_ent = self._cell_last_entered.get(nxt_c, -10**9)
            if step_idx - last_ent <= follow_gap:
                stuck[tid] = True
                reserved.setdefault(cur_c, tid)  # keep own cell
                continue
            # If the target cell is currently occupied by another and that other isn't moving out, block
            occ_tids = occupied_now.get(nxt_c, [])
            if any(wants.get(otid) == current_cell[otid] for otid in occ_tids):
                stuck[tid] = True
                reserved.setdefault(cur_c, tid)
                continue
            # Prevent two trucks reserving the same cell
            if nxt_c in reserved and reserved[nxt_c] != tid:
                stuck[tid] = True
                reserved.setdefault(cur_c, tid)
                continue
            # Swap prevention: A->Bcell and B->Acell in same tick
            swap_with = None
            for other in self.trucks:
                if other.tid == tid:
                    continue
                if current_cell[other.tid] == nxt_c and next_cell[other.tid] == cur_c:
                    swap_with = other.tid
                    break
            if swap_with is not None:
                # allow lower-id to move, block higher-id
                if tid > swap_with:
                    stuck[tid] = True
                    reserved.setdefault(cur_c, tid)
                    continue
                else:
                    # Reserve own next cell and mark the other as stuck to avoid both moving
                    reserved[nxt_c] = tid
                    stuck[swap_with] = True
                    continue
            # Otherwise, reserve next cell
            reserved[nxt_c] = tid

    # Apply movement blocks based on reservations
        for t in self.trucks:
            if getattr(t, 'inactive', False):
                continue
            tid = t.tid
            cur_c = current_cell[tid]
            nxt_c = next_cell[tid]
            # Near-depot free zone: relax blocks within DEPOT_FREE_RADIUS_M to let trucks dock
            free_r = float(self.cfg.get("DEPOT_FREE_RADIUS_M", 0.0))
            near_depot = (free_r > 0.0) and (math.hypot(t.pos[0]-self.city.depot[0], t.pos[1]-self.city.depot[1]) <= free_r)
            # Block if marked stuck or if someone else reserved our next cell (unless near depot)
            if stuck.get(tid, False) and not near_depot:
                t.block_move = True
            elif nxt_c != cur_c and reserved.get(nxt_c) != tid and not near_depot:
                t.block_move = True
            else:
                t.block_move = False

        # Enforce intersection holds AFTER reservation logic so they cannot be overridden
        if intersection_held_ids:
            for t in self.trucks:
                if t.tid in intersection_held_ids:
                    t.block_move = True

    # Direction-aware following: block if moving into a spot that is too close and ahead of another truck along same direction
        min_space = float(self.cfg.get("MIN_TRUCK_SPACING_M", 1.5))
        def too_close_ahead(a_cur, a_nxt, b_cur, b_nxt) -> bool:
            # Vector of A and B
            ax, ay = a_nxt[0]-a_cur[0], a_nxt[1]-a_cur[1]
            bx, by = b_nxt[0]-b_cur[0], b_nxt[1]-b_cur[1]
            # If both not moving much, skip
            if abs(ax)+abs(ay) < 1e-6 and abs(bx)+abs(by) < 1e-6:
                return False
            # Similar direction if dot > 0
            same_dir = (ax*bx + ay*by) > 0
            if not same_dir:
                return False
            # Predict positions (continuous) and measure distance
            dx, dy = a_nxt[0]-b_nxt[0], a_nxt[1]-b_nxt[1]
            return math.hypot(dx, dy) < min_space
        # Block higher-id follower if violating spacing along same direction
        for i in range(len(self.trucks)):
            for j in range(i+1, len(self.trucks)):
                ti, tj = self.trucks[i], self.trucks[j]
                if getattr(ti, 'inactive', False) or getattr(tj, 'inactive', False):
                    continue
                ai_cur = intended.get(ti.tid, ti.pos)
                aj_cur = intended.get(tj.tid, tj.pos)
                ai_nxt = intended.get(ti.tid, ti.pos)
                aj_nxt = intended.get(tj.tid, tj.pos)
                if too_close_ahead(ai_cur, ai_nxt, aj_cur, aj_nxt):
                    # Block higher-id
                    if ti.tid > tj.tid:
                        ti.block_move = True
                    else:
                        tj.block_move = True

        # Final pairwise guard: block moves that would bring trucks closer than min_space
        live_next: Dict[str, tuple] = {}
        for t in self.trucks:
            pos_nxt = intended.get(t.tid, t.pos)
            # if movement blocked, next is current
            if getattr(t, 'block_move', False):
                pos_nxt = t.pos
            live_next[t.tid] = pos_nxt
        for i in range(len(self.trucks)):
            for j in range(i+1, len(self.trucks)):
                ti, tj = self.trucks[i], self.trucks[j]
                if getattr(ti, 'inactive', False) or getattr(tj, 'inactive', False):
                    continue
                pi, pj = live_next[ti.tid], live_next[tj.tid]
                # keep guard unless both are within depot free radius
                near_dep_i = math.hypot(ti.pos[0]-self.city.depot[0], ti.pos[1]-self.city.depot[1]) <= float(self.cfg.get("DEPOT_FREE_RADIUS_M", 0.0))
                near_dep_j = math.hypot(tj.pos[0]-self.city.depot[0], tj.pos[1]-self.city.depot[1]) <= float(self.cfg.get("DEPOT_FREE_RADIUS_M", 0.0))
                if math.hypot(pi[0]-pj[0], pi[1]-pj[1]) < min_space and not (near_dep_i and near_dep_j):
                    # block higher-id this tick
                    if ti.tid > tj.tid:
                        ti.block_move = True
                        live_next[ti.tid] = ti.pos
                    else:
                        tj.block_move = True
                        live_next[tj.tid] = tj.pos

        # After determining who can move, update cooldown for cells that will be entered
        for t in self.trucks:
            tid = t.tid
            cur_c = current_cell[tid]
            nxt_c = next_cell[tid]
            if (not t.block_move) and (nxt_c != cur_c):
                self._cell_last_entered[nxt_c] = step_idx

        # 3b. Trucks step (with potential movement blocks)
        step_events = []
        for t in self.trucks:
            if getattr(t, 'inactive', False):
                continue
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
