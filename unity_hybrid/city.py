from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Set
import math, random

Point = Tuple[float, float]

@dataclass
class Road:
    id: str
    a_idx: int
    b_idx: int
    polyline: List[Point]

class City:
    def __init__(self, cfg: Dict):
        self.cfg = cfg
        self.w, self.h = cfg["MAP_SIZE"]
        self.seed = cfg.get("SEED", 42)
        self.rnd = random.Random(self.seed)

        # Road layout: manual or generated grid
        if cfg.get("ROAD_LAYOUT", "manual") == "grid":
            self._build_grid_roads(cfg)
        else:
            self.waypoints = cfg["WAYPOINTS"]
            self.roads = []
            rid = 0
            for (a, b) in cfg["ROADS"]:
                self.roads.append(Road(
                    id=f"r{rid}", a_idx=a, b_idx=b,
                    polyline=[self.waypoints[a], self.waypoints[b]]
                ))
                rid += 1

        self.depot = cfg["DEPOT"]
        self.sidewalk_offset = cfg.get("SIDEWALK_OFFSET_M", 2.0)
        self.bins = self._place_bins(cfg["N_BINS"], cfg["BIN_CAPACITY"])
        self._compute_deadends()

    def _place_bins(self, n: int, cap: int):
        bins = []
        placed: List[Point] = []
        # Support legacy key BIN_MIN_SPACING if MIN_BIN_SEP_M not present
        if "MIN_BIN_SEP_M" in self.cfg:
            min_sep = float(self.cfg.get("MIN_BIN_SEP_M", 6.0))
        else:
            min_sep = float(self.cfg.get("BIN_MIN_SPACING", 6.0))
        road_half = float(self.cfg.get("ROAD_HALF_WIDTH", 3.5))
        extra_margin = float(self.cfg.get("SIDEWALK_MARGIN_M", 0.75))
        bin_clearance = float(self.cfg.get("BIN_CLEARANCE_M", 0.0))
        # curb_offset ensures bins are beyond roadway + margin + optional clearance
        curb_offset = max(self.sidewalk_offset, road_half + extra_margin) + bin_clearance
        # Try to distribute attempts across roads to avoid clustering
        road_indices = list(range(len(self.roads)))
        attempt = 0
        ridx = 0
        while len(bins) < n and attempt < n * 40:
            attempt += 1
            if not road_indices:
                road_indices = list(range(len(self.roads)))
            r = self.roads[road_indices[ridx % len(road_indices)]]
            ridx += 1
            (x1,y1),(x2,y2) = r.polyline
            L = math.hypot(x2-x1,y2-y1)
            if L < 1e-6:
                continue
            # avoid endpoints/intersections: keep t away from 0 and 1
            t = self.rnd.uniform(0.15, 0.85)
            cx = x1 + t*(x2-x1)
            cy = y1 + t*(y2-y1)
            # normal vector
            nx,ny = (-(y2-y1)/L,(x2-x1)/L)
            side = -1 if self.rnd.random()<0.5 else 1
            pos = (cx+side*curb_offset*nx, cy+side*curb_offset*ny)
            # spacing check vs already placed bin visual positions
            ok = True
            for (px,py) in placed:
                if math.hypot(px-pos[0], py-pos[1]) < min_sep:
                    ok = False; break
            if not ok:
                continue
            bobj = {"id": f"b{len(bins)}", "pos": pos, "curb": (cx, cy), "capacity": cap, "fill": self.rnd.randint(0,cap//2)}
            bins.append(bobj)
            placed.append(pos)
        # If failed to place all, fill remaining without spacing (as last resort)
        while len(bins) < n:
            r = self.rnd.choice(self.roads)
            (x1,y1),(x2,y2) = r.polyline
            L = math.hypot(x2-x1,y2-y1)
            t = self.rnd.uniform(0.2,0.8)
            cx = x1 + t*(x2-x1)
            cy = y1 + t*(y2-y1)
            nx,ny = (-(y2-y1)/L,(x2-x1)/L) if L>1e-6 else (0.0,1.0)
            side = -1 if self.rnd.random()<0.5 else 1
            pos = (cx+side*curb_offset*nx, cy+side*curb_offset*ny)
            bins.append({"id": f"b{len(bins)}", "pos": pos, "curb": (cx, cy), "capacity": cap, "fill": self.rnd.randint(0,cap//2)})
        return bins

    def road_graph(self):
        coords = {i:self.waypoints[i] for i in range(len(self.waypoints))}
        adj = {i:[] for i in coords}
        for r in self.roads:
            (x1,y1),(x2,y2) = r.polyline
            d = math.hypot(x2-x1,y2-y1)
            adj[r.a_idx].append((r.b_idx,d))
            adj[r.b_idx].append((r.a_idx,d))
        return adj,coords

    def nearest_waypoint_idx(self, p: Point) -> int:
        px, py = p
        best_i, best_d = 0, float("inf")
        for i,(x,y) in enumerate(self.waypoints):
            d = math.hypot(px-x, py-y)
            if d < best_d:
                best_d, best_i = d, i
        return best_i

    def _dijkstra(self, start_idx: int, goal_idx: int):
        adj,_coords = self.road_graph()
        import heapq
        dist = {i: float("inf") for i in adj}
        prev = {i: None for i in adj}
        dist[start_idx] = 0.0
        pq = [(0.0, start_idx)]
        seen = set()
        while pq:
            d,u = heapq.heappop(pq)
            if u in seen: continue
            seen.add(u)
            if u == goal_idx: break
            if d > dist[u]: continue
            for v,w in adj[u]:
                nd = d+w
                if nd < dist[v]:
                    dist[v] = nd; prev[v] = u
                    heapq.heappush(pq,(nd,v))
        if prev[goal_idx] is None and goal_idx != start_idx:
            return None
        path = []
        u = goal_idx
        while u is not None:
            path.append(u); u = prev[u]
        path.reverse()
        return path

    def _dijkstra_no_immediate_backtrack(self, start_idx: int, goal_idx: int, prev_idx: Optional[int], uturn_penalty: Optional[float] = None):
        """Dijkstra over expanded state (node, prev) that discourages immediate backtracking (U-turns).
        Adds a penalty when transitioning back to the previous node. Allows it if it's the only way.
        """
        adj, _coords = self.road_graph()
        import heapq
        # State is (node, prev_node) where prev_node can be None for the start
        start_state = (start_idx, prev_idx)
        if uturn_penalty is None:
            uturn_penalty = float(self.cfg.get("UTURN_PENALTY", 200.0))
        forbid_when_alt = bool(self.cfg.get("FORBID_UTURN_IF_ALTERNATIVE", True))
        dist: Dict[Tuple[int, Optional[int]], float] = {start_state: 0.0}
        prv: Dict[Tuple[int, Optional[int]], Optional[Tuple[int, Optional[int]]]] = {start_state: None}
        pq: List[Tuple[float, Tuple[int, Optional[int]]]] = [(0.0, start_state)]
        best_goal_state: Optional[Tuple[int, Optional[int]]] = None
        best_goal_cost = float('inf')
        while pq:
            d, (u, pu) = heapq.heappop(pq)
            if d > dist[(u, pu)]:
                continue
            if u == goal_idx and d < best_goal_cost:
                best_goal_cost = d
                best_goal_state = (u, pu)
                # don't break; a later path may be even cheaper due to penalties
            # Determine if there is any neighbor other than pu (previous). If so and forbid_when_alt, block backtrack.
            has_alternative = any((v2 != pu) for (v2, _w2) in adj[u])
            for v, w in adj[u]:
                cost = w
                if pu is not None and v == pu:
                    if forbid_when_alt and has_alternative:
                        continue  # hard forbid when there is any alternative
                    cost += float(uturn_penalty)
                nd = d + cost
                sv = (v, u)
                if nd < dist.get(sv, float('inf')):
                    dist[sv] = nd
                    prv[sv] = (u, pu)
                    heapq.heappush(pq, (nd, sv))
        if best_goal_state is None:
            return None
        # Reconstruct state path back to start
        path_nodes: List[int] = []
        cur: Optional[Tuple[int, Optional[int]]] = best_goal_state
        while cur is not None:
            u, _pu = cur
            path_nodes.append(u)
            cur = prv.get(cur)
        path_nodes.reverse()
        return path_nodes

    def plan_route(self, start: Point, goal: Point, prev_idx: Optional[int] = None) -> List[Point]:
        si = self.nearest_waypoint_idx(start)
        gi = self.nearest_waypoint_idx(goal)
        # Honor config that allows U-turn only at deadends
        allow_deadend_only = bool(self.cfg.get("ALLOW_UTURN_DEADEND_ONLY", False))
        if allow_deadend_only and prev_idx is not None and prev_idx == gi and not self.is_deadend_waypoint(gi):
            # If goal is simply going back to previous non-deadend node, adjust search by forbidding it via large penalty
            # Force algorithm to explore alternatives first
            idx_path = self._dijkstra_no_immediate_backtrack(si, gi, prev_idx)
        else:
            idx_path = self._dijkstra_no_immediate_backtrack(si, gi, prev_idx)
        if idx_path is None:
            # fallback to nearest reachable waypoint to goal
            adj, _ = self.road_graph()
            from collections import deque
            q, vis = deque([si]), {si}
            while q:
                u = q.popleft()
                for v, _w in adj[u]:
                    if v not in vis:
                        vis.add(v)
                        q.append(v)
            gx, gy = self.waypoints[gi]
            if not vis:
                return [start]
            gi2 = min(vis, key=lambda i: (self.waypoints[i][0]-gx)**2 + (self.waypoints[i][1]-gy)**2)
            idx_path = self._dijkstra_no_immediate_backtrack(si, gi2, prev_idx) or [si]
        route = [self.waypoints[i] for i in idx_path]
        # prefix with current start if far from first road node for continuity
        if route:
            if math.hypot(start[0]-route[0][0], start[1]-route[0][1]) > 1.0:
                route = [route[0]] + route  # keep first road node twice to ease snapping
        else:
            route = [start]
        # de-dup consecutive
        dedup = [route[0]]
        for p in route[1:]:
            if p != dedup[-1]:
                dedup.append(p)
        return dedup

    # -----------------
    # Grid layout helper
    # -----------------
    def _build_grid_roads(self, cfg: Dict):
        rows = int(cfg.get("GRID_ROWS", 6))
        cols = int(cfg.get("GRID_COLS", 6))
        spacing = float(cfg.get("GRID_SPACING", 20.0))
        margin = float(cfg.get("GRID_MARGIN", 10.0))
        # Build waypoint lattice across the map bounds
        xs = []
        ys = []
        if spacing > 0:
            x = margin
            while x <= self.w - margin + 1e-6:
                xs.append(x); x += spacing
            y = margin
            while y <= self.h - margin + 1e-6:
                ys.append(y); y += spacing
        else:
            # derive from rows/cols
            if cols < 2: cols = 2
            if rows < 2: rows = 2
            xs = [margin + i * (self.w - 2*margin) / (cols-1) for i in range(cols)]
            ys = [margin + j * (self.h - 2*margin) / (rows-1) for j in range(rows)]
        self.waypoints = []
        idx = {}
        for j,y in enumerate(ys):
            for i,x in enumerate(xs):
                idx[(i,j)] = len(self.waypoints)
                self.waypoints.append((x,y))
        self.roads = []
        rid = 0
        # Connect orthogonally
        for j in range(len(ys)):
            for i in range(len(xs)):
                a = idx[(i,j)]
                if i+1 < len(xs):
                    b = idx[(i+1,j)]
                    self.roads.append(Road(id=f"r{rid}", a_idx=a, b_idx=b, polyline=[self.waypoints[a], self.waypoints[b]])); rid += 1
                if j+1 < len(ys):
                    b = idx[(i,j+1)]
                    self.roads.append(Road(id=f"r{rid}", a_idx=a, b_idx=b, polyline=[self.waypoints[a], self.waypoints[b]])); rid += 1
        # Place depot roughly central on a node
        ci, cj = len(xs)//2, len(ys)//2
        self.depot = self.waypoints[idx[(ci, cj)]]

    def _compute_deadends(self):
        adj,_ = self.road_graph()
        self.deadend_nodes = {n for n, nbrs in adj.items() if len(nbrs) <= 1}

    def is_deadend_waypoint(self, idx: int) -> bool:
        return hasattr(self, 'deadend_nodes') and idx in self.deadend_nodes
