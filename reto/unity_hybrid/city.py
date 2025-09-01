from dataclasses import dataclass
from typing import List, Tuple, Dict
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
        self.w, self.h = cfg["MAP_SIZE"]
        self.seed = cfg.get("SEED", 42)
        self.rnd = random.Random(self.seed)

        # Road layout: manual or generated grid
        if cfg.get("ROAD_LAYOUT", "manual") == "grid":
            self._build_grid_roads(cfg)
        else:
            self.waypoints: List[Point] = cfg["WAYPOINTS"]
            self.roads: List[Road] = []
            rid = 0
            for (a,b) in cfg["ROADS"]:
                self.roads.append(Road(
                    id=f"r{rid}", a_idx=a, b_idx=b,
                    polyline=[self.waypoints[a], self.waypoints[b]]
                ))
                rid += 1

        self.depot = cfg["DEPOT"]
        self.sidewalk_offset = cfg.get("SIDEWALK_OFFSET_M", 2.0)
        self.bins = self._place_bins(cfg["N_BINS"], cfg["BIN_CAPACITY"])

    def _place_bins(self, n: int, cap: int):
        bins = []
        for i in range(n):
            r = self.rnd.choice(self.roads)
            (x1,y1),(x2,y2) = r.polyline
            t = self.rnd.uniform(0.2,0.8)
            cx = x1 + t*(x2-x1)
            cy = y1 + t*(y2-y1)
            L = math.hypot(x2-x1,y2-y1)
            nx,ny = (-(y2-y1)/L,(x2-x1)/L) if L>1e-6 else (0.0,1.0)
            side = -1 if self.rnd.random()<0.5 else 1
            pos = (cx+side*self.sidewalk_offset*nx, cy+side*self.sidewalk_offset*ny)
            bins.append({"id": f"b{i}", "pos": pos, "curb": (cx, cy), "capacity": cap, "fill": self.rnd.randint(0,cap//2)})
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

    def plan_route(self, start: Point, goal: Point) -> List[Point]:
        """Plan road-constrained route.
        - Snap start/goal to nearest road waypoints
        - Use Dijkstra over the road graph
        - Return polyline along waypoints only; callers should append exact curb/depot if needed
        """
        si = self.nearest_waypoint_idx(start)
        gi = self.nearest_waypoint_idx(goal)
        idx_path = self._dijkstra(si, gi)
        if idx_path is None:
            # fallback to nearest reachable waypoint to goal
            adj,_ = self.road_graph()
            from collections import deque
            q,vis = deque([si]), {si}
            while q:
                u = q.popleft()
                for v,_w in adj[u]:
                    if v not in vis:
                        vis.add(v); q.append(v)
            gx,gy = self.waypoints[gi]
            if not vis:
                return [start]
            gi2 = min(vis, key=lambda i:(self.waypoints[i][0]-gx)**2+(self.waypoints[i][1]-gy)**2)
            idx_path = self._dijkstra(si, gi2) or [si]
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
            if p != dedup[-1]: dedup.append(p)
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
