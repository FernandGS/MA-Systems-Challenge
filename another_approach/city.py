# city.py
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

        self.waypoints: List[Point] = cfg["WAYPOINTS"]
        self.roads: List[Road] = []
        rid = 0
        for (a,b) in cfg["ROADS"]:
            self.roads.append(Road(
                id=f"r{rid}",
                a_idx=a,
                b_idx=b,
                polyline=[self.waypoints[a], self.waypoints[b]]
            ))
            rid += 1

        self.depot: Point = cfg["DEPOT"]
        self.sidewalk_offset = cfg.get("SIDEWALK_OFFSET_M", 2.0)
        self.bins = self._place_bins(cfg["N_BINS"], cfg["BIN_CAPACITY"])

    def _polyline_length(self, pl: List[Point]) -> float:
        return sum(math.hypot(pl[i+1][0]-pl[i][0], pl[i+1][1]-pl[i][1]) for i in range(len(pl)-1))

    def _place_bins(self, n: int, cap: int):
        bins = []
        for i in range(n):
            r = self.rnd.choice(self.roads)
            (x1,y1),(x2,y2) = r.polyline
            t = self.rnd.uniform(0.2,0.8)
            cx = x1 + t*(x2-x1)
            cy = y1 + t*(y2-y1)
            L = math.hypot(x2-x1,y2-y1)
            nx,ny = -(y2-y1)/L,(x2-x1)/L
            side = -1 if self.rnd.random()<0.5 else 1
            pos = (cx+side*self.sidewalk_offset*nx, cy+side*self.sidewalk_offset*ny)
            bins.append({"id": f"b{i}", "pos": pos, "capacity": cap, "fill": self.rnd.randint(0,cap//2)})
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
        """Return index of the nearest waypoint to a given position."""
        px, py = p
        best_i, best_d = 0, float("inf")
        for i, (x, y) in enumerate(self.waypoints):
            d = math.hypot(px - x, py - y)
            if d < best_d:
                best_d, best_i = d, i
        return best_i

    def _dijkstra(self, start_idx: int, goal_idx: int):
        """Dijkstra on waypoint graph; returns list of waypoint indices."""
        adj, _coords = self.road_graph()
        import heapq
        dist = {i: float("inf") for i in adj}
        prev = {i: None for i in adj}
        dist[start_idx] = 0.0
        pq = [(0.0, start_idx)]
        while pq:
            d, u = heapq.heappop(pq)
            if u == goal_idx:
                break
            if d > dist[u]:
                continue
            for v, w in adj[u]:
                nd = d + w
                if nd < dist[v]:
                    dist[v] = nd
                    prev[v] = u
                    heapq.heappush(pq, (nd, v))
        # reconstruct
        path = []
        u = goal_idx
        if prev[u] is None and u != start_idx:
            return [start_idx, goal_idx]  # disconnected fallback
        while u is not None:
            path.append(u)
            u = prev[u]
        path.reverse()
        return path

    def _closest_point_on_segment(self, p, a, b):
        ax, ay = a; bx, by = b; px, py = p
        vx, vy = bx-ax, by-ay
        wx, wy = px-ax, py-ay
        vv = vx*vx + vy*vy
        if vv <= 1e-9:
            return a, 0.0
        t = max(0.0, min(1.0, (wx*vx + wy*vy)/vv))
        cx, cy = ax + t*vx, ay + t*vy
        return (cx, cy), t

    def snap_to_road(self, p: Point) -> Point:
        """Return the closest point to p on any road polyline (curb point)."""
        best_pt, best_d2 = None, float("inf")
        for r in self.roads:
            a = self.waypoints[r.a_idx]; b = self.waypoints[r.b_idx]
            c, _t = self._closest_point_on_segment(p, a, b)
            dx, dy = p[0]-c[0], p[1]-c[1]
            d2 = dx*dx + dy*dy
            if d2 < best_d2:
                best_d2 = d2
                best_pt = c
        return best_pt

    def plan_route(self, start: Point, goal: Point) -> list[Point]:
        """
        Route along roads by snapping start/goal to nearest road points,
        then add a tiny final hop to the off-road goal (e.g., sidewalk bin).
        """
        s_snap = self.snap_to_road(start)
        g_snap = self.snap_to_road(goal)

        si = self.nearest_waypoint_idx(s_snap)
        gi = self.nearest_waypoint_idx(g_snap)
        idx_path = self._dijkstra(si, gi)

        route = []
        route.append(start if start == s_snap else s_snap)
        route += [self.waypoints[i] for i in idx_path]
        route.append(g_snap)

        # final short hop to actual goal (sidewalk offset is small)
        if (goal[0]-g_snap[0])**2 + (goal[1]-g_snap[1])**2 > 1e-6:
            route.append(goal)

        # de-dup consecutive identical points
        dedup = [route[0]]
        for p in route[1:]:
            if p != dedup[-1]:
                dedup.append(p)
        return dedup
