import agentpy as ap
import json
import math
import random
from typing import List, Tuple, Optional
from heapq import heappush, heappop

# -------------------------- GRID / PATH HELPERS -------------------------- #

# streets_mask[y][x] = 1 if passable, else 0
def make_default_streets_mask(N: int) -> list[list[int]]:
    """Placeholder street grid: a few vertical & horizontal corridors.
       Replace with a real mask (or load from PNG) when you have your map."""
    mask = [[0] * N for _ in range(N)]
    cols = [10, 30, 50, 70, 90, 110, 130]  # tune for your grid_size
    rows = [10, 30, 50, 70, 90, 110, 130]
    for y in range(N):
        for x in cols:
            if 0 <= x < N:
                mask[y][x] = 1
    for x in range(N):
        for y in rows:
            if 0 <= y < N:
                mask[y][x] = 1
    return mask

def neighbors4(x, y, W, H):
    if x > 0:   yield x - 1, y
    if x < W-1: yield x + 1, y
    if y > 0:   yield x, y - 1
    if y < H-1: yield x, y + 1

def astar_grid(start, goal, passable):
    """A* on a 4-neighborhood grid with Manhattan heuristic."""
    W, H = len(passable[0]), len(passable)
    sx, sy = start; gx, gy = goal
    if not (0 <= sx < W and 0 <= sy < H and 0 <= gx < W and 0 <= gy < H):
        return []
    if not passable[sy][sx] or not passable[gy][gx]:
        return []

    def h(a, b):  # Manhattan
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    openh = []
    g = {start: 0.0}
    came = {}
    heappush(openh, (h(start, goal), 0.0, start))
    closed = set()

    while openh:
        _, cost, u = heappop(openh)
        if u in closed:
            continue
        closed.add(u)

        if u == goal:
            path = [u]
            while u in came:
                u = came[u]
                path.append(u)
            path.reverse()
            return path

        ux, uy = u
        for vx, vy in neighbors4(ux, uy, W, H):
            if not passable[vy][vx]:
                continue
            nd = cost + 1.0
            if (vx, vy) not in g or nd < g[(vx, vy)]:
                g[(vx, vy)] = nd
                came[(vx, vy)] = u
                heappush(openh, (nd + h((vx, vy), goal), nd, (vx, vy)))

    return []  # no path

def to_cell(pos):  # world -> grid index
    return (int(round(pos[0])), int(round(pos[1])))

def to_center(cell):  # grid index -> world center
    return [float(cell[0]), float(cell[1])]

# -------------------------- AGENT DEFINITIONS -------------------------- #

class TruckAgent(ap.Agent):
    """Truck agent with capacity, load cycle, and movement states."""

    def setup(self):
        self.pos = [0.0, 0.0]
        self.capacity = self.p.agent_capacity
        self.load = 0
        # States: IDLE, TO_BIN, SERVICING, TO_DEPOT, DUMPING
        self.status = "IDLE"
        self.target_bin: Optional[ap.Agent] = None  # BinAgent or 'DEPOT'
        self.distance_traveled: float = 0.0
        self.services = 0
        self.dumps = 0
        # path buffers (waypoints in grid cells)
        self._path_cells: List[Tuple[int, int]] = []
        self._path_i: int = 0

    def move(self):
        if self.status not in ("TO_BIN", "TO_DEPOT"):
            return

        # If path empty, recompute a last-chance path
        if not self._path_cells or self._path_i >= len(self._path_cells):
            start = to_cell(self.pos)
            if self.target_bin == 'DEPOT':
                goal = self.model.depot_cell
            else:
                bx, by = to_cell(self.target_bin.pos)  # type: ignore
                goal = self.model.adjacent_road_to_cell((bx, by))
            self._path_cells = astar_grid(start, goal, self.model.streets) or [start]
            self._path_i = 0
            if not self._path_cells:
                return

        # current waypoint
        target_cell = self._path_cells[self._path_i]
        target_pos = to_center(target_cell)
        cx, cy = self.pos
        tx, ty = target_pos
        dx, dy = (tx - cx), (ty - cy)
        dist = math.hypot(dx, dy)

        # speeds
        base_speed = getattr(self.model.p, 'truck_speed', 1.0)
        if self.status == "TO_DEPOT":
            base_speed *= getattr(self.model.p, 'return_speed_factor', 1.0)

        if dist <= base_speed:  # arrive to waypoint
            self.pos = [tx, ty]
            self.model.space.move_to(self, (tx, ty))
            self.distance_traveled += dist
            self._path_i += 1
            # If final waypoint reached, trigger arrival
            if self._path_i >= len(self._path_cells):
                if self.status == "TO_BIN":
                    self.status = "SERVICING"
                    self.service_bin()
                elif self.status == "TO_DEPOT":
                    self.status = "DUMPING"
                    self.dump_load()
            return
        else:
            step_x = (dx / dist) * base_speed
            step_y = (dy / dist) * base_speed
            self.pos = [cx + step_x, cy + step_y]
            self.model.space.move_by(self, (step_x, step_y))
            self.distance_traveled += base_speed

    def service_bin(self):
        if self.target_bin in (None, 'DEPOT') or self.status != "SERVICING":
            return
        bin_agent: 'BinAgent' = self.target_bin  # type: ignore
        if bin_agent.remaining <= 0:
            bin_agent.locked_by = None
            self.status = "IDLE"
            self.target_bin = None
            return
        can_take = min(bin_agent.remaining, self.capacity - self.load)
        if can_take > 0:
            bin_agent.remaining -= can_take
            self.load += can_take
            self.services += 1
            self.model.total_waste_collected += can_take
            self.model.log_event(self.model.t, "SERVICE", self.id, bin_agent.id, amount=can_take)
        # Release lock after service attempt
        bin_agent.locked_by = None
        bin_agent.lock_step = None
        if self.load >= self.capacity:
            self.target_bin = 'DEPOT'
            self.status = "TO_DEPOT"
            # plan path to depot
            start = to_cell(self.pos)
            goal = self.model.depot_cell
            self._path_cells = astar_grid(start, goal, self.model.streets) or [start]
            self._path_i = 0
        elif bin_agent.remaining > 0:
            self.status = "IDLE"
            self.target_bin = None
        else:
            self.model.log_event(self.model.t, "BIN_EMPTY", self.id, bin_agent.id)
            self.status = "IDLE"
            self.target_bin = None

    def dump_load(self):
        if self.status != "DUMPING":
            return
        if self.load > 0:
            dumped = self.load
            self.load = 0
            self.dumps += 1
            self.model.log_event(self.model.t, "DUMP", self.id, None, amount=dumped)
        self.status = "IDLE"
        self.target_bin = None


class BinAgent(ap.Agent):
    """Waste bin with quantity and lock."""

    def setup(self):
        self.pos = [0.0, 0.0]
        self.remaining = random.randint(self.p.bin_waste_min, self.p.bin_waste_max)
        self.initial = self.remaining
        self.locked_by: Optional[int] = None
        self.lock_step: Optional[int] = None

# -------------------------- MODEL DEFINITION -------------------------- #

class WasteModel(ap.Model):
    """The main model for the waste collection simulation."""

    # --- small helpers that need access to self ---
    def nearest_road_cell(self, x: int, y: int) -> Tuple[int, int]:
        """Find a nearby road cell by expanding ring search."""
        for r in range(0, 6):
            for dx in range(-r, r + 1):
                for dy in (-r, r):
                    cx, cy = x + dx, y + dy
                    if 0 <= cx < self.N and 0 <= cy < self.N and self.streets[cy][cx]:
                        return (cx, cy)
            for dy in range(-r + 1, r):
                for dx in (-r, r):
                    cx, cy = x + dx, y + dy
                    if 0 <= cx < self.N and 0 <= cy < self.N and self.streets[cy][cx]:
                        return (cx, cy)
        return (x, y)

    def adjacent_road_to_cell(self, cell_xy: Tuple[int, int]) -> Tuple[int, int]:
        """If cell is off-road, return an adjacent road cell; else return itself."""
        x, y = cell_xy
        if 0 <= x < self.N and 0 <= y < self.N and self.streets[y][x]:
            return (x, y)
        best, bestd = None, 1e9
        for nx, ny in neighbors4(x, y, self.N, self.N):
            if 0 <= nx < self.N and 0 <= ny < self.N and self.streets[ny][nx]:
                d = abs(nx - x) + abs(ny - y)
                if d < bestd:
                    bestd, best = d, (nx, ny)
        return best if best else (x, y)

    # --- model lifecycle ---
    def setup(self):
        # Seed control (optional)
        if hasattr(self.p, 'seed') and self.p.seed is not None:
            random.seed(self.p.seed)

        # Environment
        self.N = self.p.grid_size
        self.space = ap.Space(self, shape=[self.N, self.N])
        self.depot_cell = (0, 0)   # change if depot not at (0,0)
        self.depot = (0.0, 0.0)

        # Agents
        self.trucks = ap.AgentList(self, self.p.num_agents, TruckAgent)
        self.bins = ap.AgentList(self, self.p.num_waste_locations, BinAgent)
        self.space.add_agents(self.trucks, random=True)
        self.space.add_agents(self.bins, random=True)
        for a in self.trucks:
            a.pos = self.space.positions[a]
        for b in self.bins:
            b.pos = self.space.positions[b]

        # Streets mask (provide via params or use default)
        if hasattr(self.p, 'streets_mask') and self.p.streets_mask:
            self.streets = self.p.streets_mask  # 2D [N][N] of 0/1
        else:
            self.streets = make_default_streets_mask(self.N)

        # Snap trucks onto roads; bins can stay off-road
        for a in self.trucks:
            cx, cy = self.nearest_road_cell(int(round(a.pos[0])), int(round(a.pos[1])))
            a.pos = [float(cx), float(cy)]

        # Path buffers per truck
        for a in self.trucks:
            a._path_cells = []
            a._path_i = 0

        # Metrics
        self.total_waste_collected = 0
        self.negotiation_messages = 0  # distance evals + broadcasts
        self.distance_cache = {}
        self.previous_idle_ids = set()
        self.previous_free_bin_ids = set()
        self.assignment_needed = True
        self.step_bins_empty = None

        # Log structure
        self.simulation_log = {
            "config": self.p,
            "initial_positions": {
                "agents": [{'x': a.pos[0], 'y': a.pos[1]} for a in self.trucks],
                "waste": [{'x': b.pos[0], 'y': b.pos[1], 'initial': b.initial} for b in self.bins]
            },
            "events": [],
            "agent_paths": {f"agent_{a.id}": [] for a in self.trucks}
        }

    # ---------------- Hungarian Assignment ---------------- #
    def hungarian_assign(self, trucks: List[TruckAgent], bins: List[BinAgent]) -> List[Tuple[TruckAgent, BinAgent]]:
        if not trucks or not bins:
            return []
        n, m = len(trucks), len(bins)
        size = max(n, m)
        LARGE = 1e9
        cost = [[LARGE] * size for _ in range(size)]
        for i, tr in enumerate(trucks):
            for j, bn in enumerate(bins):
                # Distance caching (Euclidean over world coords; OK for matching)
                key = (tr.id, bn.id)
                tpos = (tr.pos[0], tr.pos[1])
                bpos = (bn.pos[0], bn.pos[1])
                cached = self.distance_cache.get(key)
                if cached and cached[1] == tpos and cached[2] == bpos:
                    d = cached[0]
                else:
                    d = math.dist(tr.pos, bn.pos)
                    self.distance_cache[key] = (d, tpos, bpos)
                    self.negotiation_messages += 1  # distance evaluation
                cost[i][j] = d
        # Hungarian algorithm
        u = [0.0] * (size + 1)
        v = [0.0] * (size + 1)
        p = [0] * (size + 1)
        way = [0] * (size + 1)
        for i in range(1, size + 1):
            p[0] = i
            j0 = 0
            minv = [float('inf')] * (size + 1)
            used = [False] * (size + 1)
            while True:
                used[j0] = True
                i0 = p[j0]
                delta = float('inf')
                j1 = 0
                for j in range(1, size + 1):
                    if not used[j]:
                        cur = cost[i0 - 1][j - 1] - u[i0] - v[j]
                        if cur < minv[j]:
                            minv[j] = cur
                            way[j] = j0
                        if minv[j] < delta:
                            delta = minv[j]
                            j1 = j
                for j in range(size + 1):
                    if used[j]:
                        u[p[j]] += delta
                        v[j] -= delta
                    else:
                        minv[j] -= delta
                j0 = j1
                if p[j0] == 0:
                    break
            while True:
                j1 = way[j0]
                p[j0] = p[j1]
                j0 = j1
                if j0 == 0:
                    break
        assignment = []
        for j in range(1, size + 1):
            if p[j] and p[j] <= n and j <= m and cost[p[j] - 1][j - 1] < LARGE / 2:
                assignment.append((trucks[p[j] - 1], bins[j - 1]))
        return assignment

    def step(self):
        # 1) Determine current idle & free sets
        idle_trucks = [t for t in self.trucks if t.status == "IDLE" and t.load < t.capacity]
        free_bins = [b for b in self.bins if b.remaining > 0 and b.locked_by is None]
        idle_ids = {t.id for t in idle_trucks}
        free_bin_ids = {b.id for b in free_bins}
        changed = (idle_ids != self.previous_idle_ids) or (free_bin_ids != self.previous_free_bin_ids) or self.assignment_needed

        # 2) Assign (and plan paths) when things change
        if changed and idle_trucks and free_bins:
            if len(idle_trucks) == 1 or len(free_bins) == 1:
                tr = idle_trucks[0]
                best_bn = min(free_bins, key=lambda b: math.dist(tr.pos, b.pos))
                assignments = [(tr, best_bn)]
            else:
                assignments = self.hungarian_assign(idle_trucks, free_bins)

            for tr, bn in assignments:
                if tr.status != "IDLE" or tr.load >= tr.capacity or bn.remaining <= 0 or bn.locked_by is not None:
                    continue
                tr.target_bin = bn
                tr.status = "TO_BIN"
                bn.locked_by = tr.id
                bn.lock_step = self.t
                self.log_event(self.t, "ASSIGN", tr.id, bn.id)

                # PLAN PATH NOW (to road cell next to bin)
                start = to_cell(tr.pos)
                goal = self.adjacent_road_to_cell(to_cell(bn.pos))
                path = astar_grid(start, goal, self.streets)
                tr._path_cells = path if path else [start]
                tr._path_i = 0

            if assignments:
                receivers = max(len(idle_trucks) - 1, 0)
                self.negotiation_messages += len(assignments) * receivers
                self.assignment_needed = False

        self.previous_idle_ids = idle_ids
        self.previous_free_bin_ids = free_bin_ids

        # 3) Movement along planned paths
        movers_list = [t for t in self.trucks if t.status in ("TO_BIN", "TO_DEPOT")]
        if movers_list:
            ap.AgentList(self, movers_list).move()

        # 4) Service/Dump actions (service and dump may also be triggered in move())
        for tr in self.trucks:
            if tr.status == "SERVICING":
                tr.service_bin()
                self.assignment_needed = True
            elif tr.status == "DUMPING":
                tr.dump_load()
                self.assignment_needed = True

        # 5) Log positions
        for tr in self.trucks:
            key = f"agent_{tr.id}"
            self.simulation_log["agent_paths"][key].append({'x': tr.pos[0], 'y': tr.pos[1]})

        # 6) Completion management
        bins_empty = all(b.remaining <= 0 for b in self.bins)
        if bins_empty and self.step_bins_empty is None:
            self.step_bins_empty = self.t
        if bins_empty:
            # Send any loaded trucks to depot, with planned path
            for tr in self.trucks:
                if tr.load > 0 and tr.status not in ("TO_DEPOT", "DUMPING"):
                    tr.target_bin = 'DEPOT'
                    tr.status = "TO_DEPOT"
                    start = to_cell(tr.pos)
                    goal = self.depot_cell
                    tr._path_cells = astar_grid(start, goal, self.streets) or [start]
                    tr._path_i = 0
            # Stop once all trucks have dumped
            if all(t.load == 0 for t in self.trucks):
                self.simulation_log["events"].append({
                    'step': self.t,
                    'type': 'ALL_COLLECTED',
                    'agent_id': None,
                    'waste_id': None
                })
                self.stop()

    def end(self):
        # Summary metrics (detailed internal log)
        detailed_metrics = {
            'total_initial_waste': sum(b.initial for b in self.bins),
            'total_waste_collected': self.total_waste_collected,
            'total_remaining_waste': sum(b.remaining for b in self.bins),
            'remaining_bins': sum(1 for b in self.bins if b.remaining > 0),
            'negotiation_messages': self.negotiation_messages,
            'agent_travel_distances': {f'agent_{a.id}': a.distance_traveled for a in self.trucks},
            'agent_services': {f'agent_{a.id}': a.services for a in self.trucks},
            'agent_dumps': {f'agent_{a.id}': a.dumps for a in self.trucks},
            'final_loads': {f'agent_{a.id}': a.load for a in self.trucks},
            'step_bins_empty': self.step_bins_empty,
            'step_all_dumped': self.t
        }
        self.simulation_log["summary_metrics"] = detailed_metrics
        self.simulation_log['final_bins'] = [{
            'id': b.id,
            'initial': b.initial,
            'remaining': b.remaining
        } for b in self.bins]
        # pathObj parity for internal log
        self.simulation_log['agent_paths_obj'] = {
            k: [{'x': p['x'], 'y': p['y']} for p in v]
            for k, v in self.simulation_log['agent_paths'].items()
        }

        # ---- Build Unity SimData format ---- #
        collected_map = {a.id: 0 for a in self.trucks}
        for ev in self.simulation_log['events']:
            if ev.get('type') == 'SERVICE' and ev.get('agent_id') is not None:
                collected_map[ev['agent_id']] += ev.get('amount', 0)

        agents_json = []
        for a in self.trucks:
            key = f"agent_{a.id}"
            raw_path = self.simulation_log['agent_paths'].get(key, [])
            int_path = [{'x': int(round(p['x'])), 'y': int(round(p['y']))} for p in raw_path]
            start = int_path[0] if int_path else {'x': int(round(a.pos[0])), 'y': int(round(a.pos[1]))}
            agents_json.append({
                'id': a.id,
                'start': [start['x'], start['y']],
                'pathObj': int_path,
                'distance': float(detailed_metrics['agent_travel_distances'][f'agent_{a.id}']),
                'collected': collected_map.get(a.id, 0),
                'capacity': a.capacity
            })

        bins_json = [{
            'id': b.id,
            'pos': [int(round(b.pos[0])), int(round(b.pos[1]))],
            'initial': b.initial,
            'remaining': b.remaining
        } for b in self.bins]

        events_json = [{
            't': ev.get('step', 0),
            'type': ev.get('type'),
            'agent': ev.get('agent_id') if ev.get('agent_id') is not None else -1,
            'bin': ev.get('waste_id') if ev.get('waste_id') is not None else -1,
            'amount': ev.get('amount', 0)
        } for ev in self.simulation_log['events']]

        distances = list(detailed_metrics['agent_travel_distances'].values())
        avg_distance = sum(distances) / len(distances) if distances else 0.0

        simdata = {
            'grid': {'width': self.p.grid_size, 'height': self.p.grid_size, 'depot': [self.depot_cell[0], self.depot_cell[1]]},
            'agents': agents_json,
            'bins': bins_json,
            'events': events_json,
            'metrics': {
                'total_collected': int(detailed_metrics['total_waste_collected']),
                'avg_distance_per_agent': avg_distance,
                'negotiation_messages': int(detailed_metrics['negotiation_messages']),
                'steps': int(detailed_metrics['step_all_dumped'])
            }
        }
        unity_name = getattr(self.p, 'unity_output_name', 'sim_run_pathObj.json')
        with open(unity_name, 'w') as f:
            json.dump(simdata, f, indent=2)
        with open('full_log.json', 'w') as f:
            json.dump(self.simulation_log, f, indent=2)
        print(f"\nUnity SimData saved to {unity_name}; full detailed log in full_log.json")

    def log_event(self, step, event_type, agent_id, waste_id, amount=None):
        event = {"step": step, "type": event_type, "agent_id": agent_id, "waste_id": waste_id}
        if amount is not None:
            event['amount'] = amount
        self.simulation_log['events'].append(event)
        print(f"Step {step}: {event_type} - Agent {agent_id} Bin {waste_id} {('amt='+str(amount)) if amount is not None else ''}")

# -------------------------- SIMULATION EXECUTION -------------------------- #

if __name__ == "__main__":
    random.seed()
    num_agents = random.randint(3, 7)
    num_waste = random.randint(10, 30)
    parameters = {
        'num_agents': num_agents,
        'num_waste_locations': num_waste,
        'grid_size': 150,
        'agent_capacity': 20,
        'bin_waste_min': 5,
        'bin_waste_max': 15,
        'steps': 100000,  # upper bound safeguard
        # 'streets_mask': your_custom_mask_here  # optional: inject your own roads
    }
    print(f"Randomized setup -> Trucks: {num_agents}, Waste bins: {num_waste}")
    model = WasteModel(parameters)
    model.run()
    print("\n--- Simulation Finished ---")
    print(model.simulation_log.get('summary_metrics', {}))

    # --- Export Unity offline JSON (SimData shape) --- #
    def export_unity_simdata(model: WasteModel, outfile: str = 'sim_run_pathObj.json'):
        log = model.simulation_log
        summary = log.get('summary_metrics', {})
        collected_map = {a.id: 0 for a in model.trucks}
        for ev in log['events']:
            if ev.get('type') == 'SERVICE' and ev.get('agent_id') is not None:
                collected_map[ev['agent_id']] += ev.get('amount', 0)
        agents_json = []
        for a in model.trucks:
            path_key = f"agent_{a.id}"
            raw_path = log['agent_paths'].get(path_key, [])
            int_path = [{'x': int(round(p['x'])), 'y': int(round(p['y']))} for p in raw_path]
            start = int_path[0] if int_path else {'x': int(round(a.pos[0])), 'y': int(round(a.pos[1]))}
            agents_json.append({
                'id': a.id,
                'start': [start['x'], start['y']],
                'pathObj': int_path,
                'distance': float(summary.get('agent_travel_distances', {}).get(path_key, 0.0)),
                'collected': collected_map.get(a.id, 0),
                'capacity': a.capacity
            })
        bins_json = [{
            'id': b.id,
            'pos': [int(round(b.pos[0])), int(round(b.pos[1]))],
            'initial': b.initial,
            'remaining': b.remaining
        } for b in model.bins]
        events_json = [{
            't': ev.get('step', 0),
            'type': ev.get('type'),
            'agent': ev.get('agent_id') if ev.get('agent_id') is not None else -1,
            'bin': ev.get('waste_id') if ev.get('waste_id') is not None else -1,
            'amount': ev.get('amount', 0)
        } for ev in log['events']]
        distances = list(summary.get('agent_travel_distances', {}).values())
        avg_distance = sum(distances) / len(distances) if distances else 0.0
        simdata = {
            'grid': {'width': model.p.grid_size, 'height': model.p.grid_size, 'depot': [model.depot_cell[0], model.depot_cell[1]]},
            'agents': agents_json,
            'bins': bins_json,
            'events': events_json,
            'metrics': {
                'total_collected': int(summary.get('total_waste_collected', 0)),
                'avg_distance_per_agent': avg_distance,
                'negotiation_messages': int(summary.get('negotiation_messages', 0)),
                'steps': int(summary.get('step_all_dumped', 0))
            }
        }
        with open(outfile, 'w') as f:
            json.dump(simdata, f, indent=2)
        print(f"Unity offline JSON exported -> {outfile} (agents={len(agents_json)}, bins={len(bins_json)})")

    export_unity_simdata(model)
