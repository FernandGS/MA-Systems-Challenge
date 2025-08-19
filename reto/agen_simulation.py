import agentpy as ap
import json
import math
import random
from typing import List, Tuple, Optional

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

    def move(self):
        if not self.target_bin or self.status not in ("TO_BIN", "TO_DEPOT"):
            return
        if self.target_bin == 'DEPOT':
            target_pos = self.model.depot
        else:
            target_pos = self.target_bin.pos
        current_pos = self.pos
        dir_x = target_pos[0] - current_pos[0]
        dir_y = target_pos[1] - current_pos[1]
        dist = math.sqrt(dir_x**2 + dir_y**2)
        if dist < 1.0:
            # Arrived
            if self.status == "TO_BIN":
                self.status = "SERVICING"
                self.service_bin()
            elif self.status == "TO_DEPOT":
                self.status = "DUMPING"
                self.dump_load()
            return
        # Speed selection
        base_speed = getattr(self.model.p, 'truck_speed', 1.0)
        if self.status == "TO_DEPOT":
            base_speed *= getattr(self.model.p, 'return_speed_factor', 1.0)
        step_dist = min(base_speed, dist)  # don't overshoot
        move_x = (dir_x / dist) * step_dist
        move_y = (dir_y / dist) * step_dist
        new_pos = [current_pos[0] + move_x, current_pos[1] + move_y]
        self.distance_traveled += step_dist
        self.model.space.move_by(self, (move_x, move_y))
        self.pos = new_pos

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
        elif bin_agent.remaining > 0:
            # Bin still has waste; can be reassigned later
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

    def setup(self):
        # --- Environment ---
        # Seed control (optional)
        if hasattr(self.p, 'seed') and self.p.seed is not None:
            random.seed(self.p.seed)
        self.space = ap.Space(self, shape=[self.p.grid_size, self.p.grid_size])
        self.depot = (0.0, 0.0)

        # --- Agents ---
        self.trucks = ap.AgentList(self, self.p.num_agents, TruckAgent)
        self.bins = ap.AgentList(self, self.p.num_waste_locations, BinAgent)
        self.space.add_agents(self.trucks, random=True)
        self.space.add_agents(self.bins, random=True)
        for agent in self.trucks:
            agent.pos = self.space.positions[agent]
        for agent in self.bins:
            agent.pos = self.space.positions[agent]

        # Metrics
        self.total_waste_collected = 0
        self.negotiation_messages = 0  # distance evals + broadcasts
        # Caches & assignment control
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
        cost = [[LARGE]*size for _ in range(size)]
        for i, tr in enumerate(trucks):
            for j, bn in enumerate(bins):
                # Distance caching
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
        u = [0.0]*(size+1); v = [0.0]*(size+1)
        p = [0]*(size+1); way = [0]*(size+1)
        for i in range(1, size+1):
            p[0] = i; j0 = 0
            minv = [float('inf')]*(size+1); used = [False]*(size+1)
            while True:
                used[j0] = True
                i0 = p[j0]; delta = float('inf'); j1 = 0
                for j in range(1, size+1):
                    if not used[j]:
                        cur = cost[i0-1][j-1] - u[i0] - v[j]
                        if cur < minv[j]: minv[j] = cur; way[j] = j0
                        if minv[j] < delta: delta = minv[j]; j1 = j
                for j in range(size+1):
                    if used[j]: u[p[j]] += delta; v[j] -= delta
                    else: minv[j] -= delta
                j0 = j1
                if p[j0] == 0: break
            while True:
                j1 = way[j0]; p[j0] = p[j1]; j0 = j1
                if j0 == 0: break
        assignment = []
        for j in range(1, size+1):
            if p[j] and p[j] <= n and j <= m and cost[p[j]-1][j-1] < LARGE/2:
                assignment.append((trucks[p[j]-1], bins[j-1]))
        return assignment

    def step(self):
        # 1. Determine current idle & free sets
        idle_trucks = [t for t in self.trucks if t.status == "IDLE" and t.load < t.capacity]
        free_bins = [b for b in self.bins if b.remaining > 0 and b.locked_by is None]
        idle_ids = {t.id for t in idle_trucks}
        free_bin_ids = {b.id for b in free_bins}
        changed = (idle_ids != self.previous_idle_ids) or (free_bin_ids != self.previous_free_bin_ids) or self.assignment_needed

        if changed and idle_trucks and free_bins:
            assignments = []
            # Shortcut: if only 1 idle OR 1 free bin, pick nearest without Hungarian
            if len(idle_trucks) == 1 or len(free_bins) == 1:
                tr = idle_trucks[0]
                # Find nearest bin
                best_bn = min(free_bins, key=lambda b: math.dist(tr.pos, b.pos))
                assignments.append((tr, best_bn))
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
            if assignments:
                receivers = max(len(idle_trucks) - 1, 0)
                self.negotiation_messages += len(assignments) * receivers
                self.assignment_needed = False
        self.previous_idle_ids = idle_ids
        self.previous_free_bin_ids = free_bin_ids

        # 2. Movement
        movers_list = [t for t in self.trucks if t.status in ("TO_BIN", "TO_DEPOT")]
        if movers_list:
            ap.AgentList(self, movers_list).move()

        # 3. Service/Dump actions (service and dump happen immediately upon arrival in move())
        for tr in self.trucks:
            if tr.status == "SERVICING":
                tr.service_bin()
                self.assignment_needed = True  # may free or partially empty bin
            elif tr.status == "DUMPING":
                tr.dump_load()
                self.assignment_needed = True
            # If truck became empty and waste still exists, it will be reassigned next step
        
        # 4. Log positions
        for tr in self.trucks:
            key = f"agent_{tr.id}"
            self.simulation_log["agent_paths"][key].append({'x': tr.pos[0], 'y': tr.pos[1]})

        # 5. Completion management:
        bins_empty = all(b.remaining <= 0 for b in self.bins)
        if bins_empty and self.step_bins_empty is None:
            self.step_bins_empty = self.t
        if bins_empty:
            # Direct any loaded trucks not already heading to depot to go dump
            for tr in self.trucks:
                if tr.load > 0 and tr.status not in ("TO_DEPOT", "DUMPING"):
                    tr.target_bin = 'DEPOT'
                    tr.status = "TO_DEPOT"
            # Once all trucks have zero load (after dumping), stop
            if all(t.load == 0 for t in self.trucks):
                self.simulation_log["events"].append({
                    'step': self.t,
                    'type': 'ALL_COLLECTED',
                    'agent_id': None,
                    'waste_id': None
                })
                self.stop()

    def end(self):
        # Summary metrics
        self.simulation_log["summary_metrics"] = {
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
        self.simulation_log['final_bins'] = [{
            'id': b.id,
            'initial': b.initial,
            'remaining': b.remaining
        } for b in self.bins]
        # pathObj parity
        self.simulation_log['agent_paths_obj'] = {k: [{'x': p['x'], 'y': p['y']} for p in v]
                                                  for k, v in self.simulation_log['agent_paths'].items()}
        with open('simulation_log.json', 'w') as f:
            json.dump(self.simulation_log, f, indent=4)
        print("\nSimulation log saved to simulation_log.json")

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
        'grid_size': 20,
        'agent_capacity': 20,
        'bin_waste_min': 5,
        'bin_waste_max': 15,
        'steps': 100000  # upper bound safeguard
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
        # Build collected map
        collected_map = {a.id: 0 for a in model.trucks}
        for ev in log['events']:
            if ev.get('type') == 'SERVICE' and ev.get('agent_id') is not None:
                collected_map[ev['agent_id']] += ev.get('amount', 0)
        # Paths already in agent_paths list of dicts -> transform to ints
        agents_json = []
        for a in model.trucks:
            path_key = f"agent_{a.id}"
            raw_path = log['agent_paths'].get(path_key, [])
            # raw_path = [{'x': float, 'y': float}, ...]
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
        bins_json = []
        for b in model.bins:
            bins_json.append({
                'id': b.id,
                'pos': [int(round(b.pos[0])), int(round(b.pos[1]))],
                'initial': b.initial,
                'remaining': b.remaining
            })
        events_json = []
        for ev in log['events']:
            events_json.append({
                't': ev.get('step', 0),
                'type': ev.get('type'),
                'agent': ev.get('agent_id') if ev.get('agent_id') is not None else -1,
                'bin': ev.get('waste_id') if ev.get('waste_id') is not None else -1,
                'amount': ev.get('amount', 0)
            })
        distances = list(summary.get('agent_travel_distances', {}).values())
        avg_distance = sum(distances)/len(distances) if distances else 0.0
        simdata = {
            'grid': {'width': model.p.grid_size, 'height': model.p.grid_size, 'depot': [0, 0]},
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