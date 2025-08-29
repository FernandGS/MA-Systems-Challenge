import agentpy as ap
import random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.lines as mlines
from collections import deque
from math import inf

# ==============================================================
# Helper: BFS next step along streets
# ==============================================================

def next_step_bfs(start, goal, street_coords):
    """Return the next cell on a shortest street-only path from start to goal.
       If start==goal or no path, return start."""
    if start == goal:
        return start
    q = deque([(start, [])])
    seen = {start}
    moves = [(1,0), (-1,0), (0,1), (0,-1)]
    while q:
        cur, path = q.popleft()
        for dx, dy in moves:
            nb = (cur[0]+dx, cur[1]+dy)
            if nb == goal:
                return path[0] if path else nb
            if nb in street_coords and nb not in seen:
                seen.add(nb)
                q.append((nb, path+[nb]))
    return start


# ==============================================================
# Agents
# ==============================================================

class Bin(ap.Agent):
    def setup(self):
        self.capacity_max = self.p.capacity_max
        self.fill_step = self.p.fill_step
        self.fill = random.randint(0, self.capacity_max // 2)

    def step(self):
        if self.fill < self.capacity_max:
            self.fill = min(self.capacity_max, self.fill + self.fill_step)


class Truck(ap.Agent):
    def setup(self):
        self.capacity_max = self.p.truck_capacity
        self.carrying = 0
        self.state = "SEEK_BIN"   # SEEK_BIN -> PICKUP -> TO_DEPOT -> DROP
        self.target_bin = None

    # ----- Utilities -----
    def pos(self):
        return self.model.space.positions[self]

    def manhattan(self, a, b):
        return abs(a[0]-b[0]) + abs(a[1]-b[1])

    def nearest_nonempty_bin(self):
        """Pick the fullest, tie-broken by distance."""
        best = None
        best_key = None  # (-fill, distance)
        mypos = self.pos()
        for b in self.model.bins:
            if b.fill > 0:
                bpos = self.model.space.positions[b]
                key = (-b.fill, self.manhattan(mypos, bpos))
                if best is None or key < best_key:
                    best = b
                    best_key = key
        return best

    # ----- Behavior -----
    def step(self):
        x, y = self.pos()
        depot = self.model.deposit_pos

        # Decide state transitions
        if self.state == "SEEK_BIN":
            if self.carrying >= self.capacity_max:
                self.state = "TO_DEPOT"
            else:
                if self.target_bin is None or self.target_bin.fill == 0:
                    self.target_bin = self.nearest_nonempty_bin()
                if self.target_bin is None:
                    # No work → head to depot idly
                    self.state = "TO_DEPOT"

        if self.state == "PICKUP":
            # pick a unit (simple MVP: carry 1, empty bin)
            if self.target_bin and self.model.space.positions[self.target_bin] == (x, y) and self.target_bin.fill > 0:
                self.carrying = min(self.capacity_max, self.carrying + 1)
                self.target_bin.fill = 0
                self.state = "TO_DEPOT"
            else:
                # not actually on the bin (race condition) → re-seek
                self.state = "SEEK_BIN"

        if self.state == "DROP":
            if (x, y) == depot:
                self.carrying = 0
                self.state = "SEEK_BIN"
            else:
                self.state = "TO_DEPOT"

        # Move or act
        if self.state == "SEEK_BIN":
            target = None
            if self.carrying < self.capacity_max:
                self.target_bin = self.nearest_nonempty_bin()
                if self.target_bin:
                    target = self.model.space.positions[self.target_bin]
            if target is None:
                # fallback: drift toward depot
                target = depot

            if (x, y) == target:
                if self.target_bin and target == self.model.space.positions[self.target_bin]:
                    self.state = "PICKUP"
                else:
                    # already at depot with nothing to do -> idle
                    pass
            else:
                nxt = next_step_bfs((x, y), target, self.model.street_coords)
                self.model.space.move_to(self, nxt)

        elif self.state == "TO_DEPOT":
            if (x, y) == depot:
                self.state = "DROP"
            else:
                nxt = next_step_bfs((x, y), depot, self.model.street_coords)
                self.model.space.move_to(self, nxt)

        elif self.state == "PICKUP":
            # handled above (no movement)
            pass

        elif self.state == "DROP":
            # handled above (no movement)
            pass


# ==============================================================
# Model
# ==============================================================

def generate_positions(n, allowed_positions, min_dist):
    positions, tries = [], 0
    allowed = list(allowed_positions)
    while len(positions) < n and tries < 5000:
        cand = random.choice(allowed)
        if all(abs(cand[0]-px)+abs(cand[1]-py) >= min_dist for px, py in positions):
            positions.append(cand)
        tries += 1
    if len(positions) < n:
        raise ValueError("Could not place agents.")
    return positions


class WasteModel(ap.Model):
    def setup(self):
        random.seed(self.p.get('seed', 42))
        self.space = ap.Grid(self, [20, 20], track_empty=True)

        # Street lattice
        self.street_coords = set()
        for i in range(20):
            self.street_coords |= {(i, 4), (i, 6), (i, 8), (i, 12), (i, 16)}
            self.street_coords |= {(4, i), (8, i), (12, i), (16, i)}

        self.deposit_pos = (1, 4)

        self.bins = ap.AgentList(self, 10, Bin)
        self.trucks = ap.AgentList(self, 3, Truck)

        allowed_bin_positions = self.street_coords - {self.deposit_pos}
        bin_positions = generate_positions(len(self.bins), allowed_bin_positions, 5)
        self.space.add_agents(self.bins, positions=bin_positions)

        truck_positions = generate_positions(len(self.trucks), self.street_coords, 10)
        self.space.add_agents(self.trucks, positions=truck_positions)

        self.t = 0
        self.history = []

    def step(self):
        self.trucks.step()
        self.bins.step()

        self.history.append({
            "step": int(self.t),
            "trucks": [
                {"id": i,
                 "x": int(self.space.positions[t][0]),
                 "y": int(self.space.positions[t][1]),
                 "carrying": int(t.carrying),
                 "state": t.state}
                for i, t in enumerate(self.trucks)
            ],
            "bins": [
                {"id": i,
                 "x": int(self.space.positions[b][0]),
                 "y": int(self.space.positions[b][1]),
                 "fill": int(b.fill)}
                for i, b in enumerate(self.bins)
            ],
        })
        self.t += 1


# ==============================================================
# Visualization
# ==============================================================

def visualize(parameters, steps=400):
    model = WasteModel(parameters)
    model.setup()

    def draw(_frame):
        ax.clear()
        ax.set_title(f"Step {model.t}")
        ax.set_xlim(-0.5, 19.5)
        ax.set_ylim(-0.5, 19.5)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_aspect("equal")

        model.step()

        # streets
        xs = [x for x, y in model.street_coords]
        ys = [y for x, y in model.street_coords]
        ax.plot(xs, ys, ".", markersize=3, color="lightgray")

        # depot
        dx, dy = model.deposit_pos
        ax.plot(dx, dy, "D", color="blue", markersize=10)

        # bins
        for b in model.bins:
            x, y = model.space.positions[b]
            frac = b.fill / b.capacity_max
            c = "green" if frac <= 0.4 else "yellow" if frac <= 0.7 else "red"
            ax.plot(x, y, "s", color=c, markersize=10)

        # trucks
        for t in model.trucks:
            x, y = model.space.positions[t]
            c = "purple" if t.carrying < t.capacity_max else "black"
            ax.plot(x, y, "^", color=c, markersize=10)

        # legend
        bin_g = mlines.Line2D([], [], color="green", marker="s", linestyle="None", markersize=8, label="Bin (low)")
        bin_y = mlines.Line2D([], [], color="yellow", marker="s", linestyle="None", markersize=8, label="Bin (mid)")
        bin_r = mlines.Line2D([], [], color="red", marker="s", linestyle="None", markersize=8, label="Bin (high)")
        truck_patch = mlines.Line2D([], [], color="purple", marker="^", linestyle="None", markersize=8, label="Truck")
        depot_patch = mlines.Line2D([], [], color="blue", marker="D", linestyle="None", markersize=8, label="Depot")
        ax.legend(handles=[bin_g, bin_y, bin_r, truck_patch, depot_patch], loc="upper right")

    fig, ax = plt.subplots(figsize=(6, 6))
    anim = FuncAnimation(fig, draw, frames=steps, interval=40, repeat=False)
    plt.show()


# ==============================================================
# Main (MVP)
# ==============================================================

def main():
    params = {
        "capacity_max": 100,  # bin capacity
        "fill_step": 2,       # faster fills to create work
        "truck_capacity": 1,  # MVP: carry a single bin at a time
        "seed": 42,
    }
    visualize(params, steps=400)

if __name__ == "__main__":
    main()
