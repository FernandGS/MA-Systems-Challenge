import agentpy as ap
import random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import json
from collections import deque

def next_step_bfs(start, goal, street_coords):
    if start == goal:
        return start
    queue = deque()
    queue.append((start, []))
    visited = set()
    visited.add(start)
    possible_moves = [(1,0), (-1,0), (0,1), (0,-1)]
    while queue:
        current, path = queue.popleft()
        for dx, dy in possible_moves:
            neighbor = (current[0] + dx, current[1] + dy)
            if neighbor == goal:
                return path[0] if path else neighbor
            if neighbor in street_coords and neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))
    return start  


# --- Define Agents ---

class Bin(ap.Agent):
    def setup(self):
        self.capacity_max = self.p.capacity_max
        self.fill_step = self.p.fill_step 
        self.fill = random.randint(0, self.capacity_max)

    def step(self):
        if self.fill < self.capacity_max:
            self.fill += self.fill_step
            if self.fill > self.capacity_max:
                self.fill = self.capacity_max

class Truck(ap.Agent):
    def setup(self):
        self.distance_traveled = 0
        self.messages_exchanged = 0
        self.target_bin = None
        self.capacity_max = self.p.truck_capacity
        self.carrying = 0      
        self.unloading = False

    def step(self):
        my_pos = self.model.space.positions[self]
        deposit_pos = self.model.deposit_pos  

        # Go to deposit if full 
        if self.carrying >= self.capacity_max or self.unloading:
            self.unloading = True
            next_pos = next_step_bfs(my_pos, deposit_pos, self.model.street_coords)
            if next_pos != my_pos:
                move = (next_pos[0] - my_pos[0], next_pos[1] - my_pos[1])
                self.model.space.move_by(self, move)
                self.distance_traveled += 1
            if self.model.space.positions[self] == deposit_pos:
                self.carrying = 0
                self.unloading = False
                self.target_bin = None
            return

        # Look for the nearest bin to empty
        if self.target_bin is None:
            bins_to_collect = [b for b in self.model.bins if (b.fill / b.capacity_max) > 0.6]
            available_bins = [b for b in bins_to_collect if b not in self.model.reserved_bins]

            if available_bins:
                target = min(
                    available_bins,
                    key=lambda b: abs(my_pos[0] - self.model.space.positions[b][0]) +
                                  abs(my_pos[1] - self.model.space.positions[b][1])
                )
                self.target_bin = target
                self.model.reserved_bins.add(target)  

        # Move towards target_bin if exists
        if self.target_bin is not None:
            target_pos = self.model.space.positions[self.target_bin]
            next_pos = next_step_bfs(my_pos, target_pos, self.model.street_coords)
            if next_pos != my_pos:
                move = (next_pos[0] - my_pos[0], next_pos[1] - my_pos[1])
                self.model.space.move_by(self, move)
                self.distance_traveled += 1

            # Empty the bin if reached
            if self.model.space.positions[self] == target_pos:
                self.target_bin.fill = 0
                self.carrying += 1
                self.model.reserved_bins.remove(self.target_bin)  
                self.target_bin = None
        else:
            # Random movement if no target
            possible_moves = [(1,0), (-1,0), (0,1), (0,-1)]
            valid_moves = [(dx, dy) for dx, dy in possible_moves
                           if (my_pos[0] + dx, my_pos[1] + dy) in self.model.street_coords]
            if valid_moves:
                move = random.choice(valid_moves)
                self.model.space.move_by(self, move)
                self.distance_traveled += 1


# --- Define Model ---

def generate_positions(n, allowed_positions, min_dist):
    positions = []
    attempts = 0
    allowed_positions = list(allowed_positions)
    while len(positions) < n and attempts < 1000:
        x, y = random.choice(allowed_positions)
        # Verifica distancia mínima con los ya colocados
        if all(abs(x - px) + abs(y - py) >= min_dist for px, py in positions):
            positions.append((x, y))
        attempts += 1
    if len(positions) < n:
        raise ValueError("No se pudieron colocar todos los bins con la distancia mínima requerida.")
    return positions

class WasteModel(ap.Model):
    def setup(self):
        self.space = ap.Grid(self, [20, 20], track_empty=True)
        self.street_coords = set()
        for i in range(20):
            self.street_coords.add((i, 4))
            self.street_coords.add((i, 6))
            self.street_coords.add((i, 8))
            self.street_coords.add((i, 12))
            self.street_coords.add((i, 16))
            self.street_coords.add((4, i))
            self.street_coords.add((8, i))
            self.street_coords.add((12, i))
            self.street_coords.add((16, i))

        self.deposit_pos = (1, 4) 
        self.bins = ap.AgentList(self, 10, Bin)
        self.trucks = ap.AgentList(self, 3, Truck)

        allowed_bin_positions = set(self.street_coords) - {self.deposit_pos}
        bin_positions = generate_positions(len(self.bins), allowed_bin_positions, min_dist=5)
        self.space.add_agents(self.bins, positions=bin_positions)

        fixed_positions = generate_positions(len(self.trucks), self.street_coords, min_dist=10)
        self.space.add_agents(self.trucks, positions=fixed_positions)

        self.reserved_bins = set()
        self.history = []

    def step(self):
        self.trucks.step()
        self.bins.step()   

        # Guardar snapshot
        state = {
            "step": self.t,
            "trucks": [
                {"id": i, "x": int(self.space.positions[t][0]), "y":0 , "z": int(self.space.positions[t][1]),
                 "carrying": t.carrying}
                for i, t in enumerate(self.trucks)
            ],
            "bins": [
                {"id": i, "x": int(self.space.positions[b][0]), "y":0 , "z": int(self.space.positions[b][1]),
                 "fill": b.fill, "fill_percent": round(b.fill / b.capacity_max * 100, 2)}
                for i, b in enumerate(self.bins)
            ]
        }
        self.history.append(state)


# --- Visualization with Matplotlib ---

def draw_frame(frame):
    ax.clear()
    ax.set_title(f"Step {frame}")
    ax.set_xlim(-0.5, 19.5)
    ax.set_ylim(-0.5, 19.5)
    ax.set_xticks([])
    ax.set_yticks([])

    # Step the model
    model.step()

    # Draw streets
    for (x, y) in model.street_coords:
        ax.plot(x, y, ".", color="lightgray", markersize=5, zorder=0)

    # Draw deposit
    dx, dy = model.deposit_pos
    ax.plot(dx, dy, "D", color="blue", markersize=10, zorder=1)

    # Draw bins
    for b in model.bins:
        x, y = model.space.positions[b]
        percent = b.fill / b.capacity_max
        if percent <= 0.4:
            bin_color = "green"
        elif percent <= 0.7:
            bin_color = "yellow"
        else:
            bin_color = "red"
        ax.plot(x, y, "s", color=bin_color, markersize=10, zorder=2)

    # Draw trucks
    for tr in model.trucks:
        x, y = model.space.positions[tr]
        capacity = tr.carrying
        if capacity < tr.capacity_max:
            truck_color = "purple"
        elif capacity == tr.capacity_max:
            truck_color = "black"
        ax.plot(x, y, "^", color= truck_color, markersize=10, zorder=3)

    # Legend
    bin_patch = mlines.Line2D([], [], color='green', marker='s', linestyle='None', markersize=10, label='Bin (green/yellow/red)')
    truck_patch = mlines.Line2D([], [], color='purple', marker='^', linestyle='None', markersize=10, label='Truck (purple/black)')
    deposit_patch = mlines.Line2D([], [], color='blue', marker='D', linestyle='None', markersize=10, label='Deposit')
    ax.legend(handles=[bin_patch, truck_patch, deposit_patch], loc='upper right')    


parameters = {
    'capacity_max': 100,   # Bin capacity
    'fill_step': 1,        # How much a bin fills each step
    'truck_capacity': 3    # Fill capacity of each truck
}

model = WasteModel(parameters)
results = model.run(steps=100)

with open("simulation.json", "w") as f:
    json.dump(model.history, f, indent=4)

model.setup()

fig, ax = plt.subplots(figsize=(5, 5))

ani = FuncAnimation(fig, draw_frame, frames=100, interval=500, repeat=False)
plt.show()