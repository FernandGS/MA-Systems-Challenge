# models/waste_model.py
import agentpy as ap
from agents.bin import Bin
from agents.truck import Truck
import random

def generate_positions(n, grid_size, min_dist):
    positions = []
    attempts = 0
    while len(positions) < n and attempts < 1000:
        x = random.randint(0, grid_size-1)
        y = random.randint(0, grid_size-1)
        if all(abs(x - px) + abs(y - py) >= min_dist for px, py in positions):
            positions.append((x, y))
        attempts += 1
    if len(positions) < n:
        raise ValueError("Could not place all bins with the required minimum distance.")
    return positions

class WasteModel(ap.Model):
    def setup(self):
        self.p.capacity_max = getattr(self.p, "capacity_max", 100)
        self.p.fill_step = getattr(self.p, "fill_step", 1)

        self.space = ap.Grid(self, [20, 20], track_empty=True)

        self.bins = ap.AgentList(self, 12, Bin)
        self.trucks = ap.AgentList(self, 3, Truck)

        # Use generate_positions to place bins
        bin_positions = generate_positions(len(self.bins), 20, min_dist=5)
        self.space.add_agents(self.bins, positions=bin_positions)

        # Fixed positions for trucks
        fixed_positions = [(0, 0), (0, 19), (19, 0)]
        self.space.add_agents(self.trucks, positions=fixed_positions)

    def step(self):
        for b in self.bins:
            b.step()
        for b in self.bins:
            if b.request_active and all(t.target_bin != b for t in self.trucks):
                free_truck = next((t for t in self.trucks if t.target_bin is None), None)
                if free_truck:
                    free_truck.target_bin = b
        self.trucks.step()
