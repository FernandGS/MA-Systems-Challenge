import agentpy as ap
from agents.bin import Bin
from agents.truck import Truck

class WasteModel(ap.Model):

    def setup(self):
        self.p.capacity_max = getattr(self.p, "capacity_max", 100)
        self.p.fill_step = getattr(self.p, "fill_step", 1)

        self.space = ap.Grid(self, [20, 20], track_empty=True)

        self.bins = ap.AgentList(self, 5, Bin)
        self.trucks = ap.AgentList(self, 3, Truck)

        # Place bins randomly
        self.space.add_agents(self.bins, random=True)

        # Fixed positions for trucks
        fixed_positions = [(0,0), (0,19), (19,0)]
        self.space.add_agents(self.trucks, positions=fixed_positions)

    def step(self):
        # Step bins first (fill & requests)
        for b in self.bins:
            b.step()

        # Assign free trucks to bins with active requests
        for b in self.bins:
            if b.request_active and all(t.target_bin != b for t in self.trucks):
                free_truck = next((t for t in self.trucks if t.target_bin is None), None)
                if free_truck:
                    free_truck.target_bin = b

        # Step trucks (move & empty bins)
        self.trucks.step()
