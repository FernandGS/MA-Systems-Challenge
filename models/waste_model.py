import agentpy as ap
from agents.bin import Bin
from agents.truck import Truck

class WasteModel(ap.Model):
    def setup(self):
        self.space = ap.Grid(self, [20, 20], track_empty=True)

        self.bins = ap.AgentList(self, 5, Bin)
        self.trucks = ap.AgentList(self, 3, Truck)

        # Place bins randomly
        self.space.add_agents(self.bins, random=True)

        # FIXED positions for trucks
        fixed_positions = [(0,0), (0,19), (19,0)]
        self.space.add_agents(self.trucks, positions=fixed_positions)

    def step(self):
        self.trucks.step()
