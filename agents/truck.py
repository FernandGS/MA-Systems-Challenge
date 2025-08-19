import agentpy as ap
import random

class Truck(ap.Agent):
    def setup(self):
        self.distance_traveled = 0
        self.target_bin = None  # for request system

    def step(self):
        # Move toward target bin if any
        if self.target_bin:
            tx, ty = self.model.space.positions[self.target_bin]
            x, y = self.model.space.positions[self]
            dx = 0 if x == tx else (1 if tx > x else -1)
            dy = 0 if y == ty else (1 if ty > y else -1)
            self.model.space.move_by(self, (dx, dy))
            self.distance_traveled += 1

            # Check if arrived at bin
            if (x + dx, y + dy) == (tx, ty):
                self.collect_from_bin()

        else:
            # Random move if no target
            direction = random.choice([(1,0), (-1,0), (0,1), (0,-1)])
            self.model.space.move_by(self, direction)
            self.distance_traveled += 1

        # Reset bin fill if passing over it
        for b in self.model.bins:
            if self.model.space.positions[self] == self.model.space.positions[b]:
                b.fill = 0
                b.request_active = False
                b.requests_served += 1

    def collect_from_bin(self):
        bin = self.target_bin
        bin.fill = 0
        bin.request_active = False
        bin.requests_served += 1
        self.target_bin = None
        print(f"Truck {self.id} collected bin {bin.id}")
