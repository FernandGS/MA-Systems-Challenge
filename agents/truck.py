import agentpy as ap
import random

class Truck(ap.Agent):
    def setup(self):
        self.distance_traveled = 0
        self.messages_exchanged = 0

    def step(self):
        # Move randomly on the grid
        direction = random.choice([(1,0), (-1,0), (0,1), (0,-1)])
        self.model.space.move_by(self, direction)
        self.distance_traveled += 1
