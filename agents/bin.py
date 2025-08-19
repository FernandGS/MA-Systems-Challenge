import agentpy as ap
import random

class Bin(ap.Agent):
    def setup(self):
        self.capacity_max = self.p.capacity_max
        self.fill_step = self.p.fill_step
        self.fill = random.randint(0, self.capacity_max)
        self.requests_made = 0
        self.requests_served = 0
        self.request_active = False  # for truck request system

    def step(self):
        # Fill the bin gradually
        if self.fill < self.capacity_max:
            self.fill += self.fill_step
            if self.fill > self.capacity_max:
                self.fill = self.capacity_max

        # Optional: trigger request if above threshold
        if self.fill >= self.capacity_max * 0.5 and not self.request_active:
            self.make_request()

    def make_request(self):
        self.request_active = True
        self.requests_made += 1
        print(f"Bin {self.id} requests a truck! Fill: {self.fill}")
