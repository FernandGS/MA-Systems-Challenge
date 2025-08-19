import agentpy as ap
import random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# --- Define Agents ---

class Bin(ap.Agent):
    def setup(self):
        self.messages_exchanged = 0  # optional attribute for future use


class Truck(ap.Agent):
    def setup(self):
        self.distance_traveled = 0
        self.messages_exchanged = 0

    def step(self):
        # Move randomly on the grid
        direction = random.choice([(1,0), (-1,0), (0,1), (0,-1)])
        self.model.space.move_by(self, direction)
        self.distance_traveled += 1


# --- Define Model ---

class WasteModel(ap.Model):
    def setup(self):
        self.space = ap.Grid(self, [20, 20], track_empty=True)

        self.bins = ap.AgentList(self, 5, Bin)
        self.trucks = ap.AgentList(self, 3, Truck)

        # Place bins randomly
        self.space.add_agents(self.bins, random=True)

        # FIXED positions for trucks
        fixed_positions = [(0, 0), (0, 19), (19, 0)]
        self.space.add_agents(self.trucks, positions=fixed_positions)

    def step(self):
        self.trucks.step()  # bins are static, trucks move


# --- Visualization with Matplotlib ---

model = WasteModel()
model.setup()  # Initialize agents

fig, ax = plt.subplots(figsize=(5, 5))

def draw_frame(frame):
    ax.clear()
    ax.set_title(f"Step {frame}")
    ax.set_xlim(-0.5, 19.5)
    ax.set_ylim(-0.5, 19.5)
    ax.set_xticks([])
    ax.set_yticks([])

    # Step the model
    model.step()

    # Draw bins
    for b in model.bins:
        x, y = model.space.positions[b]
        ax.plot(x, y, "bs", markersize=10)  # blue squares

    # Draw trucks
    for tr in model.trucks:
        x, y = model.space.positions[tr]
        ax.plot(x, y, "^", color="purple", markersize=10)  # purple triangles

ani = FuncAnimation(fig, draw_frame, frames=100, interval=500, repeat=False)
plt.show()
