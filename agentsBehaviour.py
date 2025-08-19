import agentpy as ap
import random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# --- Define Agents ---

class Trash(ap.Agent):
    def setup(self):
        self.collected = False


class Bin(ap.Agent):
    def setup(self):
        self.trash_collected = 0

    def manage(self, trash_list):
        for t in trash_list:
            if self.model.space.positions[self] == self.model.space.positions[t]:
                t.collected = True
                self.trash_collected += 1


class Truck(ap.Agent):
    def step(self):
        # Move randomly on the grid
        direction = random.choice([(1,0), (-1,0), (0,1), (0,-1)])
        self.model.space.move_by(self, direction)

        # Collect trash if present
        for t in self.model.trash:
            if (not t.collected 
                and self.model.space.positions[self] == self.model.space.positions[t]):
                t.collected = True


# --- Define Model ---

class WasteModel(ap.Model):
    def setup(self):
        self.space = ap.Grid(self, [20, 20], track_empty=True)

        self.bins = ap.AgentList(self, 5, Bin)
        self.trucks = ap.AgentList(self, 3, Truck)
        self.trash = ap.AgentList(self, 30, Trash)

        self.space.add_agents(self.bins, random=True)
        self.space.add_agents(self.trucks, random=True)
        self.space.add_agents(self.trash, random=True)

    def step(self):
        self.trucks.step()
        for b in self.bins:
            b.manage(self.trash)

    def update(self):
        if all(t.collected for t in self.trash):
            self.stop()


# --- Visualization with Matplotlib ---

model = WasteModel()
model.setup()  # <-- FIX: Initialize agents before animation

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

    # Draw trash
    for t in model.trash:
        if not t.collected:
            x, y = model.space.positions[t]
            ax.plot(x, y, "go", markersize=6)  # green circles

    # Draw bins
    for b in model.bins:
        x, y = model.space.positions[b]
        ax.plot(x, y, "bs", markersize=10)  # blue squares

    # Draw trucks
    for tr in model.trucks:
        x, y = model.space.positions[tr]
        ax.plot(x, y, "r^", markersize=10)  # red triangles

ani = FuncAnimation(fig, draw_frame, frames=50, interval=500, repeat=False)
plt.show()
