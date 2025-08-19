import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from models.waste_model import WasteModel

parameters = {
    'capacity_max': 100,
    'fill_step': 1
}

model = WasteModel(parameters)
model.setup()

fig, ax = plt.subplots(figsize=(5,5))

def draw_frame(frame):
    ax.clear()
    ax.set_title(f"Step {frame}")
    ax.set_xlim(-0.5, 19.5)
    ax.set_ylim(-0.5, 19.5)
    ax.set_xticks([])
    ax.set_yticks([])

    # Step the model
    model.step()

    # Draw bins with color depending on fill
    for b in model.bins:
        x, y = model.space.positions[b]
        percent = b.fill / b.capacity_max
        if percent <= 0.2:
            color = "green"
        elif percent <= 0.6:
            color = "yellow"
        else:
            color = "red"
        ax.plot(x, y, "s", markersize=10, color=color)

    # Draw trucks
    for tr in model.trucks:
        x, y = model.space.positions[tr]
        ax.plot(x, y, "^", color="purple", markersize=10)

ani = FuncAnimation(fig, draw_frame, frames=100, interval=500, repeat=False)
plt.show()
