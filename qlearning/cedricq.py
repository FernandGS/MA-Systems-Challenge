import agentpy as ap
import random
import json
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.lines as mlines
from collections import deque
from math import inf

# ==============================================================
# Helper: BFS next step along streets
# ==============================================================

def next_step_bfs(start, goal, street_coords):
    if goal is None or start == goal:
        return start

    # If goal somehow isn't a street cell, snap it to the nearest street cell
    if goal not in street_coords:
        gx, gy = goal
        goal = min(street_coords, key=lambda p: abs(p[0]-gx) + abs(p[1]-gy))

    # Standard BFS
    q = deque([(start, [])])
    seen = {start}
    moves = [(1,0), (-1,0), (0,1), (0,-1)]
    while q:
        cur, path = q.popleft()
        for dx, dy in moves:
            nb = (cur[0]+dx, cur[1]+dy)
            if nb == goal:
                return path[0] if path else nb
            if nb in street_coords and nb not in seen:
                seen.add(nb)
                q.append((nb, path+[nb]))

    # Fallback: no discovered path (shouldn't happen on your lattice)
    # take a greedy Manhattan step toward the goal if that neighbor is a street
    sx, sy = start
    gx, gy = goal
    candidates = sorted([(sx+1, sy), (sx-1, sy), (sx, sy+1), (sx, sy-1)],
                        key=lambda p: abs(p[0]-gx)+abs(p[1]-gy))
    for c in candidates:
        if c in street_coords:
            return c
    return start



# ==============================================================
# Agents
# ==============================================================

class Bin(ap.Agent):
    def setup(self):
        self.capacity_max = self.p["capacity_max"]
        self.fill_step = self.p["fill_step"]
        # start with some trash to create work immediately
        self.fill = random.randint(self.capacity_max//3, self.capacity_max)

    def step(self):
        if self.fill < self.capacity_max:
            self.fill = min(self.capacity_max, self.fill + self.fill_step)


class Truck(ap.Agent):
    def setup(self):
        # capacities / energy
        self.capacity_max = self.p["truck_capacity"]
        self.carrying = 0
        self.energy = 100  # 0..100

        # Q-learning hyperparams
        self.alpha = self.p["alpha"]
        self.gamma = self.p["gamma"]
        self.epsilon = self.p["epsilon_start"]
        self.epsilon_min = self.p["epsilon_min"]
        self.epsilon_decay = self.p["epsilon_decay"]

        # Actions (planner collapses 4 moves into a single "move_towards_target")
        self.actions = ["move", "pickup", "drop", "recharge", "wait"]

        # Reward params
        self.R = self.p["rewards"]
        self.ETA = self.p["potential_eta"]  # shaping weight

        # Q-table
        self.Q = {}
        if hasattr(self.model, "loaded_Q") and self.model.loaded_Q is not None:
            idx = self.model.trucks.index(self)
            if idx < len(self.model.loaded_Q) and isinstance(self.model.loaded_Q[idx], dict):
                self.Q = self.model.loaded_Q[idx]

    # ------------ utilities ------------
    def pos(self):
        return self.model.space.positions[self]

    def manhattan(self, a, b):
        return abs(a[0]-b[0]) + abs(a[1]-b[1])

    def _bucket_energy(self, e):
        return int(max(0, min(100, e)) // 10) * 10

    def get_state(self):
        x, y = self.pos()
        # compact state; we don’t put target bin id in state to keep it small
        return (int(x), int(y), int(self.carrying), self._bucket_energy(self.energy))

    def ensure_state(self, s):
        if s not in self.Q:
            self.Q[s] = {a: 0.0 for a in self.actions}

    # ------------ target selection ------------
    def fullest_bin(self):
        """Return (pos, bin, fullness_ratio) of the fullest non-empty bin (tie by distance)."""
        best = None
        best_key = None
        mypos = self.pos()
        for b in self.model.bins:
            if b.fill > 0:
                bpos = self.model.space.positions[b]
                fullness = b.fill / b.capacity_max
                # sort by fullness desc, then distance asc
                key = (-fullness, self.manhattan(mypos, bpos))
                if best is None or key < best_key:
                    best, best_key = b, key
        if best is None:
            return (None, None, 0.0)
        bpos = self.model.space.positions[best]
        return (bpos, best, best.fill / best.capacity_max)

    def current_target(self):
        depot = self.model.deposit_pos
        # to be safe, always prefer depot if low energy or already loaded
        if self.energy <= 10 or self.carrying >= self.capacity_max:
            return depot, None, None
        # else pick fullest non-empty bin
        bpos, b, fullness = self.fullest_bin()
        if bpos is not None:
            return bpos, b, fullness
        # fallback: go to depot (never return None)
        return depot, None, None

    # ------------ masked action set ------------
    def valid_actions(self):
        acts = ["move"]  # always allow moving 1 BFS step toward current target

        x, y = self.pos()
        depot = self.model.deposit_pos

        # pickup only if a bin with fill>0 is here and capacity left
        if self.carrying < self.capacity_max and any(
            self.model.space.positions[b] == (x, y) and b.fill > 0 for b in self.model.bins
        ):
            acts.append("pickup")

        # drop only at depot with any load
        if (x, y) == depot and self.carrying > 0:
            acts.append("drop")

        # recharge only at depot if energy < 100
        if (x, y) == depot and self.energy < 100:
            acts.append("recharge")

        return acts


    # ------------ policy ------------
    def choose_action(self, s):
        self.ensure_state(s)
        acts = self.valid_actions()
        if random.random() < self.epsilon:
            return random.choice(acts)
        max_v = max(self.Q[s][a] for a in acts)
        best = [a for a in acts if self.Q[s][a] == max_v]
        return random.choice(best)

    # ------------ reward ------------
    def get_reward(self, *, s_old, s_new, target_old, target_new,
                   moved, pickup_units, fullness_before, dumped_units,
                   overflowed, energy_zero):
        r = 0.0
        R = self.R
        # small punishments
        r += R["step_cost"]
        r += R["action_cost"]
        if moved:
            r += R["move_cost"]

        # terminal-ish events
        if dumped_units > 0:
            r += dumped_units * R["dump_unit_reward"]

        if pickup_units > 0:
            r += pickup_units * R["pickup_base_bonus_per_unit"]
            if fullness_before is not None:
                r += R["pickup_fullness_bonus_scale"] * fullness_before

        if overflowed:
            r += R["overflow_punish"]
        if energy_zero:
            r += R["energy_zero_punish"]

        # potential-based shaping: distance-to-target decrease
        #   Φ(s) = -dist_to_current_target(s)
        def phi(s, target):
            if target is None:  # no work
                return 0.0
            sx, sy, load, e = s
            return - (abs(sx - target[0]) + abs(sy - target[1]))

        r += self.ETA * (phi(s_old, target_old) - phi(s_new, target_new))

        return r

    # ------------ a single truck step ------------
    def step(self):
        # decay exploration
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        s = self.get_state()
        # figure out current target BEFORE acting (for shaping)
        tgt_pos_old, tgt_bin_old, tgt_fullness_old = self.current_target()

        a = self.choose_action(s)
        x0, y0 = self.pos()

        moved = False
        pickup_units = 0
        dumped_units = 0
        overflowed = False

        # movement resolution (planner picks the single best step)
        if a == "move":
            step_to = next_step_bfs((x0, y0), tgt_pos_old, self.model.street_coords)
            if step_to != (x0, y0):
                self.model.space.move_to(self, step_to)
                moved = True
                self.energy = max(0, self.energy - 1)

        elif a == "pickup":
            capacity_left = self.capacity_max - self.carrying
            if capacity_left <= 0:
                overflowed = True
            else:
                for b in self.model.bins:
                    if self.model.space.positions[b] == (x0, y0) and b.fill > 0:
                        amount = min(capacity_left, b.fill)
                        pickup_units = amount
                        b.fill -= amount
                        self.carrying += amount
                        break

        elif a == "drop":
            if (x0, y0) == self.model.deposit_pos and self.carrying > 0:
                dumped_units = self.carrying
                self.carrying = 0

        elif a == "recharge":
            if (x0, y0) == self.model.deposit_pos and self.energy < 100:
                self.energy = 100  # small positive reward comes via action_cost being small negative; add more if wanted

        elif a == "wait":
            pass

        energy_zero = (self.energy <= 0)

        s2 = self.get_state()
        # new target AFTER acting (for shaping term)
        tgt_pos_new, _, _ = self.current_target()

        r = self.get_reward(
            s_old=s, s_new=s2,
            target_old=tgt_pos_old, target_new=tgt_pos_new,
            moved=moved, pickup_units=pickup_units,
            fullness_before=tgt_fullness_old if pickup_units > 0 else None,
            dumped_units=dumped_units,
            overflowed=overflowed, energy_zero=energy_zero
        )

        self.ensure_state(s2)
        self.Q[s][a] += self.alpha * (r + self.gamma * max(self.Q[s2].values()) - self.Q[s][a])


# ==============================================================
# Model
# ==============================================================

def generate_positions(n, allowed_positions, min_dist):
    positions, tries = [], 0
    allowed = list(allowed_positions)
    while len(positions) < n and tries < 5000:
        cand = random.choice(allowed)
        if all(abs(cand[0]-px)+abs(cand[1]-py) >= min_dist for px, py in positions):
            positions.append(cand)
        tries += 1
    if len(positions) < n:
        raise ValueError("Could not place agents.")
    return positions


class WasteModel(ap.Model):
    def setup(self):
        random.seed(self.p.get("seed", 42))
        self.space = ap.Grid(self, [20, 20], track_empty=True)

        # Street lattice
        self.street_coords = set()
        for i in range(20):
            self.street_coords |= {(i, 4), (i, 6), (i, 8), (i, 12), (i, 16)}
            self.street_coords |= {(4, i), (8, i), (12, i), (16, i)}

        self.deposit_pos = (1, 4)

        self.bins = ap.AgentList(self, self.p["n_bins"], Bin)
        self.trucks = ap.AgentList(self, self.p["n_trucks"], Truck)

        allowed_bin_positions = self.street_coords - {self.deposit_pos}
        bin_positions = generate_positions(len(self.bins), allowed_bin_positions, 5)
        self.space.add_agents(self.bins, positions=bin_positions)

        truck_positions = generate_positions(len(self.trucks), self.street_coords, 10)
        self.space.add_agents(self.trucks, positions=truck_positions)

        self.loaded_Q = None
        self.t = 0
        self.history = []

    def step(self):
        self.trucks.step()
        self.bins.step()

        self.history.append({
            "step": int(self.t),
            "trucks": [
                {"id": i,
                 "x": int(self.space.positions[t][0]),
                 "y": int(self.space.positions[t][1]),
                 "carrying": int(t.carrying),
                 "energy": int(t.energy)}
                for i, t in enumerate(self.trucks)
            ],
            "bins": [
                {"id": i,
                 "x": int(self.space.positions[b][0]),
                 "y": int(self.space.positions[b][1]),
                 "fill": int(b.fill)}
                for i, b in enumerate(self.bins)
            ],
        })
        self.t += 1


# ==============================================================
# Train / Save / Load / Visualize
# ==============================================================

def train_and_save(parameters, steps_total=80_000, save_q="trained_agents.pkl", save_sim="simulation.json"):
    model = WasteModel(parameters)
    model.setup()
    for _ in tqdm(range(steps_total), desc="Training"):
        model.step()

    with open(save_sim, "w") as f:
        json.dump(model.history[-1000:], f, indent=2)

    trained_Q = [truck.Q for truck in model.trucks]
    with open(save_q, "wb") as f:
        pickle.dump(trained_Q, f)

    print(f"✅ Training finished. Files saved:\n - {save_sim}\n - {save_q}")


def load_Q_if_available(path="trained_agents.pkl"):
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None


def visualize(parameters, steps=600, q_path="trained_agents.pkl"):
    model = WasteModel(parameters)
    model.setup()

    loaded = load_Q_if_available(q_path)
    if loaded is not None:
        model.loaded_Q = loaded
        for i, tr in enumerate(model.trucks):
            if i < len(loaded) and isinstance(loaded[i], dict):
                tr.Q = loaded[i]
            tr.epsilon = 0.02  # near-greedy in viz
        print("ℹ️ Loaded trained Q-tables for visualization.")
    else:
        print("ℹ️ No trained Q-tables found; visualizing untrained behavior.")

    fig, ax = plt.subplots(figsize=(6, 6))

    def draw(_frame):
        ax.clear()
        ax.set_title(f"Step {model.t}")
        ax.set_xlim(-0.5, 19.5)
        ax.set_ylim(-0.5, 19.5)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_aspect("equal")

        model.step()

        # streets
        xs = [x for x, y in model.street_coords]
        ys = [y for x, y in model.street_coords]
        ax.plot(xs, ys, ".", markersize=3, color="lightgray")

        # depot
        dx, dy = model.deposit_pos
        ax.plot(dx, dy, "D", color="blue", markersize=10)

        # bins
        for b in model.bins:
            x, y = model.space.positions[b]
            frac = b.fill / b.capacity_max
            if   frac <= 0.33: c = "green"
            elif frac <= 0.66: c = "yellow"
            else:              c = "red"
            ax.plot(x, y, "s", color=c, markersize=10)

            # show text with current fullness
            ax.text(x, y+0.3, f"{b.fill}/{b.capacity_max}", ha="center", va="bottom", fontsize=6)


        # trucks
        for t in model.trucks:
            x, y = model.space.positions[t]
            c = "purple" if t.carrying < t.capacity_max else "black"
            ax.plot(x, y, "^", color=c, markersize=10)
            ax.text(x, y+0.3, f"{t.carrying}/{t.capacity_max}", ha="center", va="bottom", fontsize=6)

        # legend
        bin_g = mlines.Line2D([], [], color="green", marker="s", linestyle="None", markersize=8, label="Bin (low)")
        bin_y = mlines.Line2D([], [], color="yellow", marker="s", linestyle="None", markersize=8, label="Bin (mid)")
        bin_r = mlines.Line2D([], [], color="red", marker="s", linestyle="None", markersize=8, label="Bin (high)")
        truck_patch = mlines.Line2D([], [], color="purple", marker="^", linestyle="None", markersize=8, label="Truck")
        depot_patch = mlines.Line2D([], [], color="blue", marker="D", linestyle="None", markersize=8, label="Depot")
        ax.legend(handles=[bin_g, bin_y, bin_r, truck_patch, depot_patch], loc="upper right")

    anim = FuncAnimation(fig, draw, frames=steps, interval=30, repeat=False)
    plt.show()
    return anim


# ==============================================================
# Default parameters & Main
# ==============================================================

def main():
    params = {
        # world
        "seed": 42,
        "n_bins": 10,
        "n_trucks": 3,

        # bins
        "capacity_max": 100,
        "fill_step": 1,

        # trucks
        "truck_capacity": 200,

        # RL
        "alpha": 0.1,
        "gamma": 0.9,
        "epsilon_start": 0.2,
        "epsilon_min": 0.02,
        "epsilon_decay": 0.999,

        # reward shaping
        "potential_eta": 0.5,  # distance-to-target shaping weight

        "rewards": {
            "step_cost": -0.01,
            "action_cost": -0.005,
            "move_cost": -0.01,
            "dump_unit_reward": +2,
            "pickup_base_bonus_per_unit": +5,
            "pickup_fullness_bonus_scale": +10000,
            "overflow_punish": -100_000_000.0,
            "energy_zero_punish": -1000.0
        }
    }

    # --- choose one ---
    train_and_save(params, steps_total=100_000)
    visualize(params, steps=600)

if __name__ == "__main__":
    main()
