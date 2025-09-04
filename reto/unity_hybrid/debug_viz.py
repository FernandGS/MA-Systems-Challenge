#!/usr/bin/env python3
# debug_viz.py
#
# Minimal live visualization for the waste-collection sim.
# - Draws roads, depot, bins (color ~ fill%), trucks, and current routes.
# - Steps the simulation in a loop and updates the figure.
#
# Usage examples:
#   python debug_viz.py --steps 600 --agents 4 --bins 12 --planner graph
#   python debug_viz.py --steps 400 --policy dqn
#
# Controls:
#   q : quit
#   p : pause/resume (toggle)
#   n : step one frame when paused
#
# Notes:
# - Run from the repo root (same folder where config.py/city.py/sim.py live).
# - Colors are intentionally simple for debugging.
# - If your matplotlib backend complains on some environments, try:
#     MPLBACKEND=TkAgg python debug_viz.py ...

import sys
import time
import argparse
from copy import deepcopy

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle, Rectangle

# Make sure local modules are importable when run from repo root
# (adjust if your files are in a subfolder)
try:
    from config import CONFIG
    from city import City
    from sim import Simulation
except Exception as e:
    print("Error importing local modules (config/city/sim). Run from repo root.")
    raise

# ----------------------------
# Utility: simple color helpers
# ----------------------------

def clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else float(x))

def bin_color(fill: int, cap: int):
    """Return an RGB tuple for a bin based on fill ratio (blue -> red)."""
    ratio = clamp01((fill or 0) / max(1, cap or 1))
    # 0 -> blue-ish, 1 -> red-ish
    return (ratio, 0.0, 1.0 - ratio)

def energy_color(frac: float):
    """Green when high energy, red when low."""
    f = clamp01(frac)
    return (1.0 - f, f, 0.0)

# ----------------------------
# Visualization object
# ----------------------------

class DebugViz:
    def __init__(self, cfg, city: City, sim: Simulation, show_ids: bool = True):
        self.cfg = cfg
        self.city = city
        self.sim = sim
        self.show_ids = show_ids

        # Figure/axes
        self.fig, self.ax = plt.subplots(figsize=(8, 7))
        self.ax.set_aspect('equal', adjustable='box')
        self.ax.set_xlim(0, cfg["MAP_SIZE"][0])
        self.ax.set_ylim(0, cfg["MAP_SIZE"][1])
        self.ax.set_title("Waste Collection — Debug Viz")

        # Keyboard controls
        self.paused = False
        self.step_once = False
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

        # Static drawings
        self._static_init()

        # Dynamic artist containers
        self.bin_patches = []
        self.bin_labels = []
        self.truck_scat = None
        self.truck_labels = []
        self.route_collections = []  # one LineCollection per truck

        # Initialize dynamic artists
        self._init_bins()
        self._init_trucks()

        self.fig.tight_layout()

    # ---------- controls ----------
    def _on_key(self, ev):
        if ev.key == 'q':
            plt.close(self.fig)
        elif ev.key == 'p':
            self.paused = not self.paused
        elif ev.key == 'n':
            self.step_once = True

    # ---------- static map ----------
    def _static_init(self):
        # Roads as line segments
        segs = []
        for r in self.city.roads:
            (x1, y1), (x2, y2) = r.polyline
            segs.append([(x1, y1), (x2, y2)])
        lc = LineCollection(segs, linewidths=2.0, alpha=0.30)
        self.ax.add_collection(lc)

        # Depot as a small rectangle
        dx, dy = self.city.depot
        dep = Rectangle((dx - 2.0, dy - 2.0), 4.0, 4.0,
                        facecolor=(0.2, 0.7, 0.2), edgecolor='k', lw=1.0, alpha=0.8)
        self.ax.add_patch(dep)
        self.ax.text(dx + 4, dy + 4, "Depot", fontsize=9, color='black')

        self.ax.grid(True, alpha=0.2)

    # ---------- bins ----------
    def _init_bins(self):
        # Draw bins as small circles where b.pos is OFF-road; label with id
        for b in self.sim.bins:
            c = Circle((b.pos[0], b.pos[1]), radius=2.0,
                       facecolor=bin_color(b.fill, b.capacity),
                       edgecolor='k', lw=0.6, alpha=0.9)
            self.ax.add_patch(c)
            self.bin_patches.append(c)
            if self.show_ids:
                txt = self.ax.text(b.pos[0] + 2.2, b.pos[1] + 2.2,
                                   f"{b.id}", fontsize=8, color="black", alpha=0.7)
                self.bin_labels.append(txt)

    # ---------- trucks ----------
    def _init_trucks(self):
        xs = [t.pos[0] for t in self.sim.trucks]
        ys = [t.pos[1] for t in self.sim.trucks]
        # Scatter for trucks; color by energy
        cols = [energy_color(t.energy / max(1.0, self.cfg["ENERGY_MAX"])) for t in self.sim.trucks]
        self.truck_scat = self.ax.scatter(xs, ys, s=70, marker='^', c=cols, edgecolors='k', linewidths=0.6, zorder=5)

        # Labels
        if self.show_ids:
            for t in self.sim.trucks:
                txt = self.ax.text(t.pos[0] + 2.2, t.pos[1] + 2.2,
                                   f"{t.tid}", fontsize=9, color="black", zorder=6)
                self.truck_labels.append(txt)

        # One route LineCollection per truck (empty initially)
        for _ in self.sim.trucks:
            lc = LineCollection([], colors=[(0.0, 0.0, 0.0, 0.25)], linewidths=1.5, zorder=2)
            self.ax.add_collection(lc)
            self.route_collections.append(lc)

    # ---------- per-frame update ----------
    def update(self):
        # Update bins (color = fill ratio)
        for patch, b in zip(self.bin_patches, self.sim.bins):
            patch.set_facecolor(bin_color(b.fill, b.capacity))

        # Update trucks (position + color by energy)
        xs = [t.pos[0] for t in self.sim.trucks]
        ys = [t.pos[1] for t in self.sim.trucks]
        cols = [energy_color(t.energy / max(1.0, self.cfg["ENERGY_MAX"])) for t in self.sim.trucks]
        self.truck_scat.set_offsets(list(zip(xs, ys)))
        self.truck_scat.set_color(cols)

        # Update truck labels
        if self.show_ids:
            for lbl, t in zip(self.truck_labels, self.sim.trucks):
                lbl.set_position((t.pos[0] + 2.2, t.pos[1] + 2.2))

        # Update current planned routes for each truck
        for lc, t in zip(self.route_collections, self.sim.trucks):
            segs = []
            pts = t.route_pts
            if pts and len(pts) >= 2:
                # Draw remaining route from current position to the first route point, then along route
                first = pts[max(0, t.route_i)]
                segs.append([(t.pos[0], t.pos[1]), (first[0], first[1])])
                for i in range(max(0, t.route_i), len(pts) - 1):
                    a = pts[i]
                    b = pts[i + 1]
                    segs.append([(a[0], a[1]), (b[0], b[1])])
            lc.set_segments(segs)

        # Title status
        self.ax.set_title(f"Waste Collection — t={int(self.sim.t)}  |  events={len(self.sim.events)}")

        # Draw
        self.fig.canvas.draw_idle()
        plt.pause(0.001)

# ----------------------------
# Main loop
# ----------------------------

def main():
    parser = argparse.ArgumentParser(description="Rough live visualization for the waste sim.")
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--agents", type=int, default=None, help="Override N_TRUCKS")
    parser.add_argument("--bins", type=int, default=None, help="Override N_BINS")
    parser.add_argument("--bin-cap", type=int, default=None)
    parser.add_argument("--planner", choices=["graph", "grid"], default="graph")
    parser.add_argument("--policy", choices=["auction", "dqn"], default=None, help="Override policy")
    parser.add_argument("--dt", type=float, default=None, help="Sim DT override")
    parser.add_argument("--show-ids", action="store_true")
    parser.add_argument("--interval-ms", type=int, default=0, help="Sleep between frames (ms), purely visual")
    args = parser.parse_args()

    cfg = deepcopy(CONFIG)
    if args.agents is not None:
        cfg["N_TRUCKS"] = int(args.agents)
    if args.bins is not None:
        cfg["N_BINS"] = int(args.bins)
    if args.bin_cap is not None:
        cfg["BIN_CAPACITY"] = int(args.bin_cap)
    if args.policy is not None:
        cfg["POLICY"] = args.policy
    if args.dt is not None:
        cfg["DT"] = float(args.dt)

    city = City(cfg)
    sim = Simulation(cfg=cfg, city=city, planner=args.planner)

    viz = DebugViz(cfg, city, sim, show_ids=args.show_ids)

    # Run loop
    for _ in range(int(args.steps)):
        if not plt.fignum_exists(viz.fig.number):
            break  # window closed
        if viz.paused and not viz.step_once:
            plt.pause(0.05)
            continue
        viz.step_once = False

        sim.step()
        viz.update()

        if args.interval_ms > 0:
            time.sleep(args.interval_ms / 1000.0)

    # Keep window open at the end
    if plt.fignum_exists(viz.fig.number):
        plt.show()

if __name__ == "__main__":
    main()
