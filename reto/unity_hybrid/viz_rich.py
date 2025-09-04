#!/usr/bin/env python3
# viz_rich.py
#
# Rich matplotlib visualization for the waste-collection sim:
#
# Modes:
#   1) live(sim, steps, ...)     -> drives sim.step() and updates in-place
#   2) playback(sim, ...)        -> animates existing sim.frames
#   3) CLI                       -> run directly: python viz_rich.py [options]
#
# Shows:
#   - Roads, depot, bins (fill% with color)
#   - Recently serviced bins (cooldown rings)
#   - Overflow flashes
#   - Trucks (triangle, energy-colored, load/energy labels)
#   - Per-truck routes (toggleable), targets, and states
#   - Quick KPIs and event ticker
#
# Usage examples (interactive from Python REPL):
#   >>> from config import CONFIG
#   >>> from city import City
#   >>> from sim import Simulation
#   >>> import viz_rich
#   >>> city = City(CONFIG)
#   >>> sim = Simulation(cfg=CONFIG, city=city, planner="graph")
#   >>> viz_rich.live(sim, steps=600, show_ids=True, interval_ms=0)
#
# Or, if you already ran:
#   >>> sim.run(600)
#   >>> viz_rich.playback(sim, show_ids=True, interval_ms=60)
#
# Usage examples (CLI from repo root):
#   $ python viz_rich.py --steps 400 --policy auction --mode live --ids
#   $ python viz_rich.py --steps 400 --policy dqn --mode playback
#   $ python viz_rich.py --steps 300 --planner grid --mode live
#
# Keys (both modes):
#   q = quit
#   p = pause/resume
#   n = step once (when paused)
#   r = toggle route lines

from __future__ import annotations
import time
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle, Rectangle
from matplotlib.animation import FuncAnimation

# ------------------------------------------------------------------------------
# Color helpers
# ------------------------------------------------------------------------------

def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else float(x))

def color_fill(frac: float):
    """Green -> Yellow -> Red based on fill fraction."""
    f = _clamp01(frac)
    if f <= 1/3:   return (0.18, 0.80, 0.44)   # green
    if f <= 2/3:   return (0.95, 0.77, 0.06)   # yellow
    return (0.91, 0.30, 0.24)                  # red

def color_energy(frac: float):
    """Red->Green based on energy fraction."""
    f = _clamp01(frac)
    return (1.0 - f, f, 0.0)

# ------------------------------------------------------------------------------
# Event buffer & helpers
# ------------------------------------------------------------------------------

@dataclass
class EventBuffer:
    max_items: int = 5
    window_s: float = 10.0  # keep events visible this many seconds
    items: List[dict] = field(default_factory=list)

    def add(self, ev: dict):
        self.items.append(ev)
        if len(self.items) > 50 * self.max_items:
            self.items = self.items[-50 * self.max_items:]

    def recent(self, now: float) -> List[dict]:
        out = [e for e in self.items if (now - float(e.get("t", 0))) <= self.window_s]
        # Keep tail-most max_items
        out = out[-self.max_items:]
        return out

# ------------------------------------------------------------------------------
# Core Viz class
# ------------------------------------------------------------------------------

class RichViz:
    def __init__(self, sim, show_ids: bool = True, draw_routes: bool = True):
        self.sim = sim
        self.city = sim.city
        self.cfg = sim.cfg
        self.show_ids = show_ids
        self.draw_routes = draw_routes

        self.fig, self.ax = plt.subplots(figsize=(9, 7))
        self.ax.set_aspect('equal', adjustable='box')
        self.ax.set_xlim(0, self.city.w)
        self.ax.set_ylim(0, self.city.h)
        self.ax.grid(True, alpha=0.2)

        # controls
        self.paused = False
        self.step_once = False
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

        # static map
        self._draw_static()

        # dynamic artists
        self.bin_patches: List[Circle] = []
        self.bin_labels: List = []
        self.cooldown_rings: List[Circle] = []  # faint ring if recently serviced
        self.truck_labels: List = []
        self.route_collections: List[LineCollection] = []
        self.target_segments: List[LineCollection] = []

        self._init_bins()
        self.trucks_scat = None
        self._init_trucks()

        # HUD texts
        self.hud_title = self.ax.text(0.01, 0.99, "", transform=self.ax.transAxes,
                                      ha="left", va="top", fontsize=11, color="#111")
        self.hud_kpi   = self.ax.text(0.01, 0.95, "", transform=self.ax.transAxes,
                                      ha="left", va="top", fontsize=9, color="#333")

        # Event ticker (right-top)
        self.hud_ev = self.ax.text(0.99, 0.99, "", transform=self.ax.transAxes,
                                   ha="right", va="top", fontsize=9, color="#222",
                                   bbox=dict(boxstyle="round,pad=0.3", fc="#f8f8f8", ec="#ddd"))

        self.evbuf = EventBuffer(max_items=6, window_s=8.0)

        self.fig.tight_layout()

        # Build quick map: bin id -> BinObj (for cooldown ring/etc.)
        self._id_to_bin = {b.id: b for b in self.sim.bins}

    # ---------------- controls ----------------
    def _on_key(self, ev):
        if ev.key == "q":
            plt.close(self.fig)
        elif ev.key == "p":
            self.paused = not self.paused
        elif ev.key == "n":
            self.step_once = True
        elif ev.key == "r":
            self.draw_routes = not self.draw_routes

    # ---------------- static background ----------------
    def _draw_static(self):
        # Roads
        segs = []
        for r in self.city.roads:
            (x1, y1), (x2, y2) = r.polyline
            segs.append([(x1, y1), (x2, y2)])
        lc = LineCollection(segs, linewidths=2.0, alpha=0.28, colors=[(0.75, 0.75, 0.75, 1.0)])
        self.ax.add_collection(lc)
        # Depot
        dx, dy = self.city.depot
        dep = Rectangle((dx - 2.5, dy - 2.5), 5.0, 5.0,
                        facecolor=(0.2, 0.7, 0.2, 0.9), edgecolor='k', lw=1.0, alpha=0.9, zorder=2)
        self.ax.add_patch(dep)
        self.ax.text(dx + 4.5, dy + 4.5, "Depot", fontsize=9, color='black', zorder=3)

    def _init_bins(self):
        for b in self.sim.bins:
            frac = (b.fill / max(1, b.capacity))
            c = Circle((b.pos[0], b.pos[1]), radius=2.0,
                       facecolor=color_fill(frac), edgecolor='k', lw=0.6, alpha=0.95, zorder=2)
            self.ax.add_patch(c)
            self.bin_patches.append(c)

            # cooldown ring (invisible initially)
            ring = Circle((b.pos[0], b.pos[1]), radius=2.6,
                          facecolor=(0,0,0,0), edgecolor=(0.2,0.6,1.0,0.0), lw=1.2, zorder=1)
            self.ax.add_patch(ring)
            self.cooldown_rings.append(ring)

            if self.show_ids:
                txt = self.ax.text(b.pos[0] + 2.2, b.pos[1] + 2.2,
                                   f"{b.id}", fontsize=8, color="#333", alpha=0.85)
                self.bin_labels.append(txt)

    def _init_trucks(self):
        xs = [t.pos[0] for t in self.sim.trucks]
        ys = [t.pos[1] for t in self.sim.trucks]
        cols = [color_energy(t.energy / max(1.0, self.cfg["ENERGY_MAX"])) for t in self.sim.trucks]
        self.trucks_scat = self.ax.scatter(xs, ys, s=85, marker='^', c=cols, edgecolors='k',
                                           linewidths=0.6, zorder=5)
        if self.show_ids:
            for t in self.sim.trucks:
                txt = self.ax.text(t.pos[0] + 2.0, t.pos[1] + 2.0,
                                   f"{t.tid}", fontsize=9, color="#111", zorder=6)
                self.truck_labels.append(txt)

        # Per-truck route and target line collections
        for _ in self.sim.trucks:
            rc = LineCollection([], colors=[(0.0,0.0,0.0,0.28)], linewidths=1.6, zorder=3)
            self.ax.add_collection(rc)
            self.route_collections.append(rc)

            tc = LineCollection([], colors=[(0.6,0.6,0.6,0.65)], linewidths=1.0, linestyles='dashed', zorder=3)
            self.ax.add_collection(tc)
            self.target_segments.append(tc)

    # ---------------- per-frame update ----------------
    def update_from_state(self):
        """Use current sim state (for live stepping)."""
        tnow = float(self.sim.t)
        self._update_bins(tnow)
        self._update_trucks()
        self._update_routes()
        self._update_hud(tnow)

    def update_from_frame(self, frame_i: int):
        """Use a saved frame (for playback)."""
        fr = self.sim.frames[frame_i]
        tnow = float(fr["t"])

        # bins (from frame dict)
        for patch, b in zip(self.bin_patches, fr["bins"]):
            frac = (b["fill"] / max(1, b["cap"]))
            patch.set_facecolor(color_fill(frac))

        # trucks scatter
        tx = [t["x"] for t in fr["trucks"]]
        ty = [t["y"] for t in fr["trucks"]]
        cols = []
        for t in fr["trucks"]:
            efrac = t["energy"] / max(1.0, self.cfg["ENERGY_MAX"])
            cols.append(color_energy(efrac))
        self.trucks_scat.set_offsets(list(zip(tx, ty)))
        self.trucks_scat.set_color(cols)
        if self.show_ids:
            for lbl, t in zip(self.truck_labels, fr["trucks"]):
                lbl.set_position((t["x"] + 2.0, t["y"] + 2.0))
                lbl.set_text(f'{t["id"]}')

        # Route/targets are not directly in frames; approximate with target rays only.
        for tc, t in zip(self.target_segments, fr["trucks"]):
            segs = []
            if t["target"] is not None and self.draw_routes:
                a = (t["x"], t["y"])
                b = (t["target"]["x"], t["target"]["y"])
                segs.append([a, b])
            tc.set_segments(segs)
        for rc in self.route_collections:
            rc.set_segments([])  # no route polyline from frames only

        # HUD & events
        self._harvest_new_events_until(tnow)
        self._update_hud(tnow)

        # cooldown rings from BinObj timestamps if available
        self._update_cooldown_rings(tnow)

    # ----- helpers -----
    def _update_bins(self, now: float):
        for patch, b in zip(self.bin_patches, self.sim.bins):
            frac = (b.fill / max(1, b.capacity))
            patch.set_facecolor(color_fill(frac))
        self._update_cooldown_rings(now)

    def _update_cooldown_rings(self, now: float):
        # Draw a faint blue ring on bins serviced within last SERVICE_COOLDOWN_S/6 seconds.
        cool_s = float(self.cfg.get("SERVICE_COOLDOWN_S", 300.0))
        show_s = max(3.0, cool_s / 6.0)
        for ring, b in zip(self.cooldown_rings, self.sim.bins):
            last = float(getattr(b, "last_service_t", -1e9))
            dt = now - last
            alpha = 0.0
            if dt >= 0.0 and dt <= show_s:
                # fade out over show_s
                alpha = max(0.15, 1.0 - dt / show_s) * 0.8
            ec = list(ring.get_edgecolor())
            if isinstance(ec, (list, tuple)) and len(ec):
                c = list(ec[0]) if isinstance(ec[0], (list, tuple)) else list(ec)
            else:
                c = [0.2, 0.6, 1.0, alpha]
            c[-1] = alpha
            ring.set_edgecolor(tuple(c))

    def _update_trucks(self):
        xs = [t.pos[0] for t in self.sim.trucks]
        ys = [t.pos[1] for t in self.sim.trucks]
        cols = [color_energy(t.energy / max(1.0, self.cfg["ENERGY_MAX"])) for t in self.sim.trucks]
        self.trucks_scat.set_offsets(list(zip(xs, ys)))
        self.trucks_scat.set_color(cols)
        if self.show_ids:
            for lbl, t in zip(self.truck_labels, self.sim.trucks):
                lbl.set_position((t.pos[0] + 2.0, t.pos[1] + 2.0))
                lbl.set_text(f"{t.tid}")

    def _update_routes(self):
        # Draw current route polylines & dashed line to target
        for rc, tc, t in zip(self.route_collections, self.target_segments, self.sim.trucks):
            # target dashed
            tseg = []
            if t.target is not None and self.draw_routes:
                tseg.append([(t.pos[0], t.pos[1]), (t.target[0], t.target[1])])
            tc.set_segments(tseg)

            # route polyline
            segs = []
            pts = t.route_pts or []
            if self.draw_routes and len(pts) >= 1:
                idx = max(0, min(int(getattr(t, "route_i", 0) or 0), len(pts) - 1))
                # from pos to next waypoint
                if idx < len(pts):
                    nx, ny = pts[idx]
                    segs.append([(t.pos[0], t.pos[1]), (nx, ny)])
                    # rest of the polyline
                    for i in range(idx, len(pts) - 1):
                        ax, ay = pts[i]
                        bx, by = pts[i + 1]
                        segs.append([(ax, ay), (bx, by)])
            rc.set_segments(segs)

    def _harvest_new_events_until(self, now: float):
        # push any sim.events with t <= now
        # (for live we’ll just take whatever is present)
        for e in self.sim.events:
            # only push once by tagging
            if "_seen" in e:  # type: ignore
                continue
            if float(e.get("t", 0)) <= now:
                self.evbuf.add(e)
                e["_seen"] = True  # type: ignore

    def _format_ticker(self, now: float) -> str:
        lines = []
        for e in self.evbuf.recent(now):
            et = e.get("type")
            if et == "assign":
                lines.append(f'ASSIGN  T={e.get("truck")} -> {e.get("bin")}')
            elif et == "pickup":
                amt = int(e.get("amount", 0))
                lines.append(f'PICKUP  T={e.get("truck")} @ {e.get("bin")} (+{amt})')
            elif et == "drop":
                lines.append(f'DUMP    T={e.get("truck")}')
            elif et == "recharge":
                lines.append(f'RECHARGE T={e.get("truck")}')
            elif et == "overflow":
                lines.append(f'OVERFLOW bin={e.get("bin")}')
        return "\n".join(lines) if lines else "—"

    def _update_hud(self, now: float):
        # title
        self.hud_title.set_text(f"t = {int(now)} s")

        # Quick KPIs (distance, collected, events)
        total_collected = sum(int(e.get("amount", 0)) for e in self.sim.events if e.get("type") == "pickup")
        km = sum(float(getattr(t, "km_total", 0.0)) for t in self.sim.trucks)
        evs = len(self.sim.events)
        kpi = [
            f"Collected: {total_collected}  |  Fleet km: {km:.1f}  |  Events: {evs}",
        ]
        # Per-truck quick line: load/E/state (# stops since depot)
        per = []
        for t in self.sim.trucks:
            efrac = t.energy / max(1.0, self.cfg["ENERGY_MAX"])
            per.append(f'{t.tid}: load {t.load}/{self.cfg.get("TRUCK_CAPACITY", 0)}  E {int(efrac*100)}%  {t.state}')
        kpi.append("\n".join(per))
        self.hud_kpi.set_text("\n".join(kpi))

        # Ticker
        self._harvest_new_events_until(now)
        self.hud_ev.set_text(self._format_ticker(now))

# ------------------------------------------------------------------------------
# Public entry points
# ------------------------------------------------------------------------------

def live(sim, steps: int, show_ids: bool = True, interval_ms: int = 0):
    """Run the sim for `steps` with a rich live viz."""
    viz = RichViz(sim, show_ids=show_ids, draw_routes=True)
    for _ in range(int(steps)):
        # window closed?
        if not plt.fignum_exists(viz.fig.number):
            break
        if viz.paused and not viz.step_once:
            plt.pause(0.05)
            continue
        viz.step_once = False

        sim.step()
        viz.update_from_state()

        if interval_ms > 0:
            time.sleep(interval_ms / 1000.0)
        # allow GUI to breathe
        plt.pause(0.001)
    if plt.fignum_exists(viz.fig.number):
        plt.show()

def playback(sim, show_ids: bool = True, interval_ms: int = 60):
    """Animate an already-run simulation using stored frames/events."""
    if not sim.frames:
        raise RuntimeError("No frames to play back. Run sim.run(...) first.")
    viz = RichViz(sim, show_ids=show_ids, draw_routes=True)

    def _upd(i):
        viz.update_from_frame(i)
        return []

    FuncAnimation(viz.fig, _upd, frames=len(sim.frames),
                  interval=max(1, int(interval_ms)), blit=False, repeat=False)
    plt.show()

# --- CLI Entrypoint ---
if __name__ == "__main__":
    import argparse
    from config import CONFIG
    from city import City
    from sim import Simulation

    parser = argparse.ArgumentParser(description="Visualize waste collection simulation")
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--policy", choices=["auction", "dqn"], default="auction")
    parser.add_argument("--planner", choices=["graph", "grid"], default="graph")
    parser.add_argument("--mode", choices=["live", "playback"], default="live")
    parser.add_argument("--ids", action="store_true")
    parser.add_argument("--interval", type=int, default=40)
    parser.add_argument("--weights-dir", type=str, default=None,
                        help="DQN weights directory (e.g., dqn_weights). If provided, used for loading.")
    parser.add_argument("--freeze-dqn", action="store_true",
                        help="Run DQN in eval-only (no learning, no replay updates).")
    parser.add_argument("--weights-dir", type=str, default=None,
                        help="DQN weights directory (e.g., dqn_weights)")
    parser.add_argument("--freeze-dqn", action="store_true",
                        help="Disable DQN learning (eval only)")

    args = parser.parse_args()

    cfg = CONFIG.copy()
    cfg["POLICY"] = args.policy
    if args.weights_dir:
        cfg["DQN_WEIGHTS_DIR"] = args.weights_dir
    if args.freeze_dqn:
        cfg["DQN_TRAIN_ENABLED"] = False
        cfg["EPS_START"] = cfg.get("EPS_END", 0.05)
        cfg["EPS_DECAY"] = 1.0


    city = City(cfg)
    sim = Simulation(cfg=cfg, city=city, planner=args.planner)

    if args.mode == "live":
        live(sim, steps=args.steps, show_ids=args.ids, interval_ms=args.interval)
    else:
        sim.run(args.steps)
        playback(sim, show_ids=args.ids, interval_ms=args.interval)
