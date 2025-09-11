import os, sys, math, argparse
from typing import Dict, List, Tuple

# Allow running either as a script (python reto/unity_hybrid/plot_sim.py) or module (python -m reto.unity_hybrid.plot_sim)
THIS_DIR = os.path.dirname(__file__)
PARENT = os.path.dirname(THIS_DIR)
GRAND = os.path.dirname(PARENT)
for p in (GRAND, PARENT, THIS_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    # When executed as module
    from .config import CONFIG  # type: ignore
    from .city import City      # type: ignore
    from .sim import Simulation # type: ignore
except Exception:
    # Fallback script-style (paths inserted above)
    from config import CONFIG   # type: ignore
    from city import City       # type: ignore
    from sim import Simulation  # type: ignore


def build_tracks(frames) -> Dict[str, List[Tuple[float, float]]]:
    tracks: Dict[str, List[Tuple[float, float]]] = {}
    for fr in frames:
        for t in fr["trucks"]:
            tid = t["id"]
            tracks.setdefault(tid, []).append((float(t["x"]), float(t["y"])) )
    return tracks


def plot_sim(city: City, sim: Simulation, out_png: str | None, show: bool, animate: bool=False, fps: int=20):
    try:
        import matplotlib.pyplot as plt
        from matplotlib import cm
    except Exception as e:
        print("matplotlib is required for plotting. Install it with: pip install matplotlib")
        raise

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(0, city.w)
    ax.set_ylim(0, city.h)
    ax.set_title("Waste Collection Simulation (quick plot)")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    # Roads
    for r in city.roads:
        (x1, y1), (x2, y2) = r.polyline
        ax.plot([x1, x2], [y1, y2], color="#A0A0A0", lw=2, zorder=1)

    # Depot
    ax.scatter([city.depot[0]], [city.depot[1]], marker='*', s=180, c='#1f77b4', edgecolor='k', zorder=5, label='Depot')

    # Bins (color by fill frac)
    bins = sim.bins
    if bins:
        fracs = [b.fill / max(1, b.capacity) for b in bins]
        colors = [cm.Reds(min(0.99, f*0.9 + 0.1)) for f in fracs]
        ax.scatter([b.pos[0] for b in bins], [b.pos[1] for b in bins], s=60, c=colors, edgecolor='k', zorder=4, label='Bins')
        # Optional: draw curbs
        for b in bins:
            if getattr(b, 'curb', None) is not None:
                ax.plot([b.curb[0], b.pos[0]], [b.curb[1], b.pos[1]], color='#FFC107', lw=1, alpha=0.6, zorder=2)

    colors = ['#2ca02c', '#ff7f0e', '#9467bd', '#17becf', '#e377c2', '#8c564b']
    tracks = build_tracks(sim.frames)
    if not animate:
        # Static: full traces
        for i, (tid, pts) in enumerate(sorted(tracks.items())):
            if len(pts) < 2:
                # still mark single point
                if pts:
                    col = colors[i % len(colors)]
                    ax.scatter([pts[0][0]], [pts[0][1]], c=col, s=40, marker='o', zorder=4, label=f"{tid}")
                continue
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            col = colors[i % len(colors)]
            ax.plot(xs, ys, color=col, lw=2.0, alpha=0.9, zorder=3, label=f"{tid}")
            ax.scatter([xs[0]], [ys[0]], c=col, s=30, marker='o', zorder=4)
            ax.scatter([xs[-1]], [ys[-1]], c=col, s=30, marker='s', zorder=4)
    else:
        # Animated: show moving markers and trail so far
        from matplotlib import animation
        truck_ids = sorted({t['id'] for fr in sim.frames for t in fr['trucks']})
        id_to_color = {tid: colors[i % len(colors)] for i, tid in enumerate(truck_ids)}
        # Initialize artists
        markers = {}
        trails = {}
        for tid in truck_ids:
            markers[tid] = ax.scatter([], [], c=id_to_color[tid], s=60, edgecolor='k', zorder=6, label=tid)
            trails[tid], = ax.plot([], [], color=id_to_color[tid], lw=1.8, alpha=0.85, zorder=3)

        def init():
            return list(markers.values()) + [trails[t] for t in truck_ids]

        def update(frame_idx: int):
            fr = sim.frames[frame_idx]
            # Build partial tracks up to this frame
            partial_tracks = build_tracks(sim.frames[:frame_idx+1])
            artists = []
            for tid in truck_ids:
                pts = partial_tracks.get(tid, [])
                if pts:
                    x, y = pts[-1]
                    markers[tid].set_offsets([[x, y]])
                    xs = [p[0] for p in pts]
                    ys = [p[1] for p in pts]
                    trails[tid].set_data(xs, ys)
                artists.append(markers[tid])
                artists.append(trails[tid])
            return artists

        frames_n = len(sim.frames)
        interval_ms = 1000.0 / max(1, fps)
        anim = animation.FuncAnimation(fig, update, init_func=init, frames=frames_n, interval=interval_ms, blit=True)
        # If output specified and endswith .mp4 or .gif, try saving
        if out_png and (out_png.lower().endswith('.mp4') or out_png.lower().endswith('.gif')):
            try:
                if out_png.lower().endswith('.mp4'):
                    anim.save(out_png, fps=fps, dpi=140)
                else:
                    anim.save(out_png, writer='imagemagick', fps=fps, dpi=140)
                print(f"Saved animation to {out_png}")
            except Exception as e:
                print(f"Could not save animation ({e}); showing interactively instead.")
        # For interactive show, defer plt.show() below

    # Pickup events
    px, py = [], []
    for ev in sim.events:
        if ev.get('type') == 'pickup':
            # find the bin position for the event
            bid = ev.get('bin')
            b = next((bb for bb in sim.bins if str(bb.id) == str(bid)), None)
            if b is not None:
                px.append(b.pos[0]); py.append(b.pos[1])
    if px:
        ax.scatter(px, py, marker='x', s=40, c='k', zorder=6, label='Pickups')

    ax.legend(loc='upper right', fontsize=8)
    fig.tight_layout()

    if out_png and (not animate or (animate and not (out_png.lower().endswith('.mp4') or out_png.lower().endswith('.gif')))):
        fig.savefig(out_png, dpi=160)
        print(f"Saved plot to {out_png}")
    if show:
        plt.show()
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Quick plot for the waste collection simulation")
    ap.add_argument('--steps', type=int, default=400)
    ap.add_argument('--trucks', type=int, default=None)
    ap.add_argument('--bins', type=int, default=None)
    ap.add_argument('--policy', choices=['auction', 'dqn'], default='dqn',
                   help='RL by default (falls back to tabular if torch DQN unavailable)')
    ap.add_argument('--planner', choices=['graph', 'grid'], default='graph')
    ap.add_argument('--out', default=None, help='Save PNG/GIF/MP4 path (optional)')
    ap.add_argument('--animate', action='store_true', help='Animate truck movement (interactive or save .gif/.mp4)')
    ap.add_argument('--fps', type=int, default=20, help='Animation frames per second')
    ap.add_argument('--near-full-frac', type=float, help='Override NEAR_FULL_FRAC for quicker demo movement')
    ap.add_argument('--fill', type=str, help='Override BIN_FILL_PER_STEP as lo,hi (e.g. 1,3) to speed bin fills')
    ap.add_argument('--show', action='store_true', help='Show window')
    args = ap.parse_args()

    cfg = CONFIG.copy()
    if args.trucks is not None:
        cfg['N_TRUCKS'] = int(args.trucks)
    if args.bins is not None:
        cfg['N_BINS'] = int(args.bins)
    if args.policy is not None:
        cfg['POLICY'] = args.policy

    # Optional overrides for faster visible activity
    if args.near_full_frac is not None:
        cfg['NEAR_FULL_FRAC'] = float(args.near_full_frac)
    if args.fill:
        try:
            lo, hi = args.fill.split(',')
            cfg['BIN_FILL_PER_STEP'] = (int(lo), int(hi))
        except Exception:
            print('Ignoring malformed --fill (expected lo,hi)')

    city = City(cfg)
    sim = Simulation(cfg=cfg, city=city, planner=args.planner)
    sim.run(args.steps)

    plot_sim(city, sim, out_png=args.out, show=args.show, animate=args.animate, fps=args.fps)


if __name__ == '__main__':
    main()
