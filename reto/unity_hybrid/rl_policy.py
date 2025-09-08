from __future__ import annotations
from typing import Dict, List, Tuple, Optional
import math, sys, os, json

# Allow importing dqn_agent.py
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

try:
    from dqn_agent import DQNAgent  # type: ignore
except Exception as e:  # pragma: no cover
    DQNAgent = None  # type: ignore

from agents import Truck, BinObj, dist

Point = Tuple[float, float]


def _candidate_bins(bins: List[BinObj], truck: Truck, k: int, now: float, cooldown_s: float) -> List[BinObj]:
    cand = [b for b in bins if b.fill > 0 and (now - getattr(b, 'last_service_t', -1e9)) >= cooldown_s]
    # rank by fill fraction, then distance
    cand.sort(key=lambda b: ((b.fill / max(1, b.capacity)), -dist(truck.pos, b.pos)), reverse=True)
    return cand[:k]


def _state_vector(truck: Truck, bins: List[BinObj], city, cfg: dict, now: float) -> List[float]:
    k = int(cfg.get("DQN_K_CANDS", 6))
    cooldown = float(cfg.get("SERVICE_COOLDOWN_S", 300.0))
    cands = _candidate_bins(bins, truck, k, now, cooldown)
    # Truck features
    cap = float(cfg.get("TRUCK_CAPACITY", 300))
    load_frac = float(truck.load) / max(1.0, cap)
    energy_max = float(cfg.get("ENERGY_MAX", 100.0))
    energy_frac = float(truck.energy) / max(1.0, energy_max)
    dep_d = dist(truck.pos, city.depot)
    map_w, map_h = cfg.get("MAP_SIZE", (220.0, 160.0))
    dep_d_norm = dep_d / max(1.0, math.hypot(map_w, map_h))
    feats: List[float] = [load_frac, energy_frac, dep_d_norm]
    # Per-candidate: [fill_frac, dist_norm]
    for b in cands:
        fill_frac = float(b.fill) / max(1.0, float(b.capacity))
        d_norm = dist(truck.pos, b.pos) / max(1.0, math.hypot(map_w, map_h))
        feats.extend([fill_frac, d_norm])
    # pad to fixed size
    while len(feats) < 3 + 2 * k:
        feats.append(0.0)
    return feats


class DQNManager:
    def __init__(self, cfg: dict):
        if DQNAgent is None:
            raise RuntimeError("DQN disabled: torch/dqn_agent.py not available. Set POLICY='auction' or install torch.")
        self.cfg = cfg
        self.k = int(cfg.get("DQN_K_CANDS", 6))
        self.action_dim = self.k + 2  # K bins, + go_depot, + idle
        self.obs_dim = 3 + 2 * self.k
        # One independent agent per truck id
        self.agents = {}
        # Per-truck transition cache
        self.prev_state = {}
        self.prev_action = {}
        self.km_prev = {}
        # Metrics (epsilon tracking per agent id)
        self.last_eps = {}

    def _ensure_dir(self, path: str):
        try:
            os.makedirs(path, exist_ok=True)
        except Exception:
            pass

    def _weights_path(self, tid: str) -> str:
        base = self.cfg.get("DQN_WEIGHTS_DIR", "dqn_weights")
        # save under repo root (two levels up from this file)
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self._ensure_dir(os.path.join(root, base))
        return os.path.join(root, base, f"agent_{tid}.pt")

    def _get_agent(self, tid: str):  # -> DQNAgent
        if tid not in self.agents:
            agent = DQNAgent(self.obs_dim, self.action_dim, self.cfg)
            # Try load weights
            try:
                wpath = self._weights_path(tid)
                if os.path.exists(wpath):
                    import torch  # lazy import
                    state = torch.load(wpath, map_location=agent.device)
                    agent.q_net.load_state_dict(state)
                    agent.target_net.load_state_dict(state)
            except Exception:
                pass
            self.agents[tid] = agent
        return self.agents[tid]

    def start_step(self, trucks: List[Truck]):
        # snapshot movement for cost shaping
        for t in trucks:
            self.km_prev[t.tid] = float(getattr(t, 'km_total', 0.0))

    def select_and_assign(self, city, bins: List[BinObj], trucks: List[Truck], now: float, plan_route) -> List[Dict]:
        events: List[Dict] = []
        cooldown = float(self.cfg.get("SERVICE_COOLDOWN_S", 300.0))
        min_gap = int(self.cfg.get("MIN_FOLLOW_GAP_STEPS", 0))
        extra_hold = int(self.cfg.get("ANTI_TAILGATE_EXTRA_HOLD", 0))
        # Build a quick occupancy map of current starting nodes (rounded positions) to discourage piling
        occ = {(round(t.pos[0],1), round(t.pos[1],1)) for t in trucks if t.route_pts}
        for t in trucks:
            if t.assigned_bin or t.route_pts or t.assign_hold_steps > 0:
                continue
            # Build state
            s = _state_vector(t, bins, city, self.cfg, now)
            agent = self._get_agent(t.tid)
            a = agent.act(s)
            self.last_eps[t.tid] = getattr(agent, 'eps', None)
            self.prev_state[t.tid] = s
            self.prev_action[t.tid] = a
            # Decode action
            if a < self.k:
                cands = _candidate_bins(bins, t, self.k, now, cooldown)
                if a < len(cands):
                    b = cands[a]
                    curb = getattr(b, "curb", b.pos)
                    route = plan_route(t.pos, curb)
                    if not route or route[-1] != curb:
                        route = route + [curb]
                    # Anti-tailgating: if another truck is already at first waypoint, insert a temporary hold
                    if min_gap > 0 and route:
                        first_wp = route[0]
                        if first_wp in occ:
                            t.assign_hold_steps = max(t.assign_hold_steps, min_gap + extra_hold)
                            events.append({"type": "tailgate_hold", "truck": t.tid, "bin": b.id, "t": now})
                    t.assign_target(route, b.id, curb)
                    events.append({"type": "assign", "truck": t.tid, "bin": b.id, "t": now})
            elif a == self.k:
                # go depot
                route = plan_route(t.pos, city.depot)
                if not route or route[-1] != city.depot:
                    route = route + [city.depot]
                if min_gap > 0 and route:
                    if route[0] in occ:
                        t.assign_hold_steps = max(t.assign_hold_steps, min_gap + extra_hold)
                        events.append({"type": "tailgate_hold", "truck": t.tid, "bin": None, "t": now})
                t.assign_target(route, None, city.depot)
            else:
                # idle / no-op
                pass
        return events

    def end_step_and_learn(self, city, bins: List[BinObj], trucks: List[Truck], now: float, step_events: List[Dict]):
        # Global overflow penalty for this step
        r_overflow = float(self.cfg.get("RL_REWARD_OVERFLOW", -5.0))
        step_overflows = sum(1 for e in step_events if e.get("type") == "overflow")
        overflow_pen = r_overflow * step_overflows
        # Truck-specific rewards
        r_pick = float(self.cfg.get("RL_REWARD_PICKUP", 1.0))
        r_dump = float(self.cfg.get("RL_REWARD_DUMP", 1.0))
        cost_per_km = float(self.cfg.get("RL_COST_PER_KM", 10.0))
        for t in trucks:
            tid = t.tid
            if tid not in self.prev_state or tid not in self.prev_action:
                continue
            # movement cost
            km_prev = self.km_prev.get(tid, float(getattr(t, 'km_total', 0.0)))
            km_now = float(getattr(t, 'km_total', 0.0))
            delta_km = max(0.0, km_now - km_prev)
            reward = overflow_pen - cost_per_km * delta_km
            # pickups and dumps for this truck this step
            for e in step_events:
                if e.get("truck") == tid:
                    if e.get("type") == "pickup":
                        reward += r_pick * float(e.get("amount", 0))
                    elif e.get("type") == "drop":
                        reward += r_dump
            # Build next state
            s2 = _state_vector(t, bins, city, self.cfg, now)
            done = False  # episodic end handled externally if desired
            # Store transition and learn
            agent = self._get_agent(tid)
            agent.store(self.prev_state[tid], int(self.prev_action[tid]), float(reward), s2, done)
            agent.update()
            self.last_eps[tid] = getattr(agent, 'eps', None)
            # Periodic save
            try:
                save_every = int(self.cfg.get("DQN_SAVE_EVERY_STEPS", 0) or 0)
                if save_every > 0 and int(now) % save_every == 0:
                    import torch
                    torch.save(agent.q_net.state_dict(), self._weights_path(tid))
            except Exception:
                pass
        # clear caches for trucks updated this step
        self.prev_state.clear()
        self.prev_action.clear()
