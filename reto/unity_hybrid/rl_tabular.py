from __future__ import annotations
from typing import Dict, List, Tuple, Callable
import math, random

Point = Tuple[float, float]

def _euclid(a: Point, b: Point) -> float:
    return math.hypot(a[0]-b[0], a[1]-b[1])

class TabularQL:
    def __init__(self, cfg: Dict):
        self.q: Dict[Tuple, float] = {}
        self.epsilon = float(cfg.get("EPS_START", 0.8))
        self.eps_min = float(cfg.get("EPS_END", 0.05))
        self.eps_decay = float(cfg.get("EPS_DECAY", 0.997))
        self.gamma = float(cfg.get("GAMMA", 0.99))
        self.lr = float(cfg.get("LR", 5e-4))
        self.rnd = random.Random(int(cfg.get("SEED", 42)))

    def _key(self, s: Tuple, a: Tuple) -> Tuple:
        return (*s, *a)

    def select(self, state: Tuple, actions: List[Tuple]) -> Tuple:
        if not actions:
            return None
        if self.rnd.random() < self.epsilon:
            return self.rnd.choice(actions)
        # greedy
        best_a, best_q = None, -1e18
        for a in actions:
            q = self.q.get(self._key(state, a), 0.0)
            if q > best_q:
                best_q, best_a = q, a
        return best_a if best_a is not None else self.rnd.choice(actions)

    def update(self, s: Tuple, a: Tuple, r: float, s2: Tuple, a2s: List[Tuple]):
        key = self._key(s, a)
        max_next = 0.0
        if a2s:
            max_next = max(self.q.get(self._key(s2, a2), 0.0) for a2 in a2s)
        cur = self.q.get(key, 0.0)
        td = r + self.gamma * max_next - cur
        self.q[key] = cur + self.lr * td
        self.epsilon = max(self.eps_min, self.epsilon * self.eps_decay)

class QManager:
    """Simple per-truck tabular Q-learning with position-dependent actions and collision guard.
    """
    def __init__(self, cfg: Dict):
        self.cfg = cfg
        self.agent = TabularQL(cfg)

    def _discretize(self, p: Point) -> Tuple[int,int]:
        grid = float(self.cfg.get("GRID_SPACING", 20.0))
        return (int(round(p[0]/grid)), int(round(p[1]/grid)))

    def _actions(self, city, truck, bins, plan_route: Callable[[Point,Point], List[Point]], others: List[Point]) -> List[Tuple[str,str]]:
        # Action types: goto bin X (top-K requests by fill/urgency), goto depot, idle
        # Candidate bins limited by distance and availability
        cand = sorted(bins, key=lambda b: (-b.fill, _euclid(truck.pos, b.pos)))[: self.cfg.get("DQN_K_CANDS", 6)]
        actions: List[Tuple[str,str]] = [("idle","-")]
        actions.append(("depot","-"))
        for b in cand:
            actions.append(("bin", str(b.id)))
        # Collision avoidance: if any other truck is too close in front along same node, prioritize different action order
        safe_dist = float(self.cfg.get("SAFE_DISTANCE_M", 3.0))
        too_close = any(_euclid(truck.pos, o) < safe_dist for o in others)
        if too_close:
            # move depot and idle to front to reduce overlap
            actions = [("idle","-")] + [("depot","-")] + [a for a in actions if a[0] == "bin"]
        return actions

    def select_and_assign(self, city, bins, trucks, t_now: float, plan_route):
        events = []
        others = [t.pos for t in trucks]
        for trk in trucks:
            s = (self._discretize(trk.pos), int(trk.load > 0))
            a_space = self._actions(city, trk, bins, plan_route, [p for p in others if p is not trk.pos])
            a = self.agent.select(s, a_space)
            if a is None:
                continue
            at, arg = a
            # remember action on the truck for credit assignment
            try:
                trk.last_action = a
                trk.prev_pos = trk.pos
                trk.prev_load = trk.load
            except Exception:
                pass
            if at == "bin":
                events.append({"truck": trk.tid, "bin": arg})
            elif at == "depot":
                events.append({"truck": trk.tid, "bin": None})
            # idle does not create an event
        return events

    def end_step_and_learn(self, city, bins, trucks, t_now: float, step_events: List[Dict]):
        # Very simple reward: pickups and distance penalty
        pickups = sum(e.get("amount",0) for e in step_events if e.get("type") == "pickup")
        dist_pen = sum(getattr(t, 'last_step_dist', 0.0) for t in trucks)
        r = float(self.cfg.get("RL_REWARD_PICKUP", 0.1)) * pickups - float(self.cfg.get("RL_COST_PER_KM", 10.0)) * (dist_pen/1000.0)
        for trk in trucks:
            s = (self._discretize(trk.prev_pos), int(trk.prev_load > 0)) if hasattr(trk,'prev_pos') else (self._discretize(trk.pos), int(trk.load > 0))
            s2 = (self._discretize(trk.pos), int(trk.load > 0))
            a2s = self._actions(city, trk, bins, city.plan_route, [t.pos for t in trucks if t is not trk])
            # Use the action taken, if stored by truck; otherwise skip
            a_taken = getattr(trk, 'last_action', None)
            if a_taken is None:
                continue
            self.agent.update(s, a_taken, r, s2, a2s)
