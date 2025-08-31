# dqn_env.py
import math
import numpy as np
from gymnasium import spaces

from sim import Simulation
from city import City
from negotiation import auction
from agents import dist


class MultiTruckEnv:
    """
    Multi-truck RL environment with:
      - Per-tick auction-based assignment (negotiation.auction)
      - Auto movement along preplanned routes unless the agent chooses WAIT
      - Recharge only at depot
      - Reward ≈ -(wage + energy + maintenance + penalties) + small service bonuses

    Actions (Discrete(5)):
      0: MOVE/CONTINUE (default; if no route/target -> treated as WAIT)
      1: RETURN_TO_DEPOT (plan route to depot, then move)
      2: (unused, behaves like MOVE)
      3: RECHARGE (only at depot, otherwise behaves like MOVE)
      4: WAIT
    """

    def __init__(self, cfg):
        base_cfg = cfg.copy()
        self.city = City(base_cfg)
        self.cfg = {**base_cfg, "plan_route_fn": self.city.plan_route}
        self.sim = Simulation(self.cfg, self.city)

        self.n_agents = int(cfg["N_TRUCKS"])
        self.max_steps = int(cfg["STEPS_PER_DAY"])
        self.current_step = 0

        # obs = truck(x,y,load,energy) + assigned (dist,fill) + nearest 3 bins (d,fill)*3 + truck_id
        self.obs_dim = (4 + 2 + 3 * 2) + 1
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(self.obs_dim,), dtype=np.float32)

        # Reward-Skalierung (nur für RL, nicht für day_costs)
        self.reward_scale = float(self.cfg.get("REWARD_SCALE", 0.01))
        # Optionales Cap pro Tick gegen Penalty-Kaskaden (None = aus)
        self.max_penalties_per_tick = self.cfg.get("MAX_PENALTIES_PER_TICK", None)

    # ---------- logging helper ----------
    def _log_frame(self):
        """Append one frame to sim.frames for visualization/export."""
        frame = {
            "t": self.sim.t,
            "trucks": [
                {
                    "id": t.tid,
                    "x": t.pos[0],
                    "y": t.pos[1],
                    "energy": t.energy,
                    "load": t.load,
                    "state": t.state,
                    "target": (None if t.target is None else {"x": t.target[0], "y": t.target[1]}),
                }
                for t in self.sim.trucks
            ],
            "bins": [
                {"id": b.id, "x": b.pos[0], "y": b.pos[1], "fill": b.fill, "cap": b.capacity}
                for b in self.sim.bins
            ],
            "events": [],  # per-step slice placeholder (global log ist in sim.events)
        }
        self.sim.frames.append(frame)

    def reset(self):
        self.city = City(self.cfg)
        self.cfg["plan_route_fn"] = self.city.plan_route
        self.sim = Simulation(self.cfg, self.city)
        self.current_step = 0
        self._log_frame()
        return self._get_obs_all()

    def step(self, actions):
        """
        actions: list[int] of length n_agents
        Returns: (obs_all, rewards_all, done_all, info)
        """
        dt = self.cfg["DT"]
        rewards = [0.0] * self.n_agents

        # 1) bins fill + overflow events
        lo, hi = self.cfg["BIN_FILL_PER_STEP"]
        rnd = self.sim._rnd()
        overflowed_ids = []
        for b in self.sim.bins:
            before = b.fill
            if b.step_fill(lo, hi, rnd) and before < b.capacity:
                overflowed_ids.append(b.id)
                self.sim.events.append({"t": self.sim.t, "type": "overflow", "bin": b.id})

        # 2) auction assignment
        auction(self.sim.bins, self.sim.trucks, self.sim.t, self.cfg, self.city.plan_route)

        # 3) action safety mask + RETURN_TO_DEPOT
        raw_actions = list(actions)
        masked_actions = []
        for idx, truck in enumerate(self.sim.trucks):
            a = raw_actions[idx]

            has_route = bool(truck.route_pts) or (truck.target is not None)
            is_assigned = truck.assigned_bin is not None
            carrying = truck.load > 0
            at_depot = dist(truck.pos, self.city.depot) < 1.0

            # NEW: action 1 = RETURN_TO_DEPOT
            if a == 1 and not at_depot:
                truck.assigned_bin = None
                truck.target = self.city.depot
                # sofort Route planen, falls nötig
                if not has_route or not truck.route_pts:
                    route = self.city.plan_route(truck.pos, self.city.depot)
                    truck.assign_target(route, None, self.city.depot)
                a = 0  # danach normal "MOVE"

            # Kein Plan/Route: MOVE bringt nichts -> als WAIT behandeln
            if a == 0 and not has_route:
                a = 4  # WAIT

            # WAIT ist schädlich, wenn en-route/assigned/carrying -> zwingend MOVE
            if a == 4 and (has_route or is_assigned or carrying):
                a = 0  # MOVE

            # RECHARGE nur am Depot, sonst weiterfahren
            if a == 3 and not at_depot:
                a = 0  # MOVE

            masked_actions.append(a)

        # 4) apply actions; Reward skalieren
        for idx, truck in enumerate(self.sim.trucks):
            r = truck.apply_action(masked_actions[idx], self.sim.bins, self.city.depot, self.cfg)

            # kleine Strafnase für intention zu WAIT in ungeeigneten Zuständen
            a_raw = raw_actions[idx]
            if a_raw == 4:
                has_route = bool(truck.route_pts) or (truck.target is not None)
                if has_route or (truck.assigned_bin is not None) or (truck.load > 0):
                    r -= 1.0

            # zentrale Reward-Skalierung
            rewards[idx] += r * self.reward_scale

        # 5) Overflow-Penalties (für RL skaliert, für Ökonomie voll)
        if overflowed_ids:
            if isinstance(self.max_penalties_per_tick, int) and len(overflowed_ids) > self.max_penalties_per_tick:
                overflowed_ids = overflowed_ids[: self.max_penalties_per_tick]

            pen_eur = float(self.cfg["OVERFLOW_PENALTY_EUR"])
            pen_scaled = pen_eur * self.reward_scale

            for bid in overflowed_ids:
                owners = [i for i, t in enumerate(self.sim.trucks) if t.assigned_bin == bid]
                if owners:
                    for i in owners:
                        rewards[i] -= pen_scaled
                else:
                    bpos = next(b.pos for b in self.sim.bins if b.id == bid)
                    i_star = min(
                        range(self.n_agents),
                        key=lambda i: math.hypot(
                            self.sim.trucks[i].pos[0] - bpos[0],
                            self.sim.trucks[i].pos[1] - bpos[1],
                        ),
                    )
                    rewards[i_star] -= pen_scaled

                # Ökonomik unverändert in €
                self.sim.day_costs["penalties_eur"] += pen_eur

        # 6) wage aggregation (per-truck wage wurde bereits im truck.apply_action vom Reward abgezogen)
        self.sim._wage_tick()

        # 7) rolling energy/maintenance aggregation
        self.sim.day_costs["energy_eur"] = sum(t.costs_eur["energy"] for t in self.sim.trucks)
        self.sim.day_costs["maintenance_eur"] = sum(t.costs_eur["maint"] for t in self.sim.trucks)

        # 8) log frame + Zeit fortschreiben
        self._log_frame()
        self.sim.t += dt
        self.current_step += 1

        obs = self._get_obs_all()
        done_flag = self.current_step >= self.max_steps
        dones = [done_flag] * self.n_agents
        info = {"costs": self.sim.summary_costs()}
        return obs, rewards, dones, info

    # ---------- observation builder ----------
    def _norm_d(self, x1, y1, x2, y2):
        """Distance normalized by map width/height, clipped to [0,1] to match Box."""
        w, h = self.cfg["MAP_SIZE"]
        dx = (x2 - x1) / max(1e-9, w)
        dy = (y2 - y1) / max(1e-9, h)
        d = math.hypot(dx, dy)
        return float(min(1.0, d))

    def _get_obs_all(self):
        return [self._get_obs(i, tr) for i, tr in enumerate(self.sim.trucks)]

    def _get_obs(self, idx, truck):
        w, h = self.cfg["MAP_SIZE"]
        x, y = truck.pos
        load = truck.load / self.cfg["TRUCK_CAPACITY"]
        energy = truck.energy / self.cfg["ENERGY_MAX"]

        # assigned bin features
        assigned_d, assigned_fill = 0.0, 0.0
        if truck.assigned_bin:
            b = next((bb for bb in self.sim.bins if bb.id == truck.assigned_bin), None)
            if b:
                assigned_d = self._norm_d(x, y, b.pos[0], b.pos[1])
                assigned_fill = b.fill / b.capacity

        # nearest 3 bins (distance, fill)
        bins = sorted(self.sim.bins, key=lambda bb: math.hypot(bb.pos[0] - x, bb.pos[1] - y))[:3]
        b_feats = []
        for b in bins:
            d = self._norm_d(x, y, b.pos[0], b.pos[1])
            f = b.fill / b.capacity
            b_feats += [d, f]
        while len(b_feats) < 6:
            b_feats.append(0.0)

        # normalized truck ID (for symmetry breaking in shared policy)
        truck_id_norm = idx / max(1, self.n_agents - 1) if self.n_agents > 1 else 0.0

        base = [
            x / w,
            y / h,
            load,
            energy,
            assigned_d,
            assigned_fill,
        ] + b_feats

        return np.array(base + [truck_id_norm], dtype=np.float32)
