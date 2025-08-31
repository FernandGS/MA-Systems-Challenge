# dqn_env.py
import math
import numpy as np
from sim import Simulation
from city import City
from gymnasium import spaces
from negotiation import auction

class MultiTruckEnv:
    def __init__(self, cfg):
        base_cfg = cfg.copy()
        self.city = City(base_cfg)
        self.cfg = {**base_cfg, "plan_route_fn": self.city.plan_route}
        self.sim = Simulation(self.cfg, self.city)

        self.n_agents = cfg["N_TRUCKS"]
        self.max_steps = cfg["STEPS_PER_DAY"]
        self.current_step = 0

        # obs: truck(x,y,load,energy) + assigned (dist,fill) + nearest 3 bins (d,fill)x3 + truck_id
        self.obs_dim = (4 + 2 + 3*2) + 1
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.obs_dim,), dtype=np.float32)

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
            "events": [],  # RL env already pushed events to sim.events; this is per-step slice
        }
        self.sim.frames.append(frame)

    def reset(self):
        self.city = City(self.cfg)
        self.cfg["plan_route_fn"] = self.city.plan_route
        self.sim = Simulation(self.cfg, self.city)
        self.current_step = 0
        # log initial state so preview() has a first frame
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

        # 3) mask & apply actions
        masked_actions = []
        for idx, truck in enumerate(self.sim.trucks):
            a = actions[idx]
            if a == 0 and (not truck.route_pts) and (truck.target is None):
                a = 4
            masked_actions.append(a)

        for idx, truck in enumerate(self.sim.trucks):
            r = truck.apply_action(masked_actions[idx], self.sim.bins, self.city.depot, self.cfg)
            rewards[idx] += r

        # 4) overflow penalties to responsible/nearest
        if overflowed_ids:
            pen = self.cfg["OVERFLOW_PENALTY_EUR"]
            for bid in overflowed_ids:
                owners = [i for i, t in enumerate(self.sim.trucks) if t.assigned_bin == bid]
                if owners:
                    for i in owners:
                        rewards[i] -= pen
                else:
                    bpos = next(b.pos for b in self.sim.bins if b.id == bid)
                    i_star = min(
                        range(self.n_agents),
                        key=lambda i: math.hypot(self.sim.trucks[i].pos[0] - bpos[0],
                                                 self.sim.trucks[i].pos[1] - bpos[1])
                    )
                    rewards[i_star] -= pen
                self.sim.day_costs["penalties_eur"] += pen

        # 5) wage + costs aggregation + time
        self.sim._wage_tick()
        # rolling energy/maint sums
        self.sim.day_costs["energy_eur"] = sum(t.costs_eur["energy"] for t in self.sim.trucks)
        self.sim.day_costs["maintenance_eur"] = sum(t.costs_eur["maint"] for t in self.sim.trucks)

        # log a frame for this tick
        self._log_frame()

        # advance sim time/step
        self.sim.t += dt
        self.current_step += 1

        obs = self._get_obs_all()
        done = self.current_step >= self.max_steps
        dones = [done] * self.n_agents
        info = {"costs": self.sim.summary_costs()}
        return obs, rewards, dones, info

    def _get_obs_all(self):
        return [self._get_obs(i, tr) for i, tr in enumerate(self.sim.trucks)]

    def _get_obs(self, idx, truck):
        x, y = truck.pos
        load = truck.load / self.cfg["TRUCK_CAPACITY"]
        energy = truck.energy / self.cfg["ENERGY_MAX"]

        # assigned target features
        assigned_d, assigned_fill = 0.0, 0.0
        if truck.assigned_bin:
            b = next((bb for bb in self.sim.bins if bb.id == truck.assigned_bin), None)
            if b:
                assigned_d = np.hypot(b.pos[0] - x, b.pos[1] - y) / max(1, self.cfg["MAP_SIZE"][0])
                assigned_fill = b.fill / b.capacity

        # nearest 3 bins (distance, fill)
        bins = sorted(self.sim.bins, key=lambda bb: np.hypot(bb.pos[0] - x, bb.pos[1] - y))[:3]
        b_feats = []
        for b in bins:
            d = np.hypot(b.pos[0] - x, b.pos[1] - y) / max(1, self.cfg["MAP_SIZE"][0])
            f = b.fill / b.capacity
            b_feats += [d, f]
        while len(b_feats) < 6:
            b_feats.append(0.0)

        truck_id_norm = idx / max(1, self.n_agents - 1) if self.n_agents > 1 else 0.0

        base = [
            x / self.cfg["MAP_SIZE"][0],
            y / self.cfg["MAP_SIZE"][1],
            load, energy,
            assigned_d, assigned_fill
        ] + b_feats

        return np.array(base + [truck_id_norm], dtype=np.float32)
