# dqn_env_multi.py
# Multi-agent Gym-like wrapper for Simulation.
# Each truck is its own agent.

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

        # Observation: truck(x,y,load,energy) + assigned (dist,fill) + nearest 3 bins (d,fill)x3
        self.obs_dim = 6 + 3 * 2
        self.action_space = spaces.Discrete(5)  # 0=move,1=pickup,2=drop,3=recharge,4=wait
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.obs_dim,), dtype=np.float32)

    def reset(self):
        self.city = City(self.cfg)
        self.cfg["plan_route_fn"] = self.city.plan_route
        self.sim = Simulation(self.cfg, self.city)
        self.current_step = 0
        return self._get_obs_all()

    def step(self, actions):
        """
        actions: list[int] of length n_agents
        Returns: (obs_all, rewards_all, done_all, info)
        """
        dt = self.cfg["DT"]
        rewards = [0.0] * self.n_agents

        # 1) Bins fill (stochastic demand) + detect overflows
        lo, hi = self.cfg["BIN_FILL_PER_STEP"]
        rnd = self.sim._rnd()
        overflowed_ids = []
        for b in self.sim.bins:
            before = b.fill
            if b.step_fill(lo, hi, rnd) and before < b.capacity:
                overflowed_ids.append(b.id)
                self.sim.events.append({"t": self.sim.t, "type": "overflow", "bin": b.id})

        # 2) Auction (coordination) assigns targets/routes
        auction(self.sim.bins, self.sim.trucks, self.sim.t, self.cfg, self.city.plan_route)

        # 3) RL actions
        for idx, truck in enumerate(self.sim.trucks):
            a = actions[idx]
            r = truck.apply_action(a, self.sim.bins, self.city.depot, self.cfg)
            rewards[idx] += r

        # 4) Charge overflow penalty to responsible trucks
        if overflowed_ids:
            pen = self.cfg["OVERFLOW_PENALTY_EUR"]
            for bid in overflowed_ids:
                owners = [i for i, t in enumerate(self.sim.trucks) if t.assigned_bin == bid]
                if owners:
                    for i in owners:
                        rewards[i] -= pen
                else:
                    # nearest truck pays (difference-reward light)
                    bpos = next(b.pos for b in self.sim.bins if b.id == bid)
                    i_star = min(
                        range(self.n_agents),
                        key=lambda i: math.hypot(self.sim.trucks[i].pos[0] - bpos[0],
                                                 self.sim.trucks[i].pos[1] - bpos[1])
                    )
                    rewards[i_star] -= pen
                self.sim.day_costs["penalties_eur"] += pen

        # 5) Wage accounting for day costs; advance time
        self.sim._wage_tick()
        self.sim.t += dt
        self.current_step += 1

        obs = self._get_obs_all()
        done = self.current_step >= self.max_steps
        dones = [done] * self.n_agents
        info = {"costs": self.sim.summary_costs()}
        return obs, rewards, dones, info

    def _get_obs_all(self):
        return [self._get_obs(tr) for tr in self.sim.trucks]

    def _get_obs(self, truck):
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

        return np.array([
            x / self.cfg["MAP_SIZE"][0],
            y / self.cfg["MAP_SIZE"][1],
            load, energy,
            assigned_d, assigned_fill
        ] + b_feats, dtype=np.float32)
