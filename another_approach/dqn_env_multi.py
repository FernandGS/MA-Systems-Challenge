# dqn_env_multi.py
# Multi-agent Gym-like wrapper for Simulation.
# Each truck is its own agent.

import numpy as np
from sim import Simulation
from city import City
from gymnasium import spaces

class MultiTruckEnv:
    def __init__(self, cfg):
        self.cfg = cfg
        self.city = City(cfg)
        self.sim = Simulation(cfg, self.city)

        self.n_agents = cfg["N_TRUCKS"]
        self.max_steps = cfg["STEPS_PER_DAY"]
        self.current_step = 0

        # Each truck has the same obs/action space
        self.obs_dim = 4 + 3*2   # truck state + nearest 3 bins
        self.action_space = spaces.Discrete(5)  # 0=move,1=pickup,2=drop,3=recharge,4=wait
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.obs_dim,), dtype=np.float32)

    def reset(self):
        self.city = City(self.cfg)
        self.sim = Simulation(self.cfg, self.city)
        self.current_step = 0
        return self._get_obs_all()

    def step(self, actions):
        """
        actions: list[int] of length n_agents
        Returns: (obs_all, rewards_all, done_all, info)
        """
        rewards = []
        for idx, truck in enumerate(self.sim.trucks):
            a = actions[idx]
            r = truck.apply_action(a, self.sim.bins, self.city.depot, self.cfg)
            rewards.append(r)

        self.sim._wage_tick()
        self.sim.t += self.cfg["DT"]
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

        bins = sorted(
            self.sim.bins, key=lambda b: np.hypot(b.pos[0]-x, b.pos[1]-y)
        )[:3]
        b_feats = []
        for b in bins:
            d = np.hypot(b.pos[0]-x, b.pos[1]-y) / max(1, self.cfg["MAP_SIZE"][0])
            f = b.fill / b.capacity
            b_feats += [d, f]
        while len(b_feats) < 6:
            b_feats.append(0.0)

        return np.array([x/self.cfg["MAP_SIZE"][0],
                         y/self.cfg["MAP_SIZE"][1],
                         load, energy] + b_feats, dtype=np.float32)
