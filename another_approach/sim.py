# sim.py
# Core simulation loop: bins fill, negotiation assigns tasks, trucks execute,
# costs are tracked, frames/events are logged, and JSON export is supported.

from typing import List, Dict
import random, math, json
from agents import Truck, BinObj
from negotiation import auction

class Simulation:
    def __init__(self, cfg, city):
        self.cfg = cfg
        self.city = city
        self.t = 0.0

        # --- Bins ---
        self.bins: List[BinObj] = [
            BinObj(b["id"], b["pos"], b["capacity"], b["fill"], b.get("curb")) for b in city.bins
        ]

        # --- Trucks ---
        self.trucks: List[Truck] = []
        for i in range(cfg["N_TRUCKS"]):
            spawn = city.depot  # start at depot (jitter optional)
            t = Truck(
                tid=f"T{i}",
                pos=spawn,
                cfg=cfg,
                energy=cfg["ENERGY_MAX"]
            )
            self.trucks.append(t)

        # --- Logs ---
        self.frames: List[Dict] = []
        self.events: List[Dict] = []
        self.day_costs = {
            "wage_eur": 0.0,
            "energy_eur": 0.0,
            "maintenance_eur": 0.0,
            "penalties_eur": 0.0,
        }

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------
    def _rnd(self):
        return random.Random(int(self.t) ^ self.cfg["SEED"])

    def _fill_bins(self):
        lo, hi = self.cfg["BIN_FILL_PER_STEP"]
        rnd = self._rnd()
        overflows = 0
        for b in self.bins:
            before = b.fill
            if b.step_fill(lo, hi, rnd) and before < b.capacity:
                # new overflow
                overflows += 1
                self.events.append({"t": self.t, "type": "overflow", "bin": b.id})
        return overflows

    def _wage_tick(self):
        dt_hours = self.cfg["DT"] / 3600.0
        self.day_costs["wage_eur"] += len(self.trucks) * self.cfg["WAGE_PER_HOUR"] * dt_hours

    # -------------------------------------------------------------------------
    # Step
    # -------------------------------------------------------------------------
    def step(self):
        dt = self.cfg["DT"]

        # 1. Bins fill
        new_ov = self._fill_bins()
        if new_ov > 0:
            self.day_costs["penalties_eur"] += new_ov * self.cfg["OVERFLOW_PENALTY_EUR"]

        # 2. Negotiation (auction assigns tasks)
        auction(self.bins, self.trucks, self.t, self.cfg, self.city.plan_route)


        # 3. Trucks act
        step_events = []
        for t in self.trucks:
            evs = t.step(dt, self.bins, self.city.depot, self.city.plan_route)
            step_events.extend(evs)
        self.events.extend(step_events)

        # 4. Wage accumulation
        self._wage_tick()

        # 5. Costs aggregation
        self.day_costs["energy_eur"] = sum(t.costs_eur["energy"] for t in self.trucks)
        self.day_costs["maintenance_eur"] = sum(t.costs_eur["maint"] for t in self.trucks)

        # 6. Log frame for replay/export
        frame = {
            "t": self.t,
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
                for t in self.trucks
            ],
            "bins": [
                {"id": b.id, "x": b.pos[0], "y": b.pos[1], "fill": b.fill, "cap": b.capacity}
                for b in self.bins
            ],
            "events": step_events,
        }
        self.frames.append(frame)

        # Advance time
        self.t += dt

    # -------------------------------------------------------------------------
    # Run loop
    # -------------------------------------------------------------------------
    def run(self, steps:int):
        for _ in range(steps):
            self.step()

    # -------------------------------------------------------------------------
    # Costs + Export
    # -------------------------------------------------------------------------
    def summary_costs(self)->Dict:
        total = sum(self.day_costs.values())
        return {**self.day_costs, "total_eur": total}

    def export_json(self, path:str):
        """Export replay (frames + events + costs) for Unity or dashboards."""
        out = {
            "frames": self.frames,
            "events": self.events,
            "costs": self.summary_costs(),
            "cfg": self.cfg,
        }
        with open(path,"w") as f:
            json.dump(out,f,indent=2)
        print(f"✅ Exported simulation JSON to {path}")
