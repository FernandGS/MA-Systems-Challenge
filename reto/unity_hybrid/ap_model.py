try:
    import agentpy as ap
except Exception:  
    ap = None

from typing import Any, List
from city import City
from sim import Simulation
from dispatch import auction


if ap:
    class TruckAgent(ap.Agent):  # type: ignore[misc]
        def setup(self):  # type: ignore[override]
            pass

        def step(self):  # type: ignore[override]
            m = self.model  # WasteSimModel
            dt = m.cfg["DT"]
            # Delegate movement/service to underlying Truck
            for ev in self.truck.step(dt, m.sim.bins, m.city.depot, m.sim._plan_route):
                ev["t"] = m.sim.t
                m.sim.events.append(ev)

    class BinAgent(ap.Agent):  # type: ignore[misc]
        def setup(self):  # type: ignore[override]
            # self.bin: agents.BinObj is injected by model
            pass
else:
    TruckAgent = object  # type: ignore
    BinAgent = object  # type: ignore


class WasteSimModel(ap.Model if ap else object):  # type: ignore[misc]

    def setup(self):  # type: ignore[override]
        p = getattr(self, 'p', None)
        if p is None:
            raise RuntimeError("WasteSimModel requires agentpy Parameters with cfg/steps/planner")
        self.cfg = dict(p.cfg)  # copy
        self.city = City(self.cfg)
        # Initialize Simulation to use its data structures & helpers, but we won't call sim.step()
        self.sim = Simulation(cfg=self.cfg, city=self.city, planner=p.planner)
        # Create agents wrapping existing domain objects
        if ap:
            self.trucks = ap.AgentList(self, len(self.sim.trucks), TruckAgent)  # type: ignore[attr-defined]
            for a, t in zip(self.trucks, self.sim.trucks):
                a.truck = t
            self.bins = ap.AgentList(self, len(self.sim.bins), BinAgent)  # type: ignore[attr-defined]
            for a, b in zip(self.bins, self.sim.bins):
                a.bin = b
        else:
            self.trucks = []  # type: ignore[assignment]
            self.bins = []    # type: ignore[assignment]
        # Tracking for data collection
        self._ev_len_prev = 0
        if ap and hasattr(self, 'record'):
            self.record("events_total", 0)

    def step(self):  # type: ignore[override]
        # 1) Bin fill
        self.sim._fill_bins()
        # 2) Auction assignments
        assigns = auction(self.sim.bins, self.sim.trucks, self.sim.t, self.cfg, self.sim._plan_route)
        for ev in assigns:
            ev["t"] = self.sim.t
            self.sim.events.append(ev)
        # 3) Step trucks via agentpy scheduler (delegates to underlying Truck)
        if ap:
            self.trucks.step()  # type: ignore[operator]
        else:
            # Fallback step if agentpy not present (shouldn't happen when using this model)
            for t in self.sim.trucks:
                for ev in t.step(self.cfg["DT"], self.sim.bins, self.city.depot, self.sim._plan_route):
                    ev["t"] = self.sim.t
                    self.sim.events.append(ev)
        # 4) Log frame
        fr = {
            "t": self.sim.t,
            "trucks": [
                {
                    "id": t.tid, "x": t.pos[0], "y": t.pos[1],
                    "energy": t.energy, "load": t.load, "state": t.state,
                    "target": (None if t.target is None else {"x": t.target[0], "y": t.target[1]}),
                } for t in self.sim.trucks
            ],
            "bins": [
                {"id": b.id, "x": b.pos[0], "y": b.pos[1], "fill": b.fill, "cap": b.capacity}
                for b in self.sim.bins
            ],
        }
        self.sim.frames.append(fr)
        # 5) Advance time
        self.sim.t += self.cfg["DT"]
        # 6) Data collection
        if ap and hasattr(self, 'record'):
            self.record("events_total", len(self.sim.events))

    def end(self):  # type: ignore[override]
        # Report basic KPIs
        total_collected = sum(e.get("amount", 0) for e in self.sim.events if e.get("type") == "pickup")
        if ap and hasattr(self, 'report'):
            self.report({  # type: ignore[attr-defined]
                "total_events": len(self.sim.events),
                "total_collected": int(total_collected),
                "steps": int(len(self.sim.frames)),
            })
