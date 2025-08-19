"""FastAPI server to run the waste collection simulation and return
Unity-compatible JSON.

Run:
  pip install fastapi uvicorn agentpy
  python server.py

Endpoint:
  POST /simulate  (JSON body with optional parameters overrides)

Returned JSON Format expected by existing Unity script (extended):
{
  "grid": {"width": int, "height": int, "depot": [x,y]},
  "agents": [
       {"id": int, "start": [x,y], "pathObj": [{"x":int,"y":int},...],
        "distance": float, "collected": int, "capacity": int}
  ],
  "bins": [ {"id": int, "pos": [x,y], "initial": int, "remaining": int} ],
  "events": [ {"t": step, "type": str, "agent": id| -1, "bin": id| -1, "amount": int } ],
  "metrics": {"total_collected": int, "avg_distance_per_agent": float,
              "negotiation_messages": int, "steps": int}
}
"""

from fastapi import FastAPI, Body, Query
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import random
from typing import Dict, Any

from agen_simulation import WasteModel  # noqa: E402

app = FastAPI(title="Waste Simulation Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


DEFAULT_PARAMS = {
    'num_agents': 5,
    'num_waste_locations': 15,
    'grid_size': 100,
    'agent_capacity': 20,
    'bin_waste_min': 5,
    'bin_waste_max': 15,
    'steps': 100000,
    # Optional extended params supported by model (seed, truck_speed, return_speed_factor)
    'seed': None,
    'truck_speed': 1.0,
    'return_speed_factor': 1.2,
}


def _convert_model_to_unity_json(model: WasteModel) -> Dict[str, Any]:
    log = model.simulation_log
    summary = log.get('summary_metrics', {})

    # Build per-agent collected amounts from SERVICE events
    collected_map = {a.id: 0 for a in model.trucks}
    for ev in log['events']:
        if ev['type'] == 'SERVICE' and ev.get('agent_id') is not None:
            collected_map[ev['agent_id']] += ev.get('amount', 0)

    agents_json = []
    for a in model.trucks:
        path_key = f"agent_{a.id}"
        path_list = log['agent_paths_obj'].get(path_key, [])
        # Round positions to int grid coordinates for Unity
        int_path = [{"x": int(round(p['x'])), "y": int(round(p['y']))} for p in path_list]
        start = int_path[0] if int_path else {"x": int(round(a.pos[0])), "y": int(round(a.pos[1]))}
        agents_json.append({
            "id": a.id,
            "start": [start['x'], start['y']],
            "pathObj": int_path,
            "distance": float(summary.get('agent_travel_distances', {}).get(path_key, 0.0)),
            "collected": collected_map.get(a.id, 0),
            "capacity": a.capacity,
        })

    bins_json = []
    for b in model.bins:
        bins_json.append({
            "id": b.id,
            "pos": [int(round(b.pos[0])), int(round(b.pos[1]))],
            "initial": b.initial,
            "remaining": b.remaining,
        })

    # Translate events
    events_json = []
    for ev in log['events']:
        events_json.append({
            "t": ev.get('step', 0),
            "type": ev.get('type'),
            "agent": ev.get('agent_id') if ev.get('agent_id') is not None else -1,
            "bin": ev.get('waste_id') if ev.get('waste_id') is not None else -1,
            "amount": ev.get('amount', 0),
        })

    distances = list(summary.get('agent_travel_distances', {}).values())
    avg_distance = sum(distances) / len(distances) if distances else 0.0

    metrics = {
        "total_collected": int(summary.get('total_waste_collected', 0)),
        "avg_distance_per_agent": avg_distance,
        "negotiation_messages": int(summary.get('negotiation_messages', 0)),
        "steps": int(summary.get('step_all_dumped', 0)),
    }

    grid = {
        "width": model.p.grid_size,
        "height": model.p.grid_size,
        "depot": [0, 0],
    }

    return {
        "grid": grid,
        "agents": agents_json,
        "bins": bins_json,
        "events": events_json,
        "metrics": metrics,
    }


@app.post("/simulate")
def simulate(custom: Dict[str, Any] = Body(default=None)) -> Dict[str, Any]:
    """Run a simulation and return Unity-compatible JSON.

    Body JSON can override any default parameter. If num_agents/num_waste_locations
    are omitted they will be randomized within min constraints (>=3 / >=10).
    """
    params = DEFAULT_PARAMS.copy()
    if custom:
        params.update({k: v for k, v in custom.items() if v is not None})

    # If user didn't explicitly set these, randomize with constraints
    if not custom or 'num_agents' not in custom or custom.get('num_agents') is None:
        params['num_agents'] = random.randint(3, 7)
    if not custom or 'num_waste_locations' not in custom or custom.get('num_waste_locations') is None:
        params['num_waste_locations'] = random.randint(10, 30)

    model = WasteModel(params)
    model.run()
    payload = _convert_model_to_unity_json(model)
    return payload


@app.get("/simulate")
def simulate_get(
        seed: int | None = Query(default=None),
        num_agents: int | None = Query(default=None, ge=1),
        num_waste_locations: int | None = Query(default=None, ge=1),
        truck_speed: float | None = Query(default=None, gt=0),
        return_speed_factor: float | None = Query(default=None, gt=0),
):
        """Convenience GET endpoint for quick browser testing.

        Example:
            http://127.0.0.1:8000/simulate?seed=42&num_agents=4&num_waste_locations=15

        Any omitted param falls back to the same randomization rules as POST.
        """
        custom = {}
        if seed is not None: custom['seed'] = seed
        if num_agents is not None: custom['num_agents'] = num_agents
        if num_waste_locations is not None: custom['num_waste_locations'] = num_waste_locations
        if truck_speed is not None: custom['truck_speed'] = truck_speed
        if return_speed_factor is not None: custom['return_speed_factor'] = return_speed_factor
        return simulate(custom if custom else None)


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
