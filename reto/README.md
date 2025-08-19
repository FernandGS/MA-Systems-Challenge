# Waste Collection Simulation + Unity Viewer

This repo contains:
- `agen_simulation.py` – AgentPy simulation logic (batch/offline run)
- `server.py` – FastAPI server exposing a `/simulate` endpoint that returns Unity‑ready JSON
- `unity.cs` – Unity MonoBehaviour (`SimulationPlayer_PathObj`) to visualize simulation (local JSON file or remote HTTP)

## 1. Python Server (Windows PowerShell)
From the project folder (`reto/`):

```powershell
# (Optional) Create / activate venv if not already
python -m venv .venv
& .\.venv\Scripts\Activate.ps1

# Install dependencies
pip install fastapi uvicorn agentpy

# Run server
python server.py
```
You should see: `Uvicorn running on http://0.0.0.0:8000`.

### Test endpoints
Open a browser:
```
http://127.0.0.1:8000/health
```
Interactive docs:
```
http://127.0.0.1:8000/docs
```
Example POST (PowerShell):
```powershell
Invoke-RestMethod -Uri http://127.0.0.1:8000/simulate -Method Post -ContentType 'application/json' -Body '{}'
```

### Override parameters
Send JSON body like:
```json
{
  "seed": 42,
  "num_agents": 5,
  "num_waste_locations": 20,
  "truck_speed": 1.3,
  "return_speed_factor": 1.6
}
```
If you omit `num_agents` or `num_waste_locations`, the server randomizes them (>=3 / >=10).

## 2. Unity Setup
1. Copy `unity.cs` into your Unity `Assets/Scripts` folder (or keep existing). The class name is `SimulationPlayer_PathObj`.
2. Create prefabs:
   - `TruckPrefab` (e.g., a colored Cube)
   - `BinPrefab` (another Cube or cylinder)
3. Create an empty GameObject in the scene, name it `SimulationPlayer` and attach the script.
4. In Inspector set:
   - `Use Remote` = true (to fetch from Python server)
   - `Remote Url` = `http://127.0.0.1:8000/simulate`
   - Optionally set `Seed Override`, `Num Agents Override`, `Num Bins Override` (0 / -1 means let server decide)
   - Adjust `Cell Size` (grid -> world scaling) & `Step Duration` (animation speed)
5. Enter Play Mode. The script POSTs to the server, receives JSON, spawns bins & trucks, then animates paths.

### Local (offline) playback
If you want to play back a previously generated run:
1. Save JSON returned by `/simulate` as `sim_run_pathObj.json`.
2. Place it in `Assets/StreamingAssets/` folder (create if missing).
3. Uncheck `Use Remote` and keep `jsonFileName = sim_run_pathObj.json`.
4. Press Play.

## 3. Troubleshooting
| Issue | Cause | Fix |
|-------|-------|-----|
| Unity logs `Remote fetch failed` | Server not running / wrong URL | Start server, verify `/health` works, update URL |
| PowerShell cannot import FastAPI | Venv not activated | `& .\.venv\Scripts\Activate.ps1` then `pip install fastapi uvicorn agentpy` |
| CORS error in logs (WebGL) | Different host/port | CORS already allows `*`; ensure no proxy blocking |
| Empty scene (no trucks) | JSON parse failed | Check Unity Console for parse error & view raw response in browser |
| Slow animation | Very long path (many steps) | Increase `stepDuration` or reduce grid size / parameters |

## 4. JSON Shape (from server)
```
{
  "grid": {"width":100,"height":100,"depot":[0,0]},
  "agents": [ {"id":0,"start":[x,y],"pathObj":[{"x":..,"y":..},...],"distance":float,"collected":int,"capacity":int}, ... ],
  "bins": [ {"id":0,"pos":[x,y],"initial":int,"remaining":int}, ... ],
  "events": [ {"t":step,"type":"SERVICE","agent":id,"bin":id,"amount":q}, ... ],
  "metrics": {"total_collected":int, "avg_distance_per_agent":float, "negotiation_messages":int, "steps":int }
}
```

## 5. Extending
- Add a WebSocket endpoint for streaming step-by-step instead of full path array.
- Visual indicators in Unity (e.g., bin fill level via scale / color).
- UI panel to send new simulation parameter overrides without exiting Play Mode.

## 6. Quick Run Summary
```powershell
# From repo root
& .\.venv\Scripts\Activate.ps1
python server.py  # keep running
```
Then Play in Unity with `Use Remote = true`.

---
Feel free to request a WebSocket live mode or incremental streaming next.
