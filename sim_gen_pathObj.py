import json, random

random.seed(7)

W, H = 20, 20
N_AGENTS = 3
N_BINS = 8
AGENT_CAPACITY = 10
BIN_MAX = 12
MAX_STEPS = 300

def step_towards(a, b):
    ax, ay = a; bx, by = b
    if ax < bx: return (ax+1, ay)
    if ax > bx: return (ax-1, ay)
    if ay < by: return (ax, ay+1)
    if ay > by: return (ax, ay-1)
    return a

def random_positions(count, avoid=set()):
    used, out = set(avoid), []
    while len(out) < count:
        p = (random.randint(0, W-1), random.randint(0, H-1))
        if p not in used:
            used.add(p); out.append(p)
    return out

bin_positions = random_positions(N_BINS)
bins = [{
    "id": i, "pos": list(bin_positions[i]),
    "remaining": random.randint(6, BIN_MAX),
    "initial": 0, "locked_by": None
} for i in range(N_BINS)]
for b in bins: b["initial"] = b["remaining"]

agent_positions = random_positions(N_AGENTS, avoid=set(bin_positions))
agents = [{
    "id": i, "pos": list(agent_positions[i]),
    "load": 0, "capacity": AGENT_CAPACITY,
    "target": None, "state": "idle",
    "path": [list(agent_positions[i])],
    "distance": 0, "collected": 0
} for i in range(N_AGENTS)]

DEPOT = (0, 0)
events = []
negotiation_msgs = 0

def all_bins_empty(): return all(b["remaining"] == 0 for b in bins)

def nearest_nonempty_bin(agent_id):
    ax, ay = agents[agent_id]["pos"]
    best, bestd = None, 10**9
    for b in bins:
        if b["remaining"] <= 0: continue
        if b["locked_by"] not in (None, agent_id): continue
        bx, by = b["pos"]
        d = abs(ax-bx) + abs(ay-by)
        if d < bestd: best, bestd = b["id"], d
    return (best, bestd) if best is not None else (None, None)

t = 0
while t < MAX_STEPS and not all_bins_empty():
    for ag in agents:
        if ag["state"] == "idle":
            if ag["load"] >= ag["capacity"]:
                ag["target"] = "depot"; ag["state"] = "to_depot"; continue
            b_id, _ = nearest_nonempty_bin(ag["id"])
            if b_id is not None:
                b = bins[b_id]
                if b["locked_by"] is None:
                    negotiation_msgs += 2
                    b["locked_by"] = ag["id"]
                ag["target"] = b_id; ag["state"] = "to_bin"
                events.append({"t": t, "type": "assign", "agent": ag["id"], "bin": b_id})
    for ag in agents:
        if ag["state"] in ("to_bin", "to_depot"):
            target_pos = DEPOT if ag["target"] == "depot" else tuple(bins[ag["target"]]["pos"])
            cur = tuple(ag["pos"])
            if cur != target_pos:
                nxt = step_towards(cur, target_pos)
                if nxt != cur: ag["pos"] = list(nxt); ag["distance"] += 1
            if tuple(ag["pos"]) == target_pos:
                ag["state"] = "servicing" if ag["state"] == "to_bin" else "dumping"
        if ag["state"] == "servicing":
            b = bins[ag["target"]]
            take = min(b["remaining"], ag["capacity"] - ag["load"])
            if take > 0:
                b["remaining"] -= take
                ag["load"] += take
                ag["collected"] += take
                events.append({"t": t, "type": "service", "agent": ag["id"], "bin": ag["target"], "amount": take})
            bins[ag["target"]]["locked_by"] = None
            if ag["load"] >= ag["capacity"]:
                ag["target"] = "depot"; ag["state"] = "to_depot"
            else:
                ag["state"] = "idle"; ag["target"] = None
        if ag["state"] == "dumping":
            events.append({"t": t, "type": "dump", "agent": ag["id"], "amount": ag["load"]})
            ag["load"] = 0; ag["state"] = "idle"; ag["target"] = None
        ag["path"].append(list(ag["pos"]))
    t += 1

def to_path_obj(path_list): return [{"x": int(p[0]), "y": int(p[1])} for p in path_list]

total_waste_collected = sum(a["collected"] for a in agents)
avg_distance = sum(a["distance"] for a in agents) / len(agents)

output = {
    "grid": {"width": W, "height": H, "depot": list(DEPOT)},
    "agents": [{
        "id": a["id"], "start": a["path"][0],
        "pathObj": to_path_obj(a["path"]),
        "distance": a["distance"], "collected": a["collected"], "capacity": a["capacity"]
    } for a in agents],
    "bins": [{
        "id": b["id"], "pos": b["pos"], "initial": b["initial"], "remaining": b["remaining"]
    } for b in bins],
    "events": events,
    "metrics": {
        "total_collected": total_waste_collected,
        "avg_distance_per_agent": avg_distance,
        "negotiation_messages": negotiation_msgs,
        "steps": t
    }
}

with open("sim_run_pathObj.json", "w") as f:
    json.dump(output, f, indent=2)

print("Wrote sim_run_pathObj.json")
