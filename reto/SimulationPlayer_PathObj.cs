using System;
using System.Collections.Generic;
using System.IO;
using System.Text; // For Encoding & JSON body build
using UnityEngine;
using UnityEngine.Networking; // Added for HTTP requests

[Serializable] public class GridInfo { public int width; public int height; public int[] depot; }
[Serializable] public class Point { public int x; public int y; }

[Serializable]
public class AgentRun
{
    public int id;
    public int[] start;     // [gx, gy]
    public Point[] pathObj; // [{x,y}, ...]
    public int distance;
    public int collected;
    public int capacity;
}

[Serializable]
public class BinInfo
{
    public int id;
    public int[] pos;   // [gx, gy]
    public int initial;
    public int remaining;
}

[Serializable]
public class EventInfo
{
    public int t;
    public string type;
    public int agent;
    public int bin;
    public int amount;
}

[Serializable]
public class MetricsInfo
{
    public int total_collected;
    public float avg_distance_per_agent;
    public int negotiation_messages;
    public int steps;
}

[Serializable]
public class SimData
{
    public GridInfo grid;
    public AgentRun[] agents;
    public BinInfo[] bins;
    public EventInfo[] events;
    public MetricsInfo metrics;
}

public class SimulationPlayer_PathObj : MonoBehaviour
{
    public string jsonFileName = "sim_run_pathObj.json";
    public GameObject TruckPrefab;
    public GameObject BinPrefab;

    public float cellSize = 1f;
    public float stepDuration = 0.1f;
    public bool smoothLerp = true;
    public float rotationSpeed = 720f; // degrees per second for yaw rotation

    // World alignment
    public Vector3 worldOrigin = Vector3.zero;           // Shift grid -> world alignment
    public Vector3 binVisualOffset = Vector3.zero;       // Nudge all bins if they appear off-sidewalks

    // Lane offsets to avoid trucks overlapping in the exact same pixel
    public bool applyLaneOffsets = true;
    public float laneOffset = 0.25f; // in world units; multiplied by +/-1 based on agent id

    // Remote mode configuration
    public bool useRemote = false;                 // Toggle: local file vs HTTP
    public string remoteUrl = "http://127.0.0.1:8000/simulate"; // FastAPI endpoint
    public bool autoRequestOnStart = true;         // Auto fetch on Start
    public bool useGetRequest = false;             // If true, call API with GET & query params; otherwise POST JSON
    public int seedOverride = -1;                  // Optional seed
    public int numAgentsOverride = 0;              // 0 means let server randomize
    public int numBinsOverride = 0;                // 0 means let server randomize
    public int stepsOverride = 0;                  // 0 means use server default
    public string planner = "graph";              // "graph" or "grid"
    public float truckSpeed = 1f;                  // Optional parameters forwarded
    public float returnSpeedFactor = 1.2f;

    private SimData data;
    private readonly Dictionary<int, GameObject> trucks = new();
    private int currentStep = 0;
    private float accum = 0f;
    private int totalSteps = 0;
    private bool dataReady = false;
    // Fallback toggle: if true, we'll rebuild paths from events when incoming paths are static
    public bool rebuildStaticPathsFromEvents = true;

    void Start()
    {
        if (useRemote && autoRequestOnStart)
        {
            StartCoroutine(FetchRemoteSimulation());
        }
        else
        {
            LoadLocalFile();
        }
    }

    void LoadLocalFile()
    {
        string path = Path.Combine(Application.streamingAssetsPath, jsonFileName);
        if (!File.Exists(path)) { Debug.LogError($"File not found: {path}"); return; }
        string raw = File.ReadAllText(path);
        data = JsonUtility.FromJson<SimData>(raw);
        if (data == null) { Debug.LogError("No se pudo parsear el JSON"); return; }
        TryRebuildStaticPaths();
        InitializeScene();
    }

    System.Collections.IEnumerator FetchRemoteSimulation()
    {
        UnityWebRequest req;
        if (useGetRequest)
        {
            // Build query string for GET
            var qp = new List<string>();
            if (seedOverride >= 0) qp.Add("seed=" + seedOverride);
            if (numAgentsOverride > 0) qp.Add("num_agents=" + numAgentsOverride);
            if (numBinsOverride > 0) qp.Add("num_waste_locations=" + numBinsOverride);
            if (stepsOverride > 0) qp.Add("steps=" + stepsOverride);
            if (!string.IsNullOrEmpty(planner)) qp.Add("planner=" + UnityWebRequest.EscapeURL(planner));
            qp.Add("truck_speed=" + truckSpeed.ToString(System.Globalization.CultureInfo.InvariantCulture));
            qp.Add("return_speed_factor=" + returnSpeedFactor.ToString(System.Globalization.CultureInfo.InvariantCulture));
            string url = remoteUrl;
            if (qp.Count > 0) url += (remoteUrl.Contains("?") ? "&" : "?") + string.Join("&", qp);
            req = UnityWebRequest.Get(url);
        }
        else
        {
            // Build JSON body manually so we can omit fields to trigger server randomization
            List<string> parts = new();
            if (seedOverride >= 0) parts.Add($"\"seed\":{seedOverride}");
            if (numAgentsOverride > 0) parts.Add($"\"num_agents\":{numAgentsOverride}");
            if (numBinsOverride > 0) parts.Add($"\"num_waste_locations\":{numBinsOverride}");
            if (stepsOverride > 0) parts.Add($"\"steps\":{stepsOverride}");
            if (!string.IsNullOrEmpty(planner)) parts.Add($"\"planner\":\"{planner}\"");
            // Always include speed params (server has defaults, but we send explicit)
            parts.Add($"\"truck_speed\":{truckSpeed.ToString(System.Globalization.CultureInfo.InvariantCulture)}");
            parts.Add($"\"return_speed_factor\":{returnSpeedFactor.ToString(System.Globalization.CultureInfo.InvariantCulture)}");
            string jsonBody = "{" + string.Join(",", parts) + "}"; // {} if empty
            req = new UnityWebRequest(remoteUrl, UnityWebRequest.kHttpVerbPOST);
            byte[] bodyRaw = Encoding.UTF8.GetBytes(jsonBody);
            req.uploadHandler = new UploadHandlerRaw(bodyRaw);
            req.SetRequestHeader("Content-Type", "application/json");
        }
        req.downloadHandler = new DownloadHandlerBuffer();
        yield return req.SendWebRequest();

        if (req.result != UnityWebRequest.Result.Success)
        {
            Debug.LogError("Remote fetch failed: " + req.error);
            yield break;
        }
        string raw = req.downloadHandler.text;
        data = JsonUtility.FromJson<SimData>(raw);
        if (data == null)
        {
            Debug.LogError("Failed to parse remote JSON");
            yield break;
        }
        TryRebuildStaticPaths();
        InitializeScene();
    }

    void InitializeScene()
    {
        // Bins
        if (data.bins != null)
        {
            foreach (var b in data.bins)
            {
                var bp = GridToWorld(b.pos[0], b.pos[1]) + binVisualOffset;
                var go = Instantiate(BinPrefab, bp, Quaternion.identity);
                go.name = $"Bin_{b.id}";
            }
        }

        // Trucks & paths
        if (data.agents != null)
        {
            foreach (var a in data.agents)
            {
                if (a.start == null || a.start.Length < 2) continue;
                Vector3 startPos = GridToWorld(a.start[0], a.start[1]);
                // apply a lane offset based on first segment if available
                if (applyLaneOffsets)
                {
                    startPos = ApplyLaneOffset(a, 0, startPos);
                }
                var go = Instantiate(TruckPrefab, startPos, Quaternion.identity);
                go.name = $"Truck_{a.id}";
                trucks[a.id] = go;

                if (a.pathObj == null || a.pathObj.Length == 0)
                {
                    Debug.LogWarning($"Agente {a.id} sin pathObj");
                    continue;
                }
                totalSteps = Mathf.Max(totalSteps, a.pathObj.Length - 1);
                Debug.Log($"Agente {a.id} → pasos en pathObj: {a.pathObj.Length}");
            }
        }

        if (data.metrics != null)
        {
            Debug.Log($"[KPIs] Collected={data.metrics.total_collected}, AvgDist={data.metrics.avg_distance_per_agent:F1}, Msgs={data.metrics.negotiation_messages}, Steps={data.metrics.steps}");
        }
        dataReady = true;
    }

    // ==========================
    // Fallback path reconstruction
    // ==========================
    void TryRebuildStaticPaths()
    {
        if (!rebuildStaticPathsFromEvents || data == null || data.agents == null) return;
        int rebuilt = 0, total = 0;
        foreach (var a in data.agents)
        {
            total++;
            if (IsStaticOrInvalidPath(a))
            {
                var rebuiltPath = BuildPathFromEvents(a);
                if (rebuiltPath != null && rebuiltPath.Length > 0)
                {
                    a.pathObj = rebuiltPath;
                    rebuilt++;
                }
            }
        }
        if (rebuilt > 0)
            Debug.Log($"[Fallback] Rebuilt {rebuilt}/{total} agent paths from events to avoid stalls.");
    }

    bool IsStaticOrInvalidPath(AgentRun a)
    {
        // Consider path invalid if null/empty, only repeats same point, or reported distance <= 0
        if (a == null) return false;
        if (a.pathObj == null || a.pathObj.Length == 0) return true;
        if (a.distance <= 0) return true;
        int unique = 0; int lastX = int.MinValue, lastY = int.MinValue;
        HashSet<long> seen = new HashSet<long>();
        foreach (var p in a.pathObj)
        {
            if (p == null) continue;
            long key = ((long)p.x << 32) ^ (uint)p.y;
            if (seen.Add(key)) unique++;
            lastX = p.x; lastY = p.y;
            if (unique > 3) break; // good enough
        }
        // Static if <= 2 unique positions across the whole path
        return unique <= 2;
    }

    Point[] BuildPathFromEvents(AgentRun a)
    {
        if (data == null || data.events == null || data.bins == null) return Array.Empty<Point>();

        // Build bin lookup
        var binById = new Dictionary<int, BinInfo>();
        foreach (var b in data.bins) binById[b.id] = b;

        // Collect targets in chronological order for this agent
        var targets = new List<Vector2Int>();
        // start -> ensure first coordinate is start
        var cur = new Vector2Int(a.start != null && a.start.Length >= 2 ? a.start[0] : 0,
                                 a.start != null && a.start.Length >= 2 ? a.start[1] : 0);

        // Sort events by time t (stable ordering for same t)
        var evs = new List<EventInfo>(data.events);
        evs.Sort((e1, e2) => e1.t.CompareTo(e2.t));

        foreach (var e in evs)
        {
            if (e.agent != a.id) continue;
            if (e.type == "DUMP")
            {
                // go to depot
                if (data.grid != null && data.grid.depot != null && data.grid.depot.Length >= 2)
                {
                    var depot = new Vector2Int(data.grid.depot[0], data.grid.depot[1]);
                    AddTargetIfNew(targets, depot);
                }
            }
            else if (e.type == "ASSIGN" || e.type == "SERVICE")
            {
                if (binById.TryGetValue(e.bin, out var b))
                {
                    var pos = new Vector2Int(b.pos[0], b.pos[1]);
                    AddTargetIfNew(targets, pos);
                }
            }
        }

        // If no targets found, just idle at start but expand to steps
        int limit = (data.metrics != null && data.metrics.steps > 0) ? data.metrics.steps : 100;
        var result = new List<Point>(Mathf.Max(2, limit));
        result.Add(new Point { x = cur.x, y = cur.y });

        if (targets.Count == 0)
        {
            while (result.Count < limit)
                result.Add(new Point { x = cur.x, y = cur.y });
            return result.ToArray();
        }

        // Build a Manhattan path visiting targets in order
        foreach (var tgt in targets)
        {
            AppendManhattan(result, ref cur, tgt, limit);
            if (result.Count >= limit) break;
        }

        // If we still have room, optionally return to depot
        if (result.Count < limit && data.grid != null && data.grid.depot != null && data.grid.depot.Length >= 2)
        {
            var depot = new Vector2Int(data.grid.depot[0], data.grid.depot[1]);
            AppendManhattan(result, ref cur, depot, limit);
        }

        // Pad with last cell if under limit
        while (result.Count < limit)
            result.Add(new Point { x = cur.x, y = cur.y });

        return result.ToArray();
    }

    void AddTargetIfNew(List<Vector2Int> list, Vector2Int pos)
    {
        if (list.Count == 0 || list[list.Count - 1] != pos)
            list.Add(pos);
    }

    void AppendManhattan(List<Point> path, ref Vector2Int cur, Vector2Int tgt, int limit)
    {
        // simple 4-neighbor Manhattan stepping: x first, then y
        while (cur.x != tgt.x && path.Count < limit)
        {
            cur.x += (cur.x < tgt.x) ? 1 : -1;
            path.Add(new Point { x = cur.x, y = cur.y });
        }
        while (cur.y != tgt.y && path.Count < limit)
        {
            cur.y += (cur.y < tgt.y) ? 1 : -1;
            path.Add(new Point { x = cur.x, y = cur.y });
        }
    }

    void Update()
    {
        if (!dataReady || data == null || data.agents == null) return;
        if (totalSteps <= 0) return;

        accum += Time.deltaTime;
        while (accum >= stepDuration)
        {
            StepOnce();
            accum -= stepDuration;
        }
    }

    void StepOnce()
    {
        currentStep = Mathf.Min(currentStep + 1, totalSteps);
        foreach (var a in data.agents)
        {
            if (!trucks.TryGetValue(a.id, out var go)) continue;
            if (a.pathObj == null || a.pathObj.Length == 0) continue;

            int idx = Mathf.Min(currentStep, a.pathObj.Length - 1);
            Vector3 target = GridToWorld(a.pathObj[idx].x, a.pathObj[idx].y);
            if (applyLaneOffsets) target = ApplyLaneOffset(a, idx, target);

            // Determine desired rotation so the truck faces its movement direction
            Vector3 startPos = go.transform.position;
            Vector3 moveDir = target - startPos;
            Quaternion desiredRot = go.transform.rotation;
            if (moveDir.sqrMagnitude > 1e-6f)
            {
                Vector3 flatDir = new Vector3(moveDir.x, 0f, moveDir.z);
                if (flatDir.sqrMagnitude > 1e-6f)
                    desiredRot = Quaternion.LookRotation(flatDir.normalized, Vector3.up);
            }

            if (smoothLerp) StartCoroutine(LerpTo(go.transform, target, stepDuration, desiredRot));
            else
            {
                go.transform.position = target;
                go.transform.rotation = desiredRot;
            }
        }
    }

    System.Collections.IEnumerator LerpTo(Transform t, Vector3 target, float duration, Quaternion targetRot)
    {
        Vector3 start = t.position; float e = 0f;
        while (e < duration)
        {
            e += Time.deltaTime; float k = Mathf.Clamp01(e / duration);
            t.position = Vector3.Lerp(start, target, k);
            // rotate toward targetRot at a limited angular speed to avoid instant flips
            float maxDeg = rotationSpeed * Time.deltaTime;
            t.rotation = Quaternion.RotateTowards(t.rotation, targetRot, maxDeg);
            yield return null;
        }
        t.position = target;
        t.rotation = targetRot;
    }

    Vector3 GridToWorld(int gx, int gy)
    {
        return new Vector3(worldOrigin.x + gx * cellSize, worldOrigin.y, worldOrigin.z + gy * cellSize);
    }

    Vector3 ApplyLaneOffset(AgentRun a, int idx, Vector3 basePos)
    {
        // compute movement direction from previous to next grid point
        if (a.pathObj == null || a.pathObj.Length == 0) return basePos;
        int i0 = Mathf.Clamp(idx - 1, 0, a.pathObj.Length - 1);
        int i1 = Mathf.Clamp(idx, 0, a.pathObj.Length - 1);
        var p0 = a.pathObj[i0];
        var p1 = a.pathObj[i1];
        Vector3 w0 = GridToWorld(p0.x, p0.y);
        Vector3 w1 = GridToWorld(p1.x, p1.y);
        Vector3 dir = (w1 - w0); dir.y = 0f;
        if (dir.sqrMagnitude < 1e-6f) return basePos;
        dir.Normalize();
        // perpendicular to dir on XZ plane
        Vector3 perp = new Vector3(-dir.z, 0f, dir.x);
        int laneIndex = (a.id % 3) - 1; // -1,0,1 pattern across agents
        float offset = laneIndex * laneOffset;
        return basePos + perp * offset;
    }
}