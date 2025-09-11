using System;
using System.Collections.Generic;
using System.IO;
using System.Text; // For Encoding & JSON body build
using UnityEngine;
using UnityEngine.Networking; // Added for HTTP requests

[Serializable] public class GridInfo { public int width; public int height; public int[] depot; }
[Serializable] public class Point { public int x; public int y; }

[Serializable] public class LaneDebugEntry { public int i; public float base_x; public float base_y; public float dx; public float dy; public float perp_x; public float perp_y; public float off; public float side_sign; public float ox; public float oy; public float final_x; public float final_y; }
[Serializable] public class FloatPoint { public float x; public float y; }
[Serializable]
public class AgentRun
{
    public int id;
    public int[] start;     // [gx, gy]
    public Point[] pathObj; // integer snapped path
    public FloatPoint[] floatPath; // high-res per-frame positions (lane already applied)
    public FloatPoint[] rawPlannedPath; // original planned waypoints pre-lane
    public LaneDebugEntry[] laneDebug; // lane diagnostics
    public int distance;
    public int collected;
    public int capacity;
    // Optional dwell ticks per compressed node (schemaVersion>=2)
    public int[] pathDwell;
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
    public int schemaVersion; // 1 = legacy (no dwell), 2 = dwell enabled
    public bool hasDwell;     // convenience boolean
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
    [Tooltip("Base lane offset (fallback if axis-specific not set)")] public float laneOffset = 0.6f;
    [Tooltip("Lane offset when moving horizontally (along +X/-X). If <=0, uses laneOffset.")] public float horizontalLaneOffset = 0f;
    [Tooltip("Lane offset when moving vertically (along +Z/-Z). If <=0, uses laneOffset.")] public float verticalLaneOffset = 0f;
    [Tooltip("If true and horizontalLaneOffset <= 0 but verticalLaneOffset is large, derive a proportional horizontal offset so horizontals still separate.")]
    public bool adaptHorizontalIfUnset = true;
    [Tooltip("Factor (0-1) of verticalLaneOffset used when deriving a horizontal offset (only if horizontalLaneOffset & laneOffset are 0).")]
    [Range(0f,1f)] public float horizontalAdaptFactor = 0.5f;
    [Tooltip("Minimum horizontal offset when auto-derived.")] public float minHorizontalAdapt = 0.25f;
    [Header("Lane Clamping (prevents sidewalk spill)")]
    [Tooltip("Clamp horizontal offsets to this maximum (world units). Set <=0 to disable for that axis.")]
    public float horizontalMaxOffset = 0f;
    [Tooltip("Clamp vertical offsets to this maximum (world units). Set <=0 to disable for that axis.")]
    public float verticalMaxOffset = 0f;
    [Tooltip("If true, applies axis max clamps above.")]
    public bool clampOffsets = true;
    public enum TrafficSide { Right, Left }
    public TrafficSide trafficSide = TrafficSide.Right; // Keep-right or keep-left

    [Header("Directional Split (horizontal only)")]
    [Tooltip("If true, when effective horizontal lane offset computes to 0 we still split left/right by direction so opposing flows don't overlap.")]
    public bool splitHorizontalWhenZero = true;
    [Tooltip("Magnitude of half-split applied when horizontal lane offset is 0 (world units). Positive value; applied + for +X travel, - for -X travel before trafficSide inversion.")]
    public float horizontalZeroSplit = 0.5f;

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
    // Dwell playback state
    private readonly Dictionary<int, int> dwellIndex = new();      // which compressed node index is active
    private readonly Dictionary<int, int> dwellRemaining = new();  // ticks left to stay on current node
    public bool useDwellIfAvailable = true;                        // master toggle

    [Header("Centering Adjustments (pre-lane offset)")]
    [Tooltip("If true, we first center positions exactly on the cell middle each frame before lane offset math.")] public bool forceCellCenter = true;
    [Tooltip("Explicit horizontal (X travel) centering correction along perpendicular (Z) axis in world units. Applies BEFORE lane offset. Useful if art grid misaligned.")] public float horizontalCenterCorrection = 0f;
    [Tooltip("Explicit vertical (Z travel) centering correction along perpendicular (X) axis in world units. Applies BEFORE lane offset.")] public float verticalCenterCorrection = 0f;
    [Tooltip("If true, prints one-time calibration suggestions for center corrections when simulation starts.")] public bool autoSuggestCentering = true;

    [Header("Debug / Gizmos")]
    public bool drawLaneGizmos = true;
    public Color gizmoCenterColor = Color.yellow;
    public Color gizmoLaneColor = Color.cyan;
    public float gizmoSize = 0.25f;

    [Header("Global Horizontal Baseline Shift")]
    [Tooltip("Extra shift (world units) applied to ALL horizontal street positions (after lane offset / floatPath). Negative moves toward -Z (\"down\" in your screenshot).")]
    public float horizontalBaselineShift = 0f;

    [Header("Intersection Control (virtual traffic lights)")]
    [Tooltip("Enable intersection right-of-way logic so perpendicular movers don't overlap visually.")]
    public bool enableIntersectionControl = true;
    [Tooltip("If true, Unity will ignore its own intersection control and trust Python-exported positions (recommended).")]
    public bool usePythonIntersection = true;
    public enum IntersectionMode { AlternateAxis, HorizontalPriority, VerticalPriority, LowIdPriority }
    [Tooltip("AlternateAxis toggles each step; others give fixed priority.")]
    public IntersectionMode intersectionMode = IntersectionMode.AlternateAxis;
    [Tooltip("If >0, only one mover may enter an intersection cell per step (recommended stay 1). Future use for multi-step clearance.")]
    public int intersectionClearanceSteps = 1;
    [Tooltip("Log intersection arbitration decisions.")]
    public bool logIntersectionDecisions = false;
    private bool allowHorizontalThisCycle = true; // flips when AlternateAxis mode

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
                Quaternion binRotation = Quaternion.Euler(0, 0, -90);
                Quaternion binPosiion = Quaternion.Euler(0, 0.6f, 0);
                var go = Instantiate(BinPrefab, bp, Quaternion.Euler(0f,0f,-90f));
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
                if (a.floatPath != null && a.floatPath.Length > 0)
                {
                    startPos = new Vector3(worldOrigin.x + a.floatPath[0].x * cellSize, worldOrigin.y, worldOrigin.z + a.floatPath[0].y * cellSize);
                }
                else if (applyLaneOffsets)
                {
                    startPos = ApplyLaneOffset(a, 0, startPos);
                }
                var go = Instantiate(TruckPrefab, startPos, Quaternion.identity);
                go.name = $"Truck_{a.id}";
                // Add lightweight avoidance bubble component
                if (go.GetComponent<TruckAvoidanceBubble>() == null)
                    go.AddComponent<TruckAvoidanceBubble>();
                trucks[a.id] = go;

                if (a.pathObj == null || a.pathObj.Length == 0)
                {
                    Debug.LogWarning($"Agente {a.id} sin pathObj");
                    continue;
                }
                totalSteps = Mathf.Max(totalSteps, a.pathObj.Length - 1);
                Debug.Log($"Agente {a.id} → pasos en pathObj: {a.pathObj.Length}");
                if (data.schemaVersion >= 2 && useDwellIfAvailable && a.pathDwell != null && a.pathDwell.Length > 0)
                {
                    dwellIndex[a.id] = 0;
                    dwellRemaining[a.id] = a.pathDwell[0];
                }
            }
        }

        if (data.metrics != null)
        {
            Debug.Log($"[KPIs] Collected={data.metrics.total_collected}, AvgDist={data.metrics.avg_distance_per_agent:F1}, Msgs={data.metrics.negotiation_messages}, Steps={data.metrics.steps}");
        }
        dataReady = true;
        // One-time horizontal / vertical center diagnostics
        if (trucks.Count > 0)
        {
            int sample = 0;
            foreach (var kv in trucks)
            {
                if (sample++ > 3) break;
                var go = kv.Value; if (go==null) continue;
                var agentTmp = data.agents[kv.Key];
                int fCount = (agentTmp.floatPath != null) ? agentTmp.floatPath.Length : 0;
                Debug.Log($"[CenterDiag] Truck {kv.Key} start worldPos={go.transform.position} horizCorr={horizontalCenterCorrection} vertCorr={verticalCenterCorrection} laneOffset={laneOffset} hLane={horizontalLaneOffset} vLane={verticalLaneOffset} floatPathLen={fCount}");
                if (agentTmp.laneDebug != null && agentTmp.laneDebug.Length > 0)
                {
                    var ld0 = agentTmp.laneDebug[0];
                    Debug.Log($"[LaneDbg] Truck {kv.Key} first i={ld0.i} base=({ld0.base_x},{ld0.base_y}) final=({ld0.final_x},{ld0.final_y}) perp=({ld0.perp_x},{ld0.perp_y}) off={ld0.off} side={ld0.side_sign}");
                }
            }
            // Auto-calibration suggestion for horizontal centering: detect bias between first two different horizontal floatPath Y values vs grid center
            if (autoSuggestCentering)
            {
                float accumBias = 0f; int biasSamples = 0;
                foreach (var a in data.agents)
                {
                    if (a.floatPath == null || a.floatPath.Length < 3) continue;
                    for (int i=1;i<a.floatPath.Length;i++)
                    {
                        var prev = a.floatPath[i-1]; var cur = a.floatPath[i];
                        float dx = cur.x - prev.x; float dy = cur.y - prev.y;
                        if (Mathf.Abs(dx) > Mathf.Abs(dy) && Mathf.Abs(dx) > 1e-3f) // horizontal segment
                        {
                            // Expect dy ~= previous.y (stays constant). Compute remainder to nearest 0.5 cell for center detection.
                            float centerLine = Mathf.Round(prev.y) + 0.0f; // assume integer center; adjust formula if visual center is integer + 0.5
                            float bias = (prev.y - centerLine) * cellSize; // world units bias
                            if (Mathf.Abs(bias) > 1e-3f)
                            {
                                accumBias += bias; biasSamples++;
                            }
                            break;
                        }
                    }
                }
                if (biasSamples > 0)
                {
                    float avgBias = accumBias / biasSamples;
                    if (Mathf.Abs(avgBias) > 0.01f)
                    {
                        Debug.Log($"[CenterSuggest] Detected average horizontal bias {avgBias:F3} world units; consider setting horizontalCenterCorrection={-avgBias:F3} to compensate.");
                    }
                }
            }
        }
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
        // If dwell active we manage advancement differently
        bool dwellMode = useDwellIfAvailable && data != null && data.schemaVersion >= 2 && data.hasDwell;
        if (!dwellMode)
            currentStep = Mathf.Min(currentStep + 1, totalSteps);
        // First pass: compute desired target cell for each agent
        var desiredCell = new Dictionary<int, Vector2Int>(data.agents.Length);
        var cellCounts = new Dictionary<Vector2Int, int>();
        foreach (var a in data.agents)
        {
            if (a.pathObj == null || a.pathObj.Length == 0) continue;
            int idx;
            if (dwellMode && dwellIndex.TryGetValue(a.id, out var dIdx)) idx = Mathf.Clamp(dIdx, 0, a.pathObj.Length - 1);
            else idx = Mathf.Min(currentStep, a.pathObj.Length - 1);
            var p = a.pathObj[idx];
            var cell = new Vector2Int(p.x, p.y);
            desiredCell[a.id] = cell;
            if (cellCounts.TryGetValue(cell, out var c)) cellCounts[cell] = c + 1; else cellCounts[cell] = 1;
        }

        // Detect head-on swaps: A at cell X -> Y while B at cell Y -> X in the same step
        var lastCellByAgent = new Dictionary<int, Vector2Int>(data.agents.Length);
        foreach (var a in data.agents)
        {
            if (!trucks.TryGetValue(a.id, out var go)) continue;
            // read current grid cell by snapping current world position
            var wp = go.transform.position - worldOrigin;
            int gx = Mathf.RoundToInt(wp.x / Mathf.Max(1e-6f, cellSize));
            int gy = Mathf.RoundToInt(wp.z / Mathf.Max(1e-6f, cellSize));
            lastCellByAgent[a.id] = new Vector2Int(gx, gy);
        }

    // Intersection arbitration (pre-hold) ---------------------------------
    var holdFlags = new Dictionary<int, bool>();
    // If dwell mode is active and we trust Python exported timing (usePythonIntersection),
    // skip Unity-side arbitration to avoid double-holding. Python already encoded waits
    // as extended dwell counts in pathDwell compression.
    if (enableIntersectionControl && !(dwellMode && usePythonIntersection))
        {
            if (intersectionMode == IntersectionMode.AlternateAxis)
            {
                allowHorizontalThisCycle = !allowHorizontalThisCycle; // flip each step
            }

            // Build movement plans
            var plansByTarget = new Dictionary<Vector2Int, List<(int id, Vector2Int from, Vector2Int to, bool horizontal)>>();
            foreach (var a in data.agents)
            {
                if (a.pathObj == null || a.pathObj.Length < 2) continue;
                if (!desiredCell.TryGetValue(a.id, out var toCell)) continue;
                if (!lastCellByAgent.TryGetValue(a.id, out var fromCell)) continue;
                if (toCell == fromCell) continue; // not moving
                // Determine orientation of movement
                bool horizontal = Mathf.Abs(toCell.x - fromCell.x) >= Mathf.Abs(toCell.y - fromCell.y);
                if (!plansByTarget.TryGetValue(toCell, out var list))
                {
                    list = new List<(int, Vector2Int, Vector2Int, bool)>();
                    plansByTarget[toCell] = list;
                }
                list.Add((a.id, fromCell, toCell, horizontal));
            }

            foreach (var kv in plansByTarget)
            {
                var list = kv.Value;
                if (list.Count <= 1) continue; // single entrant fine
                // Determine if we have perpendicular contention
                bool anyH = false, anyV = false;
                foreach (var pl in list) { if (pl.horizontal) anyH = true; else anyV = true; }
                // Decide winner set
                List<int> allowed = new List<int>();
                if (anyH && anyV)
                {
                    switch (intersectionMode)
                    {
                        case IntersectionMode.AlternateAxis:
                            if (allowHorizontalThisCycle) { foreach (var pl in list) if (pl.horizontal) allowed.Add(pl.id); }
                            else { foreach (var pl in list) if (!pl.horizontal) allowed.Add(pl.id); }
                            break;
                        case IntersectionMode.HorizontalPriority:
                            foreach (var pl in list) if (pl.horizontal) allowed.Add(pl.id);
                            break;
                        case IntersectionMode.VerticalPriority:
                            foreach (var pl in list) if (!pl.horizontal) allowed.Add(pl.id);
                            break;
                        case IntersectionMode.LowIdPriority:
                            int minId = int.MaxValue; int minOrient = 0; // orient: 1 horizontal, 0 vertical just informational
                            foreach (var pl in list) if (pl.id < minId) { minId = pl.id; minOrient = pl.horizontal ? 1 : 0; }
                            allowed.Add(minId);
                            break;
                    }
                }
                else
                {
                    // Same orientation conflict (multiple approaching same cell). Allow one based on mode (LowId or first) to avoid all holding.
                    switch (intersectionMode)
                    {
                        case IntersectionMode.LowIdPriority:
                            int minId = int.MaxValue; foreach (var pl in list) if (pl.id < minId) minId = pl.id; allowed.Add(minId); break;
                        default:
                            // Allow lowest id anyway
                            int lowest = int.MaxValue; foreach (var pl in list) if (pl.id < lowest) lowest = pl.id; allowed.Add(lowest); break;
                    }
                }
                // If allowed empty (e.g., AlternateAxis but no movers of chosen axis), fallback to lowest id
                if (allowed.Count == 0)
                {
                    int lowest = int.MaxValue; foreach (var pl in list) if (pl.id < lowest) lowest = pl.id; allowed.Add(lowest);
                }
                // Mark holds for disallowed
                foreach (var pl in list) if (!allowed.Contains(pl.id)) holdFlags[pl.id] = true;
                if (logIntersectionDecisions)
                {
                    string desc = $"[Intersection] Cell {kv.Key} allow=[{string.Join(",", allowed)}] hold=[";
                    List<int> denied = new List<int>(); foreach (var pl in list) if (!allowed.Contains(pl.id)) denied.Add(pl.id);
                    desc += string.Join(",", denied) + "] mode=" + intersectionMode + (intersectionMode==IntersectionMode.AlternateAxis? (allowHorizontalThisCycle?"(H)":"(V)") : "");
                    Debug.Log(desc);
                }
            }
        }

        // Second pass: move or hold based on occupancy, intersection arbitration, and head-on swap rule
        foreach (var a in data.agents)
        {
            if (!trucks.TryGetValue(a.id, out var go)) continue;
            if (a.pathObj == null || a.pathObj.Length == 0) continue;
            int idx;
            if (dwellMode && dwellIndex.TryGetValue(a.id, out var dIdx)) idx = Mathf.Clamp(dIdx, 0, a.pathObj.Length - 1); else idx = Mathf.Min(currentStep, a.pathObj.Length - 1);

            // If this target cell is occupied by multiple agents, hold position for one frame
            bool hold = false;
            if (desiredCell.TryGetValue(a.id, out var cell))
            {
                if (cellCounts.TryGetValue(cell, out var cnt) && cnt > 1)
                    hold = true; // provisional; may be overridden by arbitration
            }

            if (holdFlags.TryGetValue(a.id, out var forcedHold) && forcedHold)
            {
                hold = true; // intersection decision overrides
            }

            // Head-on swap detection: if someone else is aiming for my last cell while I aim for theirs,
            // hold the higher id to break the tie deterministically.
            if (!hold && desiredCell.TryGetValue(a.id, out var myNext))
            {
                if (lastCellByAgent.TryGetValue(a.id, out var myLast))
                {
                    foreach (var kv in desiredCell)
                    {
                        int otherId = kv.Key; if (otherId == a.id) continue;
                        var otherNext = kv.Value;
                        if (!lastCellByAgent.TryGetValue(otherId, out var otherLast)) continue;
                        if (myNext == otherLast && otherNext == myLast && a.id > otherId)
                        {
                            hold = true; break;
                        }
                    }
                }
            }

            Vector3 target = GridToWorld(a.pathObj[idx].x, a.pathObj[idx].y);
            bool alreadyLane = false;
            if (a.floatPath != null && idx < a.floatPath.Length)
            {
                var fp = a.floatPath[idx];
                target = new Vector3(worldOrigin.x + fp.x * cellSize, worldOrigin.y, worldOrigin.z + fp.y * cellSize);
                alreadyLane = true;
            }
            if (applyLaneOffsets && !alreadyLane)
                target = ApplyLaneOffset(a, idx, target);

            // Universal horizontal baseline shift: applies even when using floatPath (alreadyLane==true)
            if (Mathf.Abs(horizontalBaselineShift) > 1e-4f)
            {
                bool isHorizontal = false;
                // Derive direction
                if (a.floatPath != null && a.floatPath.Length > 1 && idx < a.floatPath.Length)
                {
                    int prevI = Mathf.Max(0, idx - 1);
                    var fPrev = a.floatPath[prevI];
                    var fCur = a.floatPath[idx];
                    float dx = fCur.x - fPrev.x;
                    float dy = fCur.y - fPrev.y;
                    isHorizontal = Mathf.Abs(dx) >= Mathf.Abs(dy);
                }
                else if (a.pathObj != null && a.pathObj.Length > 1)
                {
                    int prevI = Mathf.Max(0, idx - 1);
                    var pPrev = a.pathObj[prevI];
                    var pCur = a.pathObj[idx];
                    float dx = pCur.x - pPrev.x;
                    float dz = pCur.y - pPrev.y; // grid y -> world z
                    isHorizontal = Mathf.Abs(dx) >= Mathf.Abs(dz);
                }
                if (isHorizontal)
                {
                    target += new Vector3(0f, 0f, horizontalBaselineShift);
                }
            }

            Vector3 startPos = go.transform.position;
            Vector3 moveDir = target - startPos;
            Quaternion desiredRot = go.transform.rotation;
            if (moveDir.sqrMagnitude > 1e-6f)
            {
                Vector3 flatDir = new Vector3(moveDir.x, 0f, moveDir.z);
                if (flatDir.sqrMagnitude > 1e-6f)
                    desiredRot = Quaternion.LookRotation(flatDir.normalized, Vector3.up);
            }

            if (hold)
            {
                // Hold overrides dwell timer consumption (don't decrement this frame)
                continue;
            }

            if (smoothLerp) StartCoroutine(LerpTo(go.transform, target, stepDuration, desiredRot));
            else { go.transform.position = target; go.transform.rotation = desiredRot; }

            if (dwellMode && dwellIndex.TryGetValue(a.id, out var curIdx) && a.pathDwell != null && curIdx < a.pathDwell.Length)
            {
                // Consume one dwell tick AFTER moving to the node (so initial full dwell applies)
                if (dwellRemaining.TryGetValue(a.id, out var remain))
                {
                    remain -= 1;
                    if (remain <= 0)
                    {
                        // Advance to next node if available
                        int nextIdx = curIdx + 1;
                        if (nextIdx < a.pathObj.Length && nextIdx < a.pathDwell.Length)
                        {
                            dwellIndex[a.id] = nextIdx;
                            dwellRemaining[a.id] = a.pathDwell[nextIdx];
                        }
                        else
                        {
                            dwellIndex[a.id] = curIdx; // stay at last
                            dwellRemaining[a.id] = int.MaxValue; // effectively infinite dwell at end
                        }
                    }
                    else
                    {
                        dwellRemaining[a.id] = remain;
                    }
                }
            }
        }

        if (dwellMode)
        {
            // Recompute synthetic totalSteps for progress (max dwellIndex)
            int maxIdx = 0;
            foreach (var kv in dwellIndex) if (kv.Value > maxIdx) maxIdx = kv.Value;
            currentStep = maxIdx; // for metrics / gizmos
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
        // Early center snapping (handles horizontal mis-centering) BEFORE computing direction
        if (forceCellCenter)
        {
            // Snap X and Z to exact grid center (worldOrigin + gx*cellSize + 0? we assume basePos already at center)
            // If art grid is shifted half a unit, user can add manual corrections below.
        }
        // Manual centering corrections (independent of lane offset). These shift perpendicular axis to fix base alignment.
        // Determine direction first to know which perpendicular axis to apply.
        if (a.pathObj == null || a.pathObj.Length < 2) return basePos + Vector3.zero; // ensure copy
        int i1 = Mathf.Clamp(idx, 1, a.pathObj.Length - 1);
        int i0 = i1 - 1;
        var p0 = a.pathObj[i0];
        var p1 = a.pathObj[i1];
        Vector3 w0 = GridToWorld(p0.x, p0.y);
        Vector3 w1 = GridToWorld(p1.x, p1.y);
        Vector3 dir = (w1 - w0); dir.y = 0f;
        if (dir.sqrMagnitude < 1e-6f) return basePos;
        dir.Normalize();
        Vector3 rightPerp = new Vector3(dir.z, 0f, -dir.x).normalized;
        bool horizontal = Mathf.Abs(dir.x) >= Mathf.Abs(dir.z);

        // Apply pre-lane centering correction (only along perpendicular axis):
        if (horizontal && Mathf.Abs(horizontalCenterCorrection) > 1e-4f)
        {
            basePos += new Vector3(0f, 0f, horizontalCenterCorrection);
        }
        else if (!horizontal && Mathf.Abs(verticalCenterCorrection) > 1e-4f)
        {
            basePos += new Vector3(verticalCenterCorrection, 0f, 0f);
        }

        if (!applyLaneOffsets) return basePos;

        float axisOff;
        if (horizontal)
        {
            axisOff = horizontalLaneOffset > 0f ? horizontalLaneOffset : laneOffset;
            // Previous logic required laneOffset > 0 to adapt; that prevented horizontal separation when only verticalLaneOffset was set.
            // New logic: if user left horizontalLaneOffset & laneOffset at 0 but provided a verticalLaneOffset, derive a proportional horizontal offset.
            if (adaptHorizontalIfUnset && horizontalLaneOffset <= 0f && axisOff <= 0f && verticalLaneOffset > 0.05f)
            {
                float derived = verticalLaneOffset * Mathf.Clamp01(horizontalAdaptFactor);
                derived = Mathf.Max(derived, minHorizontalAdapt);
                axisOff = Mathf.Min(derived, verticalLaneOffset); // never exceed vertical magnitude
            }
        }
        else
        {
            axisOff = verticalLaneOffset > 0f ? verticalLaneOffset : laneOffset;
        }
        if (axisOff <= 0f)
        {
            // Optional directional split: even with zero configured offset we separate by travel direction
            if (horizontal && splitHorizontalWhenZero && Mathf.Abs(horizontalZeroSplit) > 1e-3f)
            {
                float dirSign = Mathf.Sign(dir.x); // +1 for +X, -1 for -X
                float trafficSign = (trafficSide == TrafficSide.Right) ? 1f : -1f; // maintain side inversion for left-side traffic
                return basePos + rightPerp * (horizontalZeroSplit * dirSign * trafficSign);
            }
            return basePos; // stay centered after corrections
        }
        if (clampOffsets)
        {
            if (horizontal && horizontalMaxOffset > 0f) axisOff = Mathf.Min(axisOff, horizontalMaxOffset);
            if (!horizontal && verticalMaxOffset > 0f) axisOff = Mathf.Min(axisOff, verticalMaxOffset);
        }
        float sideSign = (trafficSide == TrafficSide.Right) ? 1f : -1f;
        return basePos + rightPerp * (axisOff * sideSign);
    }

    void OnDrawGizmosSelected()
    {
        if (!drawLaneGizmos || trucks == null) return;
        Gizmos.color = gizmoCenterColor;
        foreach (var kv in trucks)
        {
            var go = kv.Value; if (go == null) continue;
            Vector3 pos = go.transform.position;
            Gizmos.DrawWireSphere(pos, gizmoSize * 0.5f);
            // Attempt to show lane offset direction arrow
            if (data != null && data.agents != null)
            {
                AgentRun ar = null;
                foreach (var a in data.agents) if (a.id == kv.Key) { ar = a; break; }
                if (ar != null && ar.pathObj != null && ar.pathObj.Length > 1 && currentStep < ar.pathObj.Length)
                {
                    int idx = Mathf.Min(currentStep, ar.pathObj.Length - 1);
                    int i1 = Mathf.Clamp(idx, 1, ar.pathObj.Length - 1); int i0 = i1 - 1;
                    var p0 = ar.pathObj[i0]; var p1 = ar.pathObj[i1];
                    Vector3 w0 = GridToWorld(p0.x, p0.y); Vector3 w1 = GridToWorld(p1.x, p1.y);
                    Vector3 dir = (w1 - w0); dir.y = 0f; if (dir.sqrMagnitude < 1e-4f) continue; dir.Normalize();
                    Vector3 rightPerp = new Vector3(dir.z, 0f, -dir.x).normalized;
                    Gizmos.color = gizmoLaneColor;
                    Gizmos.DrawLine(pos, pos + rightPerp * gizmoSize * 1.75f);
                }
            }
        }
    }
}