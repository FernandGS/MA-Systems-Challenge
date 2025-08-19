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

    // Remote mode configuration
    public bool useRemote = false;                 // Toggle: local file vs HTTP
    public string remoteUrl = "http://127.0.0.1:8000/simulate"; // FastAPI endpoint
    public bool autoRequestOnStart = true;         // Auto fetch on Start
    public int seedOverride = -1;                  // Optional seed
    public int numAgentsOverride = 0;              // 0 means let server randomize
    public int numBinsOverride = 0;                // 0 means let server randomize
    public float truckSpeed = 1f;                  // Optional parameters forwarded
    public float returnSpeedFactor = 1.2f;

    private SimData data;
    private readonly Dictionary<int, GameObject> trucks = new();
    private int currentStep = 0;
    private float accum = 0f;
    private int totalSteps = 0;
    private bool dataReady = false;

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
        InitializeScene();
    }

    System.Collections.IEnumerator FetchRemoteSimulation()
    {
        // Build JSON body manually so we can omit fields to trigger server randomization
        List<string> parts = new();
        if (seedOverride >= 0) parts.Add($"\"seed\":{seedOverride}");
        if (numAgentsOverride > 0) parts.Add($"\"num_agents\":{numAgentsOverride}");
        if (numBinsOverride > 0) parts.Add($"\"num_waste_locations\":{numBinsOverride}");
        // Always include speed params (server has defaults, but we send explicit)
        parts.Add($"\"truck_speed\":{truckSpeed.ToString(System.Globalization.CultureInfo.InvariantCulture)}");
        parts.Add($"\"return_speed_factor\":{returnSpeedFactor.ToString(System.Globalization.CultureInfo.InvariantCulture)}");
        string jsonBody = "{" + string.Join(",", parts) + "}"; // {} if empty
        using UnityWebRequest req = new UnityWebRequest(remoteUrl, UnityWebRequest.kHttpVerbPOST);
        byte[] bodyRaw = Encoding.UTF8.GetBytes(jsonBody);
        req.uploadHandler = new UploadHandlerRaw(bodyRaw);
        req.downloadHandler = new DownloadHandlerBuffer();
        req.SetRequestHeader("Content-Type", "application/json");
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
        InitializeScene();
    }

    void InitializeScene()
    {
        // Bins
        if (data.bins != null)
        {
            foreach (var b in data.bins)
            {
                var go = Instantiate(BinPrefab, GridToWorld(b.pos[0], b.pos[1]), Quaternion.identity);
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

            if (smoothLerp) StartCoroutine(LerpTo(go.transform, target, stepDuration));
            else go.transform.position = target;
        }
    }

    System.Collections.IEnumerator LerpTo(Transform t, Vector3 target, float duration)
    {
        Vector3 start = t.position; float e = 0f;
        while (e < duration)
        {
            e += Time.deltaTime; float k = Mathf.Clamp01(e / duration);
            t.position = Vector3.Lerp(start, target, k);
            yield return null;
        }
        t.position = target;
    }

    Vector3 GridToWorld(int gx, int gy) => new Vector3(gx * cellSize, 0f, gy * cellSize);
}