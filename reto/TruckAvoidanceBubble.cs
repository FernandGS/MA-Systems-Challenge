using UnityEngine;

// Lightweight avoidance: apply a tiny lateral nudge when neighbors are too close
[RequireComponent(typeof(Transform))]
public class TruckAvoidanceBubble : MonoBehaviour
{
    [Tooltip("Radius within which other trucks cause a nudge (world units)")]
    public float radius = 1.2f;
    [Tooltip("Max lateral nudge per frame (world units)")]
    public float maxNudge = 0.15f;

    static readonly int TruckLayer = -1; // optional: assign a Truck layer and filter via LayerMask if desired

    void LateUpdate()
    {
        var myPos = transform.position;
        // Find nearby trucks (simple: scan all with same component). For large counts, use a spatial partition.
        var neighbors = FindObjectsOfType<TruckAvoidanceBubble>();
        Vector3 accum = Vector3.zero;
        int count = 0;
        foreach (var n in neighbors)
        {
            if (n == this) continue;
            var delta = myPos - n.transform.position;
            delta.y = 0f;
            float d2 = delta.sqrMagnitude;
            if (d2 < radius * radius && d2 > 1e-6f)
            {
                accum += delta.normalized / Mathf.Max(0.3f, Mathf.Sqrt(d2));
                count++;
            }
        }
        if (count > 0)
        {
            var move = accum.normalized * Mathf.Min(maxNudge, accum.magnitude * 0.05f);
            transform.position += new Vector3(move.x, 0f, move.z);
        }
    }
}
