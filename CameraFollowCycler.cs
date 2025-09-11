using System.Collections.Generic;
using UnityEngine;
#if ENABLE_INPUT_SYSTEM
using UnityEngine.InputSystem;
#endif

public class CameraFollowCycler : MonoBehaviour
{
    [Header("Targeting")]
    public string truckTag = "Truck"; // tag your trucks
    public List<Transform> targets = new List<Transform>(); // optional: fill from code
    public bool autoDiscover = true; // find by tag if targets empty


    [Header("Follow")]
    public Vector3 offset = new Vector3(0f, 20f, -24f);
    public float followSmooth = 5f;           // position smoothing
    public float lookSmooth = 5f;             // rotation smoothing

    [Header("Cycling")]
    public bool autoCycle = true;
    public float switchInterval = 5f;

    [Header("Dynamic")]
    public bool autoRefresh = true;          // rediscover at runtime
    public float refreshInterval = 3f;       // seconds between refreshes

    private int _i = 0;
    private float _t = 0f;
    private Vector3 _vel = Vector3.zero;
    private float _refreshT = 0f;

    void LateUpdate()
    {
        EnsureTargets();

        if (targets.Count == 0) return;
        if (_i >= targets.Count) _i = 0;

    // Manual controls (supports both old and new Input Systems)
    if (PressedNext()) Next();
    if (PressedPrev()) Prev();
    if (PressedToggle()) autoCycle = !autoCycle;

        // Auto cycle
        if (autoCycle)
        {
            _t += Time.deltaTime;
            if (_t >= switchInterval) { _t = 0f; Next(); }
        }

    // Periodic refresh for dynamic spawn/despawn
    RefreshTargetsIfNeeded();

    var t = targets.Count > 0 ? targets[_i] : null;
        if (t == null) { CullNulls(); return; }

        // Desired camera position (world offset; adjust to taste)
        Vector3 desired = t.position + offset;

        // Smooth follow
        transform.position = Vector3.SmoothDamp(transform.position, desired, ref _vel, 1f / Mathf.Max(0.001f, followSmooth));

        // Smooth look-at
        var dir = (t.position - transform.position);
        if (dir.sqrMagnitude > 0.001f)
        {
            var targetRot = Quaternion.LookRotation(dir.normalized, Vector3.up);
            transform.rotation = Quaternion.Slerp(transform.rotation, targetRot, lookSmooth * Time.deltaTime);
        }
    }

    // -------- Input helpers --------
    private bool PressedNext()
    {
#if ENABLE_INPUT_SYSTEM
    var kb = Keyboard.current;
    return kb != null && kb.rightBracketKey.wasPressedThisFrame;
#else
    return Input.GetKeyDown(KeyCode.RightBracket);
#endif
    }

    private bool PressedPrev()
    {
#if ENABLE_INPUT_SYSTEM
    var kb = Keyboard.current;
    return kb != null && kb.leftBracketKey.wasPressedThisFrame;
#else
    return Input.GetKeyDown(KeyCode.LeftBracket);
#endif
    }

    private bool PressedToggle()
    {
#if ENABLE_INPUT_SYSTEM
    var kb = Keyboard.current;
    return kb != null && kb.spaceKey.wasPressedThisFrame;
#else
    return Input.GetKeyDown(KeyCode.Space);
#endif
    }

    public void SetTargets(List<Transform> list)
    {
        targets = list ?? new List<Transform>();
        CullNulls();
        _i = Mathf.Clamp(_i, 0, Mathf.Max(0, targets.Count - 1));
    }

    public void Next()
    {
        if (targets.Count == 0) return;
        _i = (_i + 1) % targets.Count;
    }

    public void Prev()
    {
        if (targets.Count == 0) return;
        _i = (_i - 1 + targets.Count) % targets.Count;
    }

    private void EnsureTargets()
    {
        if (autoDiscover && targets.Count == 0)
            DiscoverByTag();
    }

    private void CullNulls()
    {
        targets.RemoveAll(t => t == null);
        if (_i >= targets.Count) _i = 0;
    }

    private void DiscoverByTag()
    {
        var trucks = GameObject.FindGameObjectsWithTag(truckTag);
        targets.Clear();
        foreach (var go in trucks) if (go != null) targets.Add(go.transform);
        CullNulls();
    }

    private void RefreshTargetsIfNeeded()
    {
        if (!autoDiscover) return;
        // fast path: if any null slipped in (destroyed), refresh now
        for (int k = 0; k < targets.Count; k++)
        {
            if (targets[k] == null) { DiscoverByTag(); return; }
        }
        if (!autoRefresh) return;
        _refreshT += Time.deltaTime;
        if (_refreshT >= refreshInterval)
        {
            _refreshT = 0f;
            DiscoverByTag();
        }
    }
}