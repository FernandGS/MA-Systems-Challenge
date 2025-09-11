#!/usr/bin/env python3
"""Simulation JSON analytics utility.

Reads a Unity-exported (or server /simulate) JSON payload (schemaVersion 1 or 2)
and produces rich metrics about efficiency, movement, dwell, service cadence,
utilization, and event rates.

Outputs (stdout human summary) + optional JSON/CSV export.

Usage:
  python -m analytics.analyze_sim --input saved_payload.json
  python -m analytics.analyze_sim --input sim_run_pathObj.json --json out_metrics.json --csv out_agent_metrics.csv

Key Metrics:
    Global:
    - total_steps
    - n_agents, n_bins
    - total_collected, avg_distance_per_agent
    - pickups, dumps, overflows, recharges
    - pickup_rate (pickups / steps)
    - avg_interpickup_steps
    - intersection_wait_events, intersection_wait_rate
    - uturn_blocks
    - avg_agent_utilization (time moving / steps)
    - avg_dwell_fraction (dwell ticks / steps) (schema v2)
        - est_energy_used (distance * ENERGY_COST_PER_M if present in payload cfg, else distance)
        - energy_per_unit_collected (est_energy_used / total_collected if collected>0)
        - mean_bin_latency (average time from last service to next overflow or end)
        - mean_service_cycle (average inter-service interval across bins)
  Per-Agent:
    - distance, collected
    - path_len (compressed) & dwell_len
    - moves (distinct position advances)
    - dwell_ticks, move_ticks
    - utilization (move_ticks / total_steps_seen)
    - mean_dwell_at_node
    - pickups, dumps, overflows handled
    - first_pickup_step, last_pickup_step, mean_interpickup

Limitations: dwell-based timing only approximate when using schemaVersion 1 (no pathDwell).
"""
from __future__ import annotations
import argparse, json, math, statistics as stats, csv, sys, glob, os, pathlib
from typing import Dict, List, Any, Tuple

EventTypeMap = {
    'ASSIGN': 'assign',
    'SERVICE': 'pickup',
    'DUMP': 'drop',
    'RECHARGE': 'recharge',
    'OVERFLOW': 'overflow',
    'INTERSECTION_WAIT': 'intersection_wait',
    'UTURN_BLOCK': 'uturn_block'
}


def load(path: str) -> Dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def is_close(a: float, b: float, tol: float = 1e-9) -> bool:
    return abs(a - b) <= tol


def compute_agent_metrics(data: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any], List[Dict[str, Any]]]:
    agents = data.get('agents', []) or []
    events = data.get('events', []) or []
    metrics = data.get('metrics', {}) or {}
    schema_version = data.get('schemaVersion', 1)
    total_steps = int(metrics.get('steps', 0))

    # Index events per agent
    ev_by_agent: Dict[int, List[Dict[str, Any]]] = {}
    for ev in events:
        a = ev.get('agent')
        if a is None:
            continue
        ev_by_agent.setdefault(a, []).append(ev)

    per_agent: List[Dict[str, Any]] = []

    total_move_ticks = 0
    total_dwell_ticks = 0
    total_pickups = 0
    total_assigns = 0
    total_dumps = 0
    total_overflows = 0
    total_recharges = 0
    total_waits = 0
    total_uturn_blocks = 0

    for a in agents:
        aid = a.get('id')
        path = a.get('pathObj') or []
        dwell = a.get('pathDwell') if schema_version >= 2 else None
        distance = int(a.get('distance', 0))
        collected = int(a.get('collected', 0))
        capacity = int(a.get('capacity', 0)) or None

        compressed_len = len(path)
        dwell_len = len(dwell) if dwell else 0
        # Derive ticks seen
        if dwell and compressed_len == dwell_len and dwell_len > 0:
            ticks = sum(int(x) for x in dwell)
            dwell_ticks = sum(int(x) - 1 for x in dwell if int(x) > 0)  # staying beyond first tick at node
            moves = compressed_len - 1
            move_ticks = moves  # first occupancy of each node counts as move advancement
        else:
            # legacy assumption: one tick per element
            ticks = max(0, compressed_len - 1)
            dwell_ticks = 0
            moves = ticks
            move_ticks = ticks
        total_move_ticks += move_ticks
        total_dwell_ticks += dwell_ticks

        # Event counts for this agent
        a_events = ev_by_agent.get(aid, [])
        pickups = sum(1 for e in a_events if e.get('type') == 'SERVICE')
        dumps = sum(1 for e in a_events if e.get('type') == 'DUMP')
        assigns = sum(1 for e in a_events if e.get('type') == 'ASSIGN')
        overflows = sum(1 for e in a_events if e.get('type') == 'OVERFLOW')
        recharges = sum(1 for e in a_events if e.get('type') == 'RECHARGE')
        waits = sum(1 for e in a_events if e.get('type') == 'INTERSECTION_WAIT')
        uturn_blocks = sum(1 for e in a_events if e.get('type') == 'UTURN_BLOCK')
        total_pickups += pickups
        total_dumps += dumps
        total_assigns += assigns
        total_overflows += overflows
        total_recharges += recharges
        total_waits += waits
        total_uturn_blocks += uturn_blocks

        pickup_steps = sorted(e.get('t') for e in a_events if e.get('type') == 'SERVICE')
        inter_pickup = []
        for i in range(1, len(pickup_steps)):
            inter_pickup.append(pickup_steps[i] - pickup_steps[i-1])
        first_pickup = pickup_steps[0] if pickup_steps else None
        last_pickup = pickup_steps[-1] if pickup_steps else None
        mean_interpickup = stats.mean(inter_pickup) if inter_pickup else None

        utilization = (move_ticks / ticks) if ticks > 0 else 0.0
        mean_dwell_at_node = (dwell_ticks / compressed_len) if (compressed_len > 0 and dwell_ticks > 0) else 0.0

        per_agent.append({
            'agent': aid,
            'distance': distance,
            'collected': collected,
            'capacity': capacity,
            'compressed_path_len': compressed_len,
            'dwell_nodes': dwell_len,
            'ticks_observed': ticks,
            'moves': moves,
            'move_ticks': move_ticks,
            'dwell_ticks': dwell_ticks,
            'utilization': utilization,
            'mean_dwell_at_node': mean_dwell_at_node,
            'pickups': pickups,
            'dumps': dumps,
            'assigns': assigns,
            'overflows': overflows,
            'recharges': recharges,
            'intersection_wait_events': waits,
            'uturn_blocks': uturn_blocks,
            'first_pickup_step': first_pickup,
            'last_pickup_step': last_pickup,
            'mean_interpickup_steps': mean_interpickup,
        })

    # OVERFLOW events are only present in the global events list (not per-agent), so recompute here
    total_overflows = sum(1 for e in events if isinstance(e, dict) and e.get('type') == 'OVERFLOW')

    global_metrics = {
        'total_steps': total_steps,
        'n_agents': len(agents),
        'n_bins': len(data.get('bins', []) or []),
        'total_collected': int(metrics.get('total_collected', 0)),
        'avg_distance_per_agent_exported': metrics.get('avg_distance_per_agent'),
        'sum_agent_distance': sum(a.get('distance', 0) for a in agents),
        'total_pickups_events': total_pickups,
        'total_assign_events': total_assigns,
        'total_dump_events': total_dumps,
        'total_overflow_events': total_overflows,
        'total_recharge_events': total_recharges,
        'intersection_wait_events': total_waits,
        'uturn_block_events': total_uturn_blocks,
    }

    if total_steps > 0:
        global_metrics.update({
            'pickup_rate_per_step': total_pickups / total_steps,
            'assign_rate_per_step': total_assigns / total_steps,
            'intersection_wait_rate_per_step': total_waits / total_steps,
        })

    # Aggregate utilization
    if per_agent:
        global_metrics['avg_agent_utilization'] = stats.mean(a['utilization'] for a in per_agent)
        global_metrics['avg_agent_mean_dwell_at_node'] = stats.mean(a['mean_dwell_at_node'] for a in per_agent)

    # Inter-pickup global stats
    all_inter_pick = [a['mean_interpickup_steps'] for a in per_agent if a['mean_interpickup_steps'] is not None]
    if all_inter_pick:
        global_metrics['avg_mean_interpickup_steps'] = stats.mean(all_inter_pick)

    # Dwell fraction
    if total_steps > 0:
        global_metrics['fleet_move_tick_fraction_est'] = total_move_ticks / (total_move_ticks + total_dwell_ticks) if (total_move_ticks + total_dwell_ticks) > 0 else 1.0
        global_metrics['fleet_dwell_tick_fraction_est'] = total_dwell_ticks / (total_move_ticks + total_dwell_ticks) if (total_move_ticks + total_dwell_ticks) > 0 else 0.0

    # Energy efficiency approximation (requires optional config embedded or fallback)
    cfg = data.get('cfg', {}) or {}
    energy_cost_per_m = cfg.get('ENERGY_COST_PER_M') or cfg.get('ENERGY_COST_PER_STEP') or 1.0
    est_energy_used = global_metrics['sum_agent_distance'] * float(energy_cost_per_m)
    global_metrics['est_energy_used'] = est_energy_used
    if global_metrics.get('total_collected', 0) > 0:
        global_metrics['energy_per_unit_collected'] = est_energy_used / global_metrics['total_collected']

    # Per-bin service cycles & latency
    bins = data.get('bins', []) or []
    bin_events: Dict[int, Dict[str, Any]] = {b.get('id'): {'services': [], 'overflows': []} for b in bins}
    for ev in events:
        et = ev.get('type')
        bid = ev.get('bin')
        if bid is None: continue
        if et == 'SERVICE':
            bin_events.setdefault(bid, {'services': [], 'overflows': []})['services'].append(ev.get('t', 0))
        elif et == 'OVERFLOW':
            bin_events.setdefault(bid, {'services': [], 'overflows': []})['overflows'].append(ev.get('t', 0))

    bin_metrics: List[Dict[str, Any]] = []
    service_intervals = []
    latency_samples = []
    for bid, rec in bin_events.items():
        sv = sorted(rec['services'])
        ov = sorted(rec['overflows'])
        inter = []
        for i in range(1, len(sv)):
            gap = sv[i] - sv[i-1]
            if gap > 0:
                inter.append(gap)
                service_intervals.append(gap)
        # Latency: time from service to next overflow; if no overflow after a service, use (total_steps - service_step)
        lat_list = []
        for s in sv:
            next_over = next((o for o in ov if o > s), None)
            if next_over is not None:
                lat = next_over - s
            else:
                lat = (total_steps - s) if total_steps and total_steps > s else None
            if lat is not None and lat >= 0:
                lat_list.append(lat)
                latency_samples.append(lat)
        bin_metrics.append({
            'bin': bid,
            'services': len(sv),
            'overflows': len(ov),
            'mean_service_interval': stats.mean(inter) if inter else None,
            'median_service_interval': stats.median(inter) if inter else None,
            'mean_latency_to_overflow': stats.mean(lat_list) if lat_list else None,
            'median_latency_to_overflow': stats.median(lat_list) if lat_list else None,
        })

    if service_intervals:
        global_metrics['mean_service_cycle'] = stats.mean(service_intervals)
        global_metrics['median_service_cycle'] = stats.median(service_intervals)
    if latency_samples:
        global_metrics['mean_bin_latency'] = stats.mean(latency_samples)
        global_metrics['median_bin_latency'] = stats.median(latency_samples)

    return per_agent, global_metrics, bin_metrics


def summarize(per_agent: List[Dict[str, Any]], global_metrics: Dict[str, Any]) -> str:
    lines = []
    g = global_metrics
    lines.append("=== Global Metrics ===")
    for k in sorted(g.keys()):
        lines.append(f"{k}: {g[k]}")
    lines.append("")
    lines.append("=== Per-Agent (topline) ===")
    header = f"agent  dist  collected  util  moves  dwell_ticks  pickups  waits"
    lines.append(header)
    for a in sorted(per_agent, key=lambda x: x['agent']):
        lines.append(
            f"{a['agent']:>5}  {a['distance']:>5}  {a['collected']:>9}  {a['utilization']*100:5.1f}%  {a['moves']:>5}  {a['dwell_ticks']:>10}  {a['pickups']:>7}  {a['intersection_wait_events']:>5}"
        )
    return "\n".join(lines)


def write_json(path: str, per_agent: List[Dict[str, Any]], global_metrics: Dict[str, Any], bin_metrics: List[Dict[str, Any]]):
    out = {
        'global': global_metrics,
        'agents': per_agent,
        'bins': bin_metrics,
    }
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)


def write_csv(path: str, per_agent: List[Dict[str, Any]]):
    if not per_agent:
        return
    keys = list(per_agent[0].keys())
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for row in per_agent:
            w.writerow(row)


def write_bin_csv(path: str, bin_metrics: List[Dict[str, Any]]):
    if not bin_metrics:
        return
    keys = list(bin_metrics[0].keys())
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for row in bin_metrics:
            w.writerow(row)

def find_latest(search_dir: str, pattern: str) -> str | None:
    base = pathlib.Path(search_dir)
    if not base.exists():
        return None
    files = list(base.glob(pattern))
    if not files:
        return None
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return str(files[0])

def main():
    ap = argparse.ArgumentParser(description="Analyze simulation JSON metrics")
    ap.add_argument('--input', help='Path to simulation JSON payload (optional if --latest used)')
    ap.add_argument('--latest', action='store_true', help='Auto-pick most recent JSON (default dir saved_payloads)')
    ap.add_argument('--search-dir', default='saved_payloads', help='Directory to search when using --latest')
    ap.add_argument('--pattern', default='*.json', help='Filename glob pattern for --latest')
    ap.add_argument('--json', help='Optional output metrics JSON path')
    ap.add_argument('--csv', help='Optional per-agent metrics CSV path')
    ap.add_argument('--bin-csv', help='Optional per-bin metrics CSV path')
    args = ap.parse_args()

    target = args.input
    if args.latest:
        target = find_latest(args.search_dir, args.pattern)
        if not target:
            print(f"[error] no files matched pattern {args.pattern} in {args.search_dir}", file=sys.stderr)
            sys.exit(2)
        else:
            print(f"[auto-selected latest] {target}")
    if not target:
        print('[error] must provide --input or use --latest', file=sys.stderr)
        sys.exit(2)
    data = load(target)
    per_agent, global_metrics, bin_metrics = compute_agent_metrics(data)
    print(summarize(per_agent, global_metrics))

    if args.json:
        write_json(args.json, per_agent, global_metrics, bin_metrics)
        print(f"[wrote metrics json] {args.json}")
    if args.csv:
        write_csv(args.csv, per_agent)
        print(f"[wrote agent csv] {args.csv}")
    if args.bin_csv:
        write_bin_csv(args.bin_csv, bin_metrics)
        print(f"[wrote bin csv] {args.bin_csv}")

if __name__ == '__main__':
    main()
