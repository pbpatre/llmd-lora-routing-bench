# SPDX-License-Identifier: Apache-2.0
"""Per-run analysis: latency per class, placement diagnostics, validity (DESIGN_LLMD_709.md §6-7).

Conventions:
- Primary TTFT = first content token minus *scheduled* time, so client-side
  queueing counts against the system. Secondary TTFT uses the actual send.
- A request that timed out before its first token has TTFT = +inf.
  Percentiles are nearest-rank, so censored requests count. Means are over
  completed requests, reported beside the timeout count.
- Requests belong to windows by arrival (scheduled) time.
- Pod attribution comes from the Envoy access log (upstream host); the
  x-served-by response header is only a cross-check.
- An adapter holds a slot on a pod while it has a running request there:
  [first token - prefill estimate, end], as seen by the client.
"""

from __future__ import annotations

import bisect
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from . import jsonl
from .schedule import BASE

Intervals = list[tuple[float, float]]
Holding = dict[str, dict[str, Intervals]]  # pod -> adapter -> merged slot-holding intervals

SLOW_S = 5.0
LAYOUT_WINDOW_S = 30.0
HOLDING_SHARE = 0.95  # slot-holding: running on the pod >= 95% of the window


@dataclass
class Run:
    dir: Path
    meta: dict
    reqs: list[dict]
    scrapes: list[dict]
    loads: list[dict]

    @classmethod
    def load(cls, d: Path) -> "Run":
        return cls(dir=d, meta=json.loads((d / "meta.json").read_text()),
                   reqs=sorted(jsonl.read(d / "requests.jsonl"), key=lambda r: r["idx"]),
                   scrapes=jsonl.read(d / "scrape.jsonl"), loads=jsonl.read(d / "loads.jsonl"))


def ttft(r: dict, ref: str = "t_sched") -> float:
    """First token minus `ref`; +inf if it timed out first, NaN for errors."""
    if r.get("t_first_token") is not None and r["status"] in ("ok", "timeout"):
        return r["t_first_token"] - r[ref]
    if r["status"] == "timeout":
        return math.inf
    return math.nan  # errors: excluded from latency, counted separately


def pct(xs: list[float], q: float) -> float | None:
    """Nearest-rank percentile, NaNs dropped; None when empty."""
    a = np.array([x for x in xs if not math.isnan(x)], dtype=float)
    if a.size == 0:
        return None
    return float(np.percentile(a, q, method="inverted_cdf"))


def latency_stats(rs: list[dict], output_tokens: int) -> dict[str, Any]:
    t = [ttft(r) for r in rs]
    t_send = [ttft(r, "t_send") for r in rs]
    counted = [x for x in t if not math.isnan(x)]
    done = [x for x in counted if math.isfinite(x)]
    itl = [(r["t_end"] - r["t_first_token"]) / (output_tokens - 1)
           for r in rs if r["status"] == "ok" and r.get("t_first_token") is not None and output_tokens > 1]
    ok = [r for r in rs if r["status"] == "ok"]
    return {
        "n": len(rs),
        "completed": len(ok),
        "completed_tokens": sum(r.get("completion_tokens") or 0 for r in ok),
        "timeouts": sum(r["status"] == "timeout" for r in rs),
        "errors": sum(r["status"] not in ("ok", "timeout") for r in rs),
        "ttft_p50": pct(t, 50),
        "ttft_p90": pct(t, 90),
        "ttft_p99": pct(t, 99),
        "ttft_max": max(counted) if counted else None,
        "ttft_mean_completed": float(np.mean(done)) if done else None,
        "ttft_send_p99": pct(t_send, 99),
        "share_over_5s": (sum(x > SLOW_S for x in counted) / len(counted)) if counted else None,
        "itl_p50": pct(itl, 50),
        "itl_p99": pct(itl, 99),
    }


def _jsonable(o: Any) -> Any:
    """+inf as the string "inf", so summaries stay standard JSON."""
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, float) and math.isinf(o):
        return "inf"
    return o


def pod_of(r: dict, addr_to_pod: dict[str, str]) -> str | None:
    return addr_to_pod.get(r.get("upstream") or "")


def holding_intervals(reqs: list[dict], addr_to_pod: dict[str, str], prefill_s: float) -> Holding:
    iv: dict[str, dict[str, list[tuple[float, float]]]] = defaultdict(lambda: defaultdict(list))
    for r in reqs:
        p = pod_of(r, addr_to_pod)
        if p is None or r["adapter"] == BASE or r.get("t_first_token") is None or r.get("t_end") is None:
            continue
        iv[p][r["adapter"]].append((r["t_first_token"] - prefill_s, r["t_end"]))
    return {p: {a: _merge(sorted(v)) for a, v in d.items()} for p, d in iv.items()}


def _merge(ivs: Intervals) -> Intervals:
    out: list[list[float]] = []
    for s, e in ivs:
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [(s, e) for s, e in out]


def _active_at(merged: Intervals, t: float) -> bool:
    i = bisect.bisect_right([s for s, _ in merged], t) - 1
    return i >= 0 and merged[i][0] <= t <= merged[i][1]


def _coverage(merged: Intervals, a: float, b: float) -> tuple[float, int]:
    """Share of [a, b] covered, and the number of idle gaps inside it."""
    tot, gaps, prev_end = 0.0, 0, None
    for s, e in merged:
        lo, hi = max(s, a), min(e, b)
        if hi <= lo:
            continue
        tot += hi - lo
        if prev_end is not None and lo > prev_end:
            gaps += 1
        prev_end = hi
    return tot / (b - a), gaps


def blocked_at_arrival(hold: Holding, reqs: list[dict], addr_to_pod: dict[str, str],
                       max_loras: int) -> dict[str, float | None]:
    """Per class: share of LoRA requests in `reqs` that, on reaching their pod,
    find >= max_loras other adapters holding slots there. `hold` is built from
    every request, including ones that arrived before the window."""
    hits: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for r in reqs:
        p = pod_of(r, addr_to_pod)
        if p is None or r["adapter"] == BASE or r["status"] not in ("ok", "timeout"):
            continue
        # Arrival at the pod is Envoy's upstream start; the client send time is
        # a close lower bound when the access log has no timing for the request.
        t = r.get("t_upstream") if r.get("t_upstream") is not None else r["t_send"]
        others = sum(1 for a, m in hold.get(p, {}).items() if a != r["adapter"] and _active_at(m, t))
        h = hits[r["cls"]]
        h[1] += 1
        h[0] += others >= max_loras
    return {c: (h[0] / h[1] if h[1] else None) for c, h in sorted(hits.items())}


def slot_holding(hold: Holding, pods: list[str], start: float, end: float) -> dict[str, dict]:
    """Per (pod, adapter) over [start, end]: running share and idle-gap count."""
    out = {}
    for p in pods:
        for a, m in sorted(hold.get(p, {}).items()):
            share, gaps = _coverage(m, start, end)
            if share > 0:
                out[f"{p}/{a}"] = {"running_share": share, "idle_gaps": gaps}
    return out


def reached_layout(hold: Holding, pods: list[str], max_loras: int, start: float, end: float,
                   window: float = LAYOUT_WINDOW_S) -> list[dict]:
    """Per window: does every pod hosting a non-holding adapter have < max_loras holding adapters?"""
    rows = []
    t = start
    while t + window <= end + 1e-9:
        row = {"t": t, "pods": {}}
        for p in pods:
            cov = {a: _coverage(m, t, t + window)[0] for a, m in hold.get(p, {}).items()}
            present = {a for a, c in cov.items() if c > 0}
            holding = {a for a, c in cov.items() if c >= HOLDING_SHARE}
            row["pods"][p] = {"holding": sorted(holding), "others": sorted(present - holding),
                              "violation": bool(present - holding) and len(holding) >= max_loras}
        row["holding_slots"] = sum(len(v["holding"]) for v in row["pods"].values())
        row["free_slots"] = len(pods) * max_loras - row["holding_slots"]
        row["violating_pods"] = sum(v["violation"] for v in row["pods"].values())
        rows.append(row)
        t += window
    return rows


def feasibility_bound(meta: dict) -> dict | None:
    """Offline bound from offered rates (DESIGN_LLMD_709.md §7). Needs capacity and unloaded E2E in meta."""
    sc = meta["scenario"]
    cap, e2e = sc.get("capacity_rps"), meta.get("e2e_unloaded_s")
    rates = meta.get("offered_rates")
    if not cap or not e2e or not rates:
        return None
    pods, slots = sc["pods"], sc["max_loras"]
    need_slots, tail_conc, compute_pods = 0, 0.0, 0
    per = {}
    for a, lam in rates.items():
        if a == BASE:
            continue
        k = max(1, math.ceil(lam / cap))
        per_pod_conc = lam * e2e / k
        holds = per_pod_conc >= 3.0  # P(idle) = e^-3 ~ 5%
        per[a] = {"rate": lam, "pods_for_compute": k, "concurrency_per_pod": per_pod_conc, "holds_slots": holds}
        if holds:
            need_slots += k
        else:
            tail_conc += 1 - math.exp(-lam * e2e)
        compute_pods += lam / cap
    base_load = rates.get(BASE, 0.0) / cap
    return {
        "adapters": per,
        "holding_slots_needed": need_slots,
        "tail_concurrency": tail_conc,
        "free_slots": pods * slots - need_slots,
        "slots_fit": pods * slots - need_slots >= tail_conc,
        "compute_pods_needed": compute_pods + base_load,
        "compute_fits": compute_pods + base_load <= pods,
    }


def imbalance(reqs: list[dict], addr_to_pod: dict[str, str], pods: list[str], start: float, end: float) -> dict:
    """Mean in-flight requests per pod (Little's law) and busiest / mean."""
    busy = {p: 0.0 for p in pods}
    for r in reqs:
        p = pod_of(r, addr_to_pod)
        if p not in busy or r.get("t_end") is None or r.get("t_send") is None:
            continue
        a, b = max(r["t_send"], start), min(r["t_end"], end)
        if b > a:
            busy[p] += b - a
    span = end - start
    inflight = {p: v / span for p, v in busy.items()}
    mean = sum(inflight.values()) / len(inflight) if inflight else 0.0
    return {"per_pod_inflight": inflight, "imbalance": (max(inflight.values()) / mean) if mean > 0 else None}


def activation_proxy(scrapes: list[dict]) -> dict[str, int]:
    """Adapters newly entering a pod's running set between consecutive scrapes.

    Used when engine logs are unavailable (simulator). On vLLM it overcounts:
    an idle adapter stays resident but leaves the running set.
    """
    last: dict[str, set[str]] = {}
    count: dict[str, int] = defaultdict(int)
    for s in sorted((s for s in scrapes if s.get("lora") is not None), key=lambda s: s["t"]):
        cur = set(s["lora"]["running"])
        prev = last.get(s["src"])
        if prev is not None:
            count[s["src"]] += len(cur - prev)
        last[s["src"]] = cur
    return dict(count)


def queue_trend(scrapes: list[dict], pods: list[str], wall_t0: float, start: float, end: float,
                span: float = 30.0) -> dict[str, Any]:
    """Median vLLM waiting queue per pod in the first and last `span` s of [start, end].

    A queue that grows across the window means the pod is past capacity and the run never
    reaches a steady state, so its p99 depends on run length. Those pods are counted.
    """
    def median_waiting(pod: str, a: float, b: float) -> float | None:
        xs = sorted(s["waiting"] for s in scrapes if s.get("src") == pod and s.get("waiting") is not None
                    and a <= s["t"] - wall_t0 < b)
        return xs[len(xs) // 2] if xs else None

    per_pod = {p: {"waiting_start": median_waiting(p, start, start + span),
                   "waiting_end": median_waiting(p, end - span, end)} for p in pods}
    growing = [p for p, q in per_pod.items() if q["waiting_start"] is not None and q["waiting_end"] is not None
               and q["waiting_end"] >= 20 and q["waiting_end"] > 2 * q["waiting_start"] + 10]
    return {"per_pod": per_pod, "growing_queue_pods": growing}


def envoy_overflow(run_dir: Path) -> dict | None:
    """Envoy error counters that moved during the run; None if a stats snapshot is missing."""
    def stat(tag):
        p = run_dir / f"envoy-stats-{tag}.json"
        if not p.exists():
            return None
        return {s["name"]: s.get("value", 0) for s in json.loads(p.read_text()).get("stats", []) if "value" in s}

    before, after = stat("before"), stat("after")
    if before is None or after is None:
        return None
    keys = [k for k in after if any(x in k for x in ("upstream_rq_pending_overflow", "upstream_rq_timeout",
                                                      "ext_proc.streams_failed", "ext_proc.failure_mode_allowed",
                                                      "ext_proc.message_timeouts", "upstream_cx_overflow",
                                                      "downstream_rq_timeout", "downstream_rq_idle_timeout"))]
    # upstream_cx_idle_timeout is not here: Envoy closes idle pod connections on purpose (envoy_config).
    return {k: after[k] - before.get(k, 0) for k in keys if after[k] - before.get(k, 0)}


def validity(run: Run, cls_full: dict, addr_to_pod: dict[str, str]) -> dict[str, Any]:
    """DESIGN_LLMD_709.md §6.4. A run is valid only if every check passes."""
    m, sc, reqs = run.meta, run.meta["scenario"], run.reqs
    ok = [r for r in reqs if r["status"] == "ok"]
    want = sc["output_tokens"]
    checks: dict[str, Any] = {
        "loop_lag_p99_s": m.get("loadgen", {}).get("loop_lag_p99_s"),
        "send_lag_p99_s": pct([r["send_lag"] for r in reqs if r.get("send_lag") is not None], 99),
        "wrong_length": sum(r.get("completion_tokens") not in (None, want) for r in ok),
        "missing_usage": sum(r.get("completion_tokens") is None for r in ok),
        "access_log_present": (run.dir / "access.jsonl").exists(),
        "unattributed": sum(not r.get("upstream") for r in ok),
        "upstream_outside_pods": sum(bool(r.get("upstream")) and r["upstream"] not in addr_to_pod for r in ok),
        "header_disagrees_with_access_log": sum(bool(r.get("served_by")) and r["served_by"] != r.get("upstream")
                                                for r in ok),
        "errors": sum(r["status"] not in ("ok", "timeout") for r in reqs),
        "envoy_overflow": envoy_overflow(run.dir),
        "envoy_stats_present": all((run.dir / f"envoy-stats-{t}.json").exists() for t in ("before", "after")),
    }
    if m.get("backend") == "vllm":
        # Every adapter is activated on its first use on a pod, so zero
        # activations means the log parsing broke, not that nothing loaded.
        lora_ok = sum(r["adapter"] != BASE for r in ok)
        checks["activations"] = sum(e["kind"] == "cpu_to_gpu" for e in run.loads)
        checks["loads_parsed"] = lora_ok == 0 or checks["activations"] > 0
    limit, base = sc.get("base_ttft_p99_limit_s"), cls_full.get("base")
    if limit is not None and base and base["ttft_p99"] is not None:
        checks["base_ttft_p99_s"] = base["ttft_p99"]
        checks["base_not_overloaded"] = base["ttft_p99"] <= limit
    checks["valid"] = bool(
        (checks["loop_lag_p99_s"] is None or checks["loop_lag_p99_s"] < 0.005)
        and (checks["send_lag_p99_s"] or 0) < 0.05
        and checks["access_log_present"] and checks["envoy_stats_present"] and not checks["envoy_overflow"]
        and checks["wrong_length"] == checks["missing_usage"] == checks["errors"] == 0
        and checks["unattributed"] == checks["upstream_outside_pods"] == 0
        and checks["header_disagrees_with_access_log"] == 0
        and checks.get("loads_parsed", True) and checks.get("base_not_overloaded", True)
    )
    checks["not_checked"] = ["residency response hook", "EPP plugin latency bound", "GPU throttling"]
    if limit is None:
        checks["not_checked"].append("base model not overloaded (base_check off, or no calibration)")
    return checks


def analyze(run_dir: Path) -> dict[str, Any]:
    """The run's summary: per-window latency, mechanism diagnostics and validity."""
    run = Run.load(run_dir)
    m, sc = run.meta, run.meta["scenario"]
    addr_to_pod = m["addr_to_pod"]
    pods = sorted(set(addr_to_pod.values()))
    max_loras, L = sc["max_loras"], sc["output_tokens"]
    steady, duration = sc["steady_after_s"], sc["duration_s"]
    prefill = m.get("prefill_estimate_s", 0.03)

    out: dict[str, Any] = {k: m.get(k) for k in ("run_id", "arm", "seed", "replicate", "backend")}
    out["scenario"] = sc["name"]
    for wname, rs in (("full", run.reqs), ("steady", [r for r in run.reqs if r["t_sched"] >= steady])):
        by_cls: dict[str, list[dict]] = defaultdict(list)
        by_ad: dict[str, list[dict]] = defaultdict(list)
        for r in rs:
            by_cls[r["cls"]].append(r)
            by_ad[r["adapter"]].append(r)
        cls = {c: latency_stats(v, L) for c, v in sorted(by_cls.items())}
        cls["lora_all"] = latency_stats([r for r in rs if r["adapter"] != BASE], L)
        ads = {a: latency_stats(v, L) for a, v in sorted(by_ad.items()) if a != BASE}
        p99 = {a: s["ttft_p99"] for a, s in ads.items() if s["ttft_p99"] is not None}
        out[wname] = {"classes": cls, "adapters": ads}
        if p99:
            w = max(p99, key=lambda a: p99[a])
            out[wname]["worst_adapter"] = {"adapter": w, "ttft_p99": p99[w]}

    st = [r for r in run.reqs if r["t_sched"] >= steady]
    ppa: dict[str, set[str]] = defaultdict(set)
    for r in st:
        p = pod_of(r, addr_to_pod)
        if p and r["adapter"] != BASE and r["status"] == "ok":
            ppa[r["adapter"]].add(p)
    out["pods_per_adapter"] = {a: len(v) for a, v in sorted(ppa.items())}
    hold = holding_intervals(run.reqs, addr_to_pod, prefill)
    out["imbalance"] = imbalance(run.reqs, addr_to_pod, pods, steady, duration)
    out["blocked_at_arrival"] = blocked_at_arrival(hold, st, addr_to_pod, max_loras)
    out["slot_holding"] = slot_holding(hold, pods, steady, duration)
    layout = reached_layout(hold, pods, max_loras, steady, duration)
    out["layout"] = {
        "windows": layout,
        "n_windows": len(layout),
        "windows_with_violation": sum(w["violating_pods"] > 0 for w in layout),
        "min_free_slots": min((w["free_slots"] for w in layout), default=None),
    }
    out["feasibility_bound"] = feasibility_bound(m)
    if run.loads:
        out["loads"] = dict(Counter(e["kind"] for e in run.loads))
    wall_t0 = m.get("loadgen", {}).get("wall_t0")
    if wall_t0 is not None:
        out["queue_trend"] = queue_trend(run.scrapes, pods, wall_t0, steady, duration)
    out["activation_proxy"] = activation_proxy(run.scrapes)
    out["activation_proxy_total"] = sum(out["activation_proxy"].values())
    out["validity"] = validity(run, out["full"]["classes"], addr_to_pod)
    return _jsonable(out)


def write(run_dir: Path) -> dict[str, Any]:
    """Analyse a run and write summary.json next to its records."""
    res = analyze(run_dir)
    (run_dir / "summary.json").write_text(json.dumps(res, indent=1))
    return res
