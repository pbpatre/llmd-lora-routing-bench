# SPDX-License-Identifier: Apache-2.0
"""S0 calibration: one-pod capacity per output length, and adapter load cost per rank.

Capacity C_L is the highest swept rate at which, relative to the lowest rate,
TTFT p99 stays within a factor and nearly every request completes: the TTFT knee. The full curve is written next to the chosen value.

Load cost (vLLM only; the simulator has none): with --max-loras 1 and
--max-cpu-loras 2, the first request to an adapter pays storage->CPU plus
CPU->GPU; alternating two already-loaded adapters pays CPU->GPU only (the CPU
tier holds both, the single GPU slot doesn't). Costs are TTFT differences
against a request to the resident adapter, five samples per rank.
"""

from __future__ import annotations

import dataclasses
import json
import time
from pathlib import Path

import numpy as np
import yaml

from . import analyze, jsonl, loadgen, schedule
from .scenario import Scenario, load
from .topology import Pod, Topology, http

LOAD_COST_PREFIX = "lc"  # S0 adapter groups named lc<rank> are the load-cost probes


def _sweep_step(pod: Pod, sc: Scenario, rate: float, L: int, duration: float, out_dir: Path, backend: str) -> dict:
    step = dataclasses.replace(sc, rate=rate, duration_s=duration, output_tokens=L,
                               steady_after_s=min(10.0, duration / 3))
    d = out_dir / f"L{L}-r{rate:g}"
    d.mkdir(parents=True, exist_ok=True)
    lg = loadgen.run(pod.url + "/v1/chat/completions", schedule.generate(step, seed=int(rate * 100) + L), d,
                     backend, timeout_s=120, run_tag=f"cal-{L}-{rate:g}", shards=2)
    rs = [r for r in jsonl.read(d / "requests.jsonl") if r["t_sched"] >= step.steady_after_s]
    st = analyze.latency_stats(rs, L)
    e2e = [r["t_end"] - r["t_sched"] for r in rs if r["status"] == "ok"]
    return {"rate": rate, "L": L, "n": st["n"], "completed_fraction": st["completed"] / max(1, st["n"]),
            "ttft_p50": st["ttft_p50"], "ttft_p99": st["ttft_p99"], "itl_p50": st["itl_p50"],
            "e2e_mean": float(np.mean(e2e)) if e2e else None, "loop_lag_p99_s": lg["loop_lag_p99_s"]}


def capacity(curve: list[dict], rule: dict) -> float | None:
    """The highest rate before the first step that breaks the rule, relative to the lowest rate.

    The rule bounds TTFT p99 and the completed share; an ITL bound is optional, because
    batching raises ITL smoothly long before a pod saturates.
    """
    base, best = curve[0], None
    for c in curve:
        if c["ttft_p99"] is None:
            break
        itl_factor = rule.get("itl_p50_factor")
        if (c["ttft_p99"] > rule["ttft_p99_factor"] * base["ttft_p99"]
                or (itl_factor and c["itl_p50"] > itl_factor * base["itl_p50"])
                or c["completed_fraction"] < rule["completed_fraction"]):
            break
        best = c["rate"]
    return best


def _ttft_once(pod: Pod, model: str) -> float:
    """Seconds for a one-token completion: TTFT plus a fixed overhead that cancels in differences."""
    body = {"model": model, "messages": [{"role": "user", "content": f"probe {time.time_ns()}"}],
            "max_tokens": 1, "stream": False}
    t = time.monotonic()
    code, _ = http(pod.url + "/v1/chat/completions", body, timeout=60)
    if code != 200:
        raise RuntimeError(f"probe for {model} failed: HTTP {code}")
    return time.monotonic() - t


def load_costs(pod: Pod, sc: Scenario) -> dict:
    out = {}
    for g in sc.groups:
        if not g.prefix.startswith(LOAD_COST_PREFIX):
            continue
        names = g.names()
        storage, activation = [], []
        for name in names:
            first = _ttft_once(pod, name)       # storage->CPU + CPU->GPU
            resident = _ttft_once(pod, name)    # already on GPU
            storage.append(first - resident)
        a, b = names[-2], names[-1]             # both in the CPU tier now
        for _ in range(5):
            _ttft_once(pod, a)
            t_b = _ttft_once(pod, b)            # evicts a from the single GPU slot
            activation.append(t_b - _ttft_once(pod, b))
        out[g.rank] = {"storage_plus_activation_s": storage, "activation_s": activation,
                       "storage_median_s": float(np.median(storage)),
                       "activation_median_s": float(np.median(activation))}
    return out


def run(path: Path, backend: str, runs_root: Path, model: str = "", adapters_dir: Path | None = None) -> Path:
    cal = yaml.safe_load(path.read_text())["calibration"]
    sc = load(path)
    out_dir = runs_root / sc.name / f"{sc.name}__{backend}__{time.strftime('%Y%m%dT%H%M%S')}"
    out_dir.mkdir(parents=True)
    topo = Topology(sc=sc, arm="none", backend=backend, run_dir=out_dir, seed=0, model=model,
                    adapters_dir=adapters_dir)
    pod = topo.pods[0]
    result: dict = {"backend": backend, "scenario": sc.name, "rule": cal["capacity_rule"], "lengths": {}}
    try:
        topo.start_pods()
        for L in cal["output_tokens"]:
            curve = []
            for rate in cal["rates"]:
                c = _sweep_step(pod, sc, rate, L, cal["step_duration_s"], out_dir, backend)
                curve.append(c)
                print(json.dumps(c), flush=True)
                if c["completed_fraction"] < 0.5:  # far past the knee
                    break
            result["lengths"][str(L)] = {"capacity_rps": capacity(curve, cal["capacity_rule"]),
                                         "e2e_unloaded_s": curve[0]["e2e_mean"],
                                         "ttft_p99_unloaded_s": curve[0]["ttft_p99"], "curve": curve}
        if backend == "vllm":
            result["load_costs"] = load_costs(pod, sc)
    finally:
        topo.stop()
    (out_dir / "calibration.json").write_text(json.dumps(result, indent=1))
    return out_dir
