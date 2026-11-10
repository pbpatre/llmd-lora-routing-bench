# SPDX-License-Identifier: Apache-2.0
"""One run end to end, and run matrices with balanced arm order."""

from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import json
import multiprocessing as mp
import os
import platform
import time
from pathlib import Path

from . import analyze, collect, jsonl, loadgen, schedule, scrape
from .scenario import Scenario, load
from .topology import SIM_DEFAULTS, Topology, tool

# Client-side estimate of prefill time, used to start an adapter's slot holding
# before its first token (DESIGN_LLMD_709.md §7). Small next to the multi-second waits measured.
PREFILL_ESTIMATE_S = {"sim": 0.03, "vllm": 0.05}
SCRAPE_INTERVAL_S = 0.1


def _seconds(s: str) -> float:
    """'30ms' -> 0.03. Units: us, ms, s."""
    s = s.strip()
    for unit, f in (("ms", 1e-3), ("us", 1e-6), ("s", 1.0)):
        if s.endswith(unit):
            return float(s[: -len(unit)]) * f
    return float(s)


def calibration_for(cal: dict | None, sc: Scenario) -> dict:
    """capacity_rps, e2e_unloaded_s and base_ttft_p99_limit_s at this scenario's output length."""
    if not cal:
        return {}
    at = cal.get("lengths", {}).get(str(sc.output_tokens))
    if not at:
        raise ValueError(f"calibration has no entry for L={sc.output_tokens}; run S0 at that length")
    out = {"capacity_rps": at["capacity_rps"], "e2e_unloaded_s": at["e2e_unloaded_s"]}
    if at.get("ttft_p99_unloaded_s") and sc.base_check:
        # DESIGN_LLMD_709.md §6.4 check 6. S0 measures one adapter, whose unloaded TTFT
        # is no lower than the base model's, so the limit is conservative.
        out["base_ttft_p99_limit_s"] = 3.0 * at["ttft_p99_unloaded_s"]
    return out


def sim_calibration(sc: Scenario) -> dict:
    """Stand-in calibration from the scenario's simulator settings."""
    p = {**SIM_DEFAULTS, **sc.sim}
    out = {"capacity_rps": sc.sim.get("capacity_rps"),
           "e2e_unloaded_s": _seconds(p["ttft"]) + sc.output_tokens * _seconds(p["itl"])}
    if sc.base_check:
        out["base_ttft_p99_limit_s"] = 10 * _seconds(p["ttft"])  # sim TTFT is fixed; 10x means real queueing
    return out


def resolve(path: Path, backend: str, calibration: dict | None = None,
            duration_scale: float = 1.0) -> tuple[Scenario, dict]:
    """The scenario with its rate resolved, and the calibration values that apply to it."""
    probe = load(path, capacity_rps=1.0)  # only for the output length and the sim block
    cal = sim_calibration(probe) if backend == "sim" else calibration_for(calibration, probe)
    sc = load(path, capacity_rps=cal.get("capacity_rps"))
    if duration_scale != 1.0:
        sc = dataclasses.replace(sc, duration_s=sc.duration_s * duration_scale,
                                 steady_after_s=sc.steady_after_s * duration_scale)
    return sc, cal


def _sha256(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _scrape_entry(pods, epp_url, out_path, stop_evt):
    """Scraper process: poll until the parent sets stop_evt."""
    async def main():
        stop = asyncio.Event()

        async def bridge():
            while not stop_evt.is_set():
                await asyncio.sleep(0.05)
            stop.set()

        await asyncio.gather(bridge(), scrape.run(pods, epp_url, Path(out_path), stop, SCRAPE_INTERVAL_S))

    asyncio.run(main())


def run_one(path: Path, arm: str, seed: int, backend: str, runs_root: Path, *, replicate: int = 0,
            duration_scale: float = 1.0, model: str = "", adapters_dir: Path | None = None,
            calibration: dict | None = None) -> Path:
    """Start the topology, replay the seed's schedule, collect and analyse. Returns the run directory."""
    sc, cal = resolve(path, backend, calibration, duration_scale)
    stamp = time.strftime("%Y%m%dT%H%M%S") + f"{time.time() % 1:.3f}"[1:]
    run_id = f"{sc.name}__{arm}__s{seed}__r{replicate}__{backend}__{stamp}"
    run_dir = runs_root / sc.name / run_id
    run_dir.mkdir(parents=True)
    reqs = schedule.generate(sc, seed)
    topo = Topology(sc=sc, arm=arm, backend=backend, run_dir=run_dir, seed=seed, model=model,
                    adapters_dir=adapters_dir)
    binaries = {"epp": _sha256(tool("epp", "LORABENCH_EPP"))}
    if backend == "sim":
        binaries["sim"] = _sha256(tool("llm-d-inference-sim", "LORABENCH_SIM"))
    meta = {
        "run_id": run_id, "arm": arm, "seed": seed, "replicate": replicate, "backend": backend,
        "schedule_sha256": schedule.write(reqs, run_dir / "schedule.jsonl"), "n_requests": len(reqs),
        "scenario": {**{k: v for k, v in dataclasses.asdict(sc).items() if k not in ("groups", "classes")},
                     "adapters": sc.adapters(), "base_ttft_p99_limit_s": cal.get("base_ttft_p99_limit_s")},
        "addr_to_pod": topo.addr_to_pod(),
        "offered_rates": {k: p * sc.rate for k, p in schedule.target_probabilities(sc).items()},
        "e2e_unloaded_s": cal.get("e2e_unloaded_s"),
        "calibration_used": backend == "vllm" and bool(cal),
        "prefill_estimate_s": PREFILL_ESTIMATE_S[backend],
        "duration_scale": duration_scale,
        "host": {"platform": platform.platform(), "python": platform.python_version()},
        "binaries": binaries,
        "router_commit": os.environ.get("LORABENCH_ROUTER_COMMIT"),  # set by setup-node.sh's env.sh
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=1))

    stop_evt = mp.get_context("spawn").Event()
    scraper = None
    try:
        topo.start()
        scraper = mp.get_context("spawn").Process(
            target=_scrape_entry, daemon=True,
            args=({p.name: p.url for p in topo.pods}, topo.epp_metrics, str(run_dir / "scrape.jsonl"), stop_evt))
        scraper.start()
        placement = {k: list(v) for k, v in sc.oracle.items()} if arm == "oracle" else None
        meta["loadgen"] = loadgen.run(topo.gateway, reqs, run_dir, backend, sc.timeout_s, run_tag=run_id,
                                      oracle_placement=placement)
    finally:
        stop_evt.set()
        if scraper is not None:
            scraper.join(timeout=10)
            if scraper.is_alive():
                scraper.terminate()
                scraper.join(timeout=5)
        topo.stop()

    wall_t0 = meta["loadgen"]["wall_t0"]
    reqs_out = jsonl.read(run_dir / "requests.jsonl")
    collect.join_access(reqs_out, collect.access_log(run_dir / "access.jsonl"), wall_t0)
    jsonl.write(run_dir / "requests.jsonl", reqs_out)
    if backend == "vllm":
        # vLLM numbers --lora-modules from 1 in order (DESIGN_LLMD_709.md §6.3).
        id_map = {i + 1: a for i, a in enumerate(sc.adapters())}
        loads = [e for p in topo.pods
                 for e in collect.vllm_loads(run_dir / "logs" / f"{p.name}.log", p.name, id_map, wall_t0)]
        jsonl.write(run_dir / "loads.jsonl", loads)
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=1))
    analyze.write(run_dir)
    return run_dir


def latin_order(arms: list[str], seeds: list[int], aa_arm: str | None = None) -> list[tuple[int, str, int]]:
    """(seed, arm, replicate) in run order. Arm order rotates per seed so no arm
    always runs first. With aa_arm, each seed block starts with that arm and
    ends with its replicate, bracketing the block (DESIGN_LLMD_709.md §8)."""
    out = []
    for i, s in enumerate(seeds):
        rest = [a for a in arms if a != aa_arm]
        k = i % len(rest) if rest else 0
        block = [(s, a, 0) for a in rest[k:] + rest[:k]]
        if aa_arm:
            block = [(s, aa_arm, 0)] + block + [(s, aa_arm, 1)]
        out += block
    return out
