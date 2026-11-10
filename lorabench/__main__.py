# SPDX-License-Identifier: Apache-2.0
"""llmd-lora-routing-bench command line.

  python -m lorabench schedule SCENARIO --seed 1
  python -m lorabench run SCENARIO --arm residency-w4 --seed 1 [--backend vllm ...]
  python -m lorabench matrix SCENARIO... --arms stock residency-w4 --seeds 1 2 3 [--aa]
  python -m lorabench calibrate --backend vllm --model M --adapters-dir D
  python -m lorabench analyze RUN_DIR
  python -m lorabench report RUNS [--backend vllm] [--gate ARM]
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
import traceback
from collections import Counter
from pathlib import Path

from . import analyze, calibrate, compare, runner, schedule


def _engine_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--backend", choices=["sim", "vllm"], default="sim")
    p.add_argument("--runs", type=Path, default=Path("runs"), help="root directory for run output")
    p.add_argument("--model", default="", help="vLLM model path")
    p.add_argument("--adapters-dir", type=Path, help="adapters written by tools/make_adapters.py")


def _run_args(p: argparse.ArgumentParser) -> None:
    _engine_args(p)
    p.add_argument("--calibration", type=Path, help="calibration.json from `calibrate`")
    p.add_argument("--duration-scale", type=float, default=1.0, help="shorten runs for smoke tests")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="lorabench")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("schedule", help="print a schedule's size and class mix")
    p.add_argument("scenario", type=Path)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--backend", choices=["sim", "vllm"], default="sim")
    p.add_argument("--calibration", type=Path)

    p = sub.add_parser("run", help="one run")
    p.add_argument("scenario", type=Path)
    p.add_argument("--arm", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--replicate", type=int, default=0)
    _run_args(p)

    p = sub.add_parser("matrix", help="arms x seeds for each scenario, in balanced order")
    p.add_argument("scenarios", type=Path, nargs="+")
    p.add_argument("--arms", nargs="+", required=True)
    p.add_argument("--seeds", type=int, nargs="+", required=True)
    p.add_argument("--aa", action="store_true", help="bracket each seed block with two residency-w4 runs (A/A)")
    p.add_argument("--keep-going", action="store_true", help="continue after a failed run")
    _run_args(p)

    p = sub.add_parser("calibrate", help="S0: one-pod capacity and adapter load costs")
    p.add_argument("scenario", type=Path, nargs="?", default=Path("scenarios/s0-calibration.yaml"))
    _engine_args(p)

    p = sub.add_parser("analyze", help="re-run the analysis of one run")
    p.add_argument("run_dir", type=Path)

    p = sub.add_parser("report", help="per-seed table, and optionally the #709 gate")
    p.add_argument("runs", type=Path)
    p.add_argument("--backend", choices=["sim", "vllm"], help="required when RUNS holds both")
    p.add_argument("--gate", metavar="ARM", help="evaluate the #709 gate for this candidate arm")
    p.add_argument("--seeds", type=int, nargs="+", default=list(compare.GATE_SEEDS),
                   help="gate seeds; the pre-registered verdict uses the default 4 5 6")

    a = ap.parse_args(argv)
    # SIGTERM/SIGHUP (tmux kill, a dropped SSH session) must still run the cleanup that stops the pods.
    for sig in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, lambda signum, _frame: sys.exit(128 + signum))
    cal = json.loads(a.calibration.read_text()) if getattr(a, "calibration", None) else None

    if a.cmd == "schedule":
        sc, _ = runner.resolve(a.scenario, a.backend, cal)
        reqs = schedule.generate(sc, a.seed)
        print(json.dumps({"scenario": sc.name, "seed": a.seed, "rate": sc.rate, "n": len(reqs),
                          "by_class": Counter(r.cls for r in reqs)}, indent=1))
    elif a.cmd == "run":
        d = runner.run_one(a.scenario, a.arm, a.seed, a.backend, a.runs, replicate=a.replicate,
                           duration_scale=a.duration_scale, model=a.model, adapters_dir=a.adapters_dir,
                           calibration=cal)
        s = json.loads((d / "summary.json").read_text())
        print(d)
        print(json.dumps({"valid": s["validity"]["valid"], "classes": {
            c: {k: v[k] for k in ("n", "ttft_p99", "share_over_5s", "timeouts")}
            for c, v in s["steady"]["classes"].items()}}, indent=1))
    elif a.cmd == "matrix":
        failures = []
        for path in a.scenarios:
            for seed, arm, rep in runner.latin_order(a.arms, a.seeds, aa_arm="residency-w4" if a.aa else None):
                try:
                    d = runner.run_one(path, arm, seed, a.backend, a.runs, replicate=rep,
                                       duration_scale=a.duration_scale, model=a.model,
                                       adapters_dir=a.adapters_dir, calibration=cal)
                    print(f"done {d}", flush=True)
                except Exception:
                    failures.append((str(path), arm, seed, rep))
                    traceback.print_exc()
                    if not a.keep_going:
                        return 1
        if failures:
            print(f"failures: {failures}", file=sys.stderr)
            return 1
    elif a.cmd == "calibrate":
        d = calibrate.run(a.scenario, a.backend, a.runs, model=a.model, adapters_dir=a.adapters_dir)
        print(d / "calibration.json")
    elif a.cmd == "analyze":
        analyze.write(a.run_dir)
        print(a.run_dir / "summary.json")
    elif a.cmd == "report":
        runs = compare.one_backend(compare.load_runs(a.runs), a.backend)
        print(compare.table(runs))
        if a.gate:
            print(json.dumps(compare.gate(runs, a.gate, tuple(a.seeds)), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
