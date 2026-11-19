# SPDX-License-Identifier: Apache-2.0
"""Cross-run report and the #709 gate (DESIGN_LLMD_709.md §8).

The gate judges a candidate arm against each reference arm on three metrics, all
computed per run by analyze.misses over the steady window:

- miss rate (primary): share of all requests with TTFT above MISS_S, pooled over seeds;
- bad seeds (reliability): seeds whose miss rate is above BAD_SEED;
- worst adapter (fairness): the highest per-adapter miss rate, on the worst seed.

S3 and S4 are win scenarios and both must be won; S1 and S5 are guards. S2 has no
feasible layout, so it is reported but not gated.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

WINDOW = "steady"
TUNING_SEEDS = (1, 2)  # a candidate's settings are chosen on these, then frozen
GATE_SEEDS = (3, 4, 5)
REFS = ("residency-w4", "residency-w4-rr2")
BAD_SEED = 0.05           # a seed whose miss rate is above this "tipped"
WIN_MARGIN = 0.02         # win: pooled miss rate at least 2 points below the reference
GUARD_MARGIN = 0.01       # guard: pooled miss rate at most 1 point above the reference
FAIRNESS_MARGIN = 0.05    # worst adapter at most 5 points above the reference
GATE = {"s3-two-hot": "win", "s4-one-giant": "win", "s1-fits": "guard", "s5-zipf": "guard"}


Key = tuple[str, str, int, int]  # (scenario, arm, seed, replicate)


def _num(v: Any) -> Any:
    """Undo analyze._jsonable: "inf" back to +inf."""
    return math.inf if v == "inf" else v


def load_runs(root: Path) -> list[dict]:
    """Every run summary under `root`, with a top-level `valid` flag."""
    out = []
    for path in sorted(root.glob("**/summary.json")):
        res = json.loads(path.read_text())
        res["valid"] = res["validity"]["valid"]
        out.append(res)
    return out


def valid_index(runs: list[dict]) -> dict[Key, dict]:
    """Valid runs by key. Two valid runs with one key (say, a set run twice into
    one tree) would make every comparison ambiguous, so that is an error."""
    idx: dict[Key, dict] = {}
    for r in runs:
        if not r["valid"]:
            continue
        key = (r["scenario"], r["arm"], r["seed"], r.get("replicate", 0))
        if key in idx:
            raise ValueError(f"two valid runs for {key}: {idx[key]['run_id']} and {r['run_id']}")
        idx[key] = r
    return idx


def one_backend(runs: list[dict], backend: str | None) -> list[dict]:
    """Never mix simulator and GPU runs in a table or gate."""
    if backend:
        return [r for r in runs if r.get("backend") == backend]
    kinds = {r.get("backend") for r in runs}
    if len(kinds) > 1:
        raise ValueError(f"runs from several backends {sorted(kinds)}; pass --backend")
    return runs


def metric(run: dict, cls: str, name: str) -> Any:
    """A class metric in the steady window. Class "all" sums counts (n, timeouts,
    errors) over every class and is None for other metrics."""
    classes = run[WINDOW]["classes"]
    if cls == "all":
        vals = [_num(c.get(name)) for k, c in classes.items() if k != "lora_all"]
        return sum(vals) if name in ("timeouts", "n", "errors") and vals else None
    c = classes.get(cls)
    return None if c is None else _num(c.get(name))


def seed_metrics(runs: list[dict]) -> dict[str, Any]:
    """The three gate metrics over one arm's runs (one per seed) in one scenario."""
    m = [r["misses"] for r in runs]
    n = sum(x["n"] for x in m)
    worst = [x["worst_adapter"]["miss_rate"] for x in m if x["worst_adapter"]]
    return {
        "miss_rate": sum(x["misses"] for x in m) / n if n else None,
        "per_seed": [x["miss_rate"] for x in m],
        "bad_seeds": sum(x["miss_rate"] > BAD_SEED for x in m),
        "worst_adapter": max(worst, default=0.0),
    }


def judge(role: str, cand: dict[str, Any], ref: dict[str, Any]) -> dict[str, bool]:
    """Each condition of a win or a guard, candidate against one reference."""
    fair = cand["worst_adapter"] <= ref["worst_adapter"] + FAIRNESS_MARGIN
    if role == "win":
        return {"miss_rate": cand["miss_rate"] <= ref["miss_rate"] - WIN_MARGIN,
                "bad_seeds": cand["bad_seeds"] <= ref["bad_seeds"], "fairness": fair}
    return {"miss_rate": cand["miss_rate"] <= ref["miss_rate"] + GUARD_MARGIN, "fairness": fair}


def gate(runs: list[dict], candidate: str, seeds: tuple[int, ...] = GATE_SEEDS) -> dict[str, Any]:
    """Evaluate the #709 gate for one candidate arm on the gate seeds.

    A missing or invalid run for any required (scenario, arm, seed) fails the gate.
    """
    one_backend(runs, None)
    idx = {(sc, arm, seed): r for (sc, arm, seed, rep), r in valid_index(runs).items() if rep == 0}
    report: dict[str, Any] = {"candidate": candidate, "seeds": list(seeds), "scenarios": {}}
    passes = True
    for sc, role in GATE.items():
        arms = (candidate, *REFS)
        missing = [(arm, s) for arm in arms for s in seeds if (sc, arm, s) not in idx]
        entry: dict[str, Any] = {"role": role, "missing_or_invalid": missing}
        report["scenarios"][sc] = entry
        if missing:
            passes = False
            continue
        metrics = {arm: seed_metrics([idx[(sc, arm, s)] for s in seeds]) for arm in arms}
        entry["metrics"] = metrics
        entry["vs"] = {ref: judge(role, metrics[candidate], metrics[ref]) for ref in REFS}
        entry["ok"] = all(all(v.values()) for v in entry["vs"].values())
        passes &= entry["ok"]
    report["passes"] = passes
    return report


def _fmt(v: Any, percent: bool = False) -> str:
    v = _num(v)
    if v is None:
        return "-"
    if isinstance(v, float) and math.isinf(v):
        return "inf"
    if percent:
        return f"{100 * v:.1f}%"
    return f"{v:.2f}" if isinstance(v, float) else str(v)


def table(runs: list[dict]) -> str:
    """Markdown table of valid replicate-0 runs: one row per (scenario, arm, class), values per seed."""
    rows: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for (sc, arm, _, rep), r in valid_index(runs).items():
        if rep == 0:
            rows[(sc, arm)].append(r)
    lines = ["| scenario | arm | seeds | class | miss rate | TTFT p99 (s) | > 5 s | timeouts | blocked-at-arrival | loads |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for (sc, arm), rs in sorted(rows.items()):
        rs.sort(key=lambda r: r["seed"])
        seeds = ",".join(str(r["seed"]) for r in rs)
        # vLLM: activations and loads from the logs; simulator: the running-set proxy.
        loads = " / ".join(_fmt(sum(r["loads"].values()) if r.get("loads") else r.get("activation_proxy_total"))
                           for r in rs)
        for cls in sorted({c for r in rs for c in r[WINDOW]["classes"] if c != "lora_all"}):
            def per_seed(name: str, percent: bool = False) -> str:
                return " / ".join(_fmt(metric(r, cls, name), percent) for r in rs)
            blocked = "-" if cls == "base" else " / ".join(_fmt(r["blocked_at_arrival"].get(cls), True) for r in rs)
            missed = " / ".join(_fmt((r.get("misses", {}).get("classes", {}).get(cls) or {}).get("miss_rate"), True)
                                for r in rs)
            lines.append(f"| {sc} | {arm} | {seeds} | {cls} | {missed} | {per_seed('ttft_p99')} | "
                         f"{per_seed('share_over_5s', True)} | {per_seed('timeouts')} | {blocked} | {loads} |")
    invalid = [r["run_id"] for r in runs if not r["valid"]]
    if invalid:
        lines.append(f"\nExcluded {len(invalid)} invalid run(s); see each summary.json `validity`: "
                     + ", ".join(invalid))
    return "\n".join(lines)
