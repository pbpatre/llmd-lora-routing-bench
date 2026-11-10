# SPDX-License-Identifier: Apache-2.0
"""Cross-run report and the pre-registered #709 gate (DESIGN_LLMD_709.md §8).

The gate compares a candidate arm with reference arms seed by seed, on the
steady window. Proportions and counts use absolute margins; p99s use the
noise band, which is the wider of the observed A/A spread and #3392's
replicate spread.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

P99_BAND_3392 = (0.70, 1.41)
P99_FLOOR_S = 0.25  # p99 differences below this never fail a guard or count as a win
WINDOW = "steady"
GATE_SEEDS = (4, 5, 6)  # seeds 1-3 are for tuning (DESIGN_LLMD_709.md §8)

# (class, metric, rule, margin). Rules:
#   ratio_max: cand <= max(ref * band_hi, ref + P99_FLOOR_S)    abs_max: cand <= ref + margin
#   abs_lower: cand <= ref - margin (win)    ratio_lower: cand < ref * band_lo and ref - cand > P99_FLOOR_S (win)
#   timeouts_frac: cand_timeouts <= ref_timeouts + margin * n
GATE: dict[str, dict[str, Any]] = {
    "s1-fits": {
        "refs": ("residency-w4",),
        "win": None,
        "guards": [("cold", "ttft_p99", "ratio_max", None), ("cold", "share_over_5s", "abs_max", 0.01)],
    },
    "s2-oversubscribed": {
        "refs": ("residency-w4",),
        "win": ("cold", "share_over_5s", "abs_lower", 0.10),
        "guards": [("warm", "share_over_5s", "abs_max", 0.05), ("all", "timeouts", "timeouts_frac", 0.01)],
    },
    "s3-two-hot": {
        "refs": ("residency-w4", "residency-w4-rr2"),
        "win": None,
        "guards": [("hot", "ttft_p99", "ratio_max", None), ("cold", "share_over_5s", "abs_max", 0.01),
                   ("all", "timeouts", "timeouts_frac", 0.0)],
    },
    "s4-one-giant": {
        "refs": ("residency-w4", "residency-w4-rr2"),
        "win": ("giant", "ttft_p99", "ratio_lower", None),
        "guards": [("cold", "share_over_5s", "abs_max", 0.01)],
    },
}


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


def noise_band(idx: dict[Key, dict], scenario: str, cls: str, arm: str = "residency-w4") -> tuple[float, float]:
    """#3392's p99 band, widened by the observed A/A spread of this class's p99
    (replicates 0 and 1 of one arm and seed)."""
    lo, hi = P99_BAND_3392
    for (sc, a_arm, seed, rep), r0 in idx.items():
        r1 = idx.get((sc, a_arm, seed, 1))
        if sc != scenario or a_arm != arm or rep != 0 or r1 is None:
            continue
        a, b = metric(r0, cls, "ttft_p99"), metric(r1, cls, "ttft_p99")
        if a and b and math.isfinite(a) and math.isfinite(b) and a > 0 and b > 0:
            ratio = max(a / b, b / a)
            lo, hi = min(lo, 1 / ratio), max(hi, ratio)
    return lo, hi


def _check(rule: str, cand: Any, ref: Any, margin: float | None, band: tuple[float, float], n: int) -> bool | None:
    """One gate comparison; None when either value is missing."""
    if cand is None or ref is None:
        return None
    if rule == "ratio_max":
        return cand <= max(ref * band[1], ref + P99_FLOOR_S)
    if rule == "abs_max":
        return cand <= ref + margin
    if rule == "abs_lower":
        return cand <= ref - margin
    if rule == "ratio_lower":
        return cand < ref * band[0] and ref - cand > P99_FLOOR_S
    if rule == "timeouts_frac":
        return cand <= ref + margin * n
    raise ValueError(rule)


def gate(runs: list[dict], candidate: str, seeds: tuple[int, ...] = GATE_SEEDS) -> dict[str, Any]:
    """Evaluate the #709 gate for one candidate arm on the pre-registered seeds.

    A missing or invalid run for any required (scenario, arm, seed) fails the gate.
    """
    one_backend(runs, None)
    full = valid_index(runs)
    idx = {(sc, arm, seed): r for (sc, arm, seed, rep), r in full.items() if rep == 0}
    report: dict[str, Any] = {"candidate": candidate, "seeds": list(seeds), "scenarios": {}}
    wins, guards_ok = [], True
    for sc, spec in GATE.items():
        missing = [(arm, s) for arm in (candidate, *spec["refs"]) for s in seeds if (sc, arm, s) not in idx]
        entry: dict[str, Any] = {"refs": spec["refs"], "guards": [], "win": None,
                                 "missing_or_invalid": missing}
        report["scenarios"][sc] = entry
        if missing:
            guards_ok = False
            wins.append(False)
            continue

        def per_seed(ref_arm, cls, name, rule, margin):
            band = noise_band(full, sc, cls)
            out = []
            for s in seeds:
                c, r = idx[(sc, candidate, s)], idx[(sc, ref_arm, s)]
                n = metric(c, cls, "n") or 0
                out.append(_check(rule, metric(c, cls, name), metric(r, cls, name), margin, band, n))
            return out

        for ref_arm in spec["refs"]:
            for cls, name, rule, margin in spec["guards"]:
                results = per_seed(ref_arm, cls, name, rule, margin)
                ok = all(x is True for x in results)
                entry["guards"].append({"ref": ref_arm, "class": cls, "metric": name, "rule": rule,
                                        "band": noise_band(full, sc, cls), "per_seed": results, "ok": ok})
                guards_ok &= ok
        if spec["win"]:
            cls, name, rule, margin = spec["win"]
            per_ref = {ref_arm: per_seed(ref_arm, cls, name, rule, margin) for ref_arm in spec["refs"]}
            won = all(x is True for v in per_ref.values() for x in v)
            entry["win"] = {"class": cls, "metric": name, "band": noise_band(full, sc, cls),
                            "per_ref": per_ref, "won": won}
            wins.append(won)
    report["passes"] = bool(any(wins) and guards_ok)
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
    lines = ["| scenario | arm | seeds | class | TTFT p99 (s) | > 5 s | timeouts | blocked-at-arrival | loads |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
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
            lines.append(f"| {sc} | {arm} | {seeds} | {cls} | {per_seed('ttft_p99')} | "
                         f"{per_seed('share_over_5s', True)} | {per_seed('timeouts')} | {blocked} | {loads} |")
    invalid = [r["run_id"] for r in runs if not r["valid"]]
    if invalid:
        lines.append(f"\nExcluded {len(invalid)} invalid run(s); see each summary.json `validity`: "
                     + ", ".join(invalid))
    return "\n".join(lines)
