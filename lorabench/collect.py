# SPDX-License-Identifier: Apache-2.0
"""Post-run collection: vLLM adapter loads from logs, Envoy access log join."""

from __future__ import annotations

import datetime as dt
import json
import re
from pathlib import Path

# Strings verified against vLLM's vllm/lora model manager (Feb 2026 checkout);
# re-verify for each pinned version (DESIGN_LLMD_709.md §6.3). The LoRA manager runs in
# the EngineCore process, whose stdout lines carry a "(EngineCore pid=N) "
# prefix (vllm/v1/engine/core.py decorate_logs).
_PREFIX = r"^(?:\(\S+ pid=\d+\) )?"
ADD_RE = re.compile(_PREFIX + r"(\S+) DEBUG \S+ Adding lora\. Model id: (\d+), int id: (\d+)")
ACT_RE = re.compile(_PREFIX + r"(\S+) DEBUG \S+ Activating LoRA\. int id: (\d+), slot index: (\d+)")
_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def _ts(s: str) -> float:
    return dt.datetime.fromisoformat(s).timestamp()


def vllm_loads(log: Path, pod: str, int_id_to_name: dict[int, str], after_wall: float) -> list[dict]:
    """Adapter loads in one pod log, ignoring lines from before load started (warm-up)."""
    out = []
    for line in log.read_text(errors="replace").splitlines():
        line = _ANSI.sub("", line)
        m = ADD_RE.match(line)
        kind = "storage_to_cpu"
        if not m:
            m = ACT_RE.match(line)
            kind = "cpu_to_gpu"
        if not m:
            continue
        t = _ts(m.group(1))
        if t < after_wall:
            continue
        int_id = int(m.group(3) if kind == "storage_to_cpu" else m.group(2))
        out.append({"t": t, "pod": pod, "kind": kind, "int_id": int_id,
                    "adapter": int_id_to_name.get(int_id, f"int-{int_id}")})
    return out


def access_log(path: Path) -> dict[str, dict]:
    """Envoy access log keyed by request id."""
    out: dict[str, dict] = {}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        rid = rec.get("rid")
        if rid and rid != "-":
            out[rid] = rec
    return out


def join_access(reqs: list[dict], acc: dict[str, dict], wall_t0: float) -> None:
    """Attach the authoritative upstream and upstream start time (seconds after t0) to each request."""
    for r in reqs:
        a = acc.get(r.get("rid"))
        if not a:
            continue
        up = a.get("upstream")
        r["upstream"] = None if up in (None, "-") else up
        r["envoy_flags"] = a.get("flags")
        try:
            start = float(a["start"])
            tx = a.get("us_tx_begin_us")
            r["t_upstream"] = start - wall_t0 + (float(tx) / 1e6 if tx not in (None, "-") else 0.0)
        except (KeyError, TypeError, ValueError):
            r["t_upstream"] = None
