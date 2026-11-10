# SPDX-License-Identifier: Apache-2.0
"""Periodic Prometheus scrapes of the pods (and EPP) during a run."""

from __future__ import annotations

import asyncio
import json
import re
import time
from pathlib import Path

import aiohttp

_LINE = re.compile(r'^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{(.*)\})?\s+(\S+)')
_LABEL = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)="((?:[^"\\]|\\.)*)"')

POD_GAUGES = {
    "vllm:num_requests_running": "running",
    "vllm:num_requests_waiting": "waiting",
    "vllm:kv_cache_usage_perc": "kv",
}
LORA_INFO = "vllm:lora_requests_info"
EPP_INTERVAL_S = 5.0


def parse(text: str) -> list[tuple[str, dict[str, str], float]]:
    """(name, labels, value) for each sample in a Prometheus text exposition."""
    out = []
    for line in text.splitlines():
        if not line or line[0] == "#":
            continue
        m = _LINE.match(line)
        if not m:
            continue
        try:
            value = float(m.group(4))
        except ValueError:
            continue
        labels = dict(_LABEL.findall(m.group(3) or ""))
        out.append((m.group(1), labels, value))
    return out


def _split(v: str) -> list[str]:
    return [a for a in v.split(",") if a]


def pod_sample(text: str) -> dict:
    """Reduce one /metrics body to the fields the analysis uses.

    vllm:lora_requests_info keeps one series per label combination with the
    creation timestamp as value; the newest series is the current state.
    """
    s: dict = {"running": None, "waiting": None, "kv": None, "lora": None}
    newest = -1.0
    for name, labels, value in parse(text):
        if name in POD_GAUGES:
            # Sum over label sets (e.g. one series per model_name).
            key = POD_GAUGES[name]
            s[key] = (s[key] or 0.0) + value
        elif name == LORA_INFO and value > newest:
            newest = value
            max_lora = labels.get("max_lora")
            s["lora"] = {
                "running": _split(labels.get("running_lora_adapters", "")),
                "waiting": _split(labels.get("waiting_lora_adapters", "")),
                "max_lora": int(max_lora) if max_lora not in (None, "") else None,
                "ts": value,
            }
    return s


async def _poll(session, name, url, interval, stop, out, reduce):
    while not stop.is_set():
        t = time.time()
        rec = {"t": t, "src": name}
        try:
            async with session.get(url) as resp:
                text = await resp.text()
            rec.update(reduce(text))
        except Exception as e:  # scrape failures are data, not crashes
            rec["error"] = repr(e)[:200]
        out.write(json.dumps(rec) + "\n")
        try:
            await asyncio.wait_for(stop.wait(), timeout=max(0.0, interval - (time.time() - t)))
        except asyncio.TimeoutError:
            pass


def epp_sample(text: str) -> dict:
    """EPP's own metrics (and process CPU/RSS), histogram buckets dropped."""
    keep = {}
    for name, labels, value in parse(text):
        if name.startswith(("llm_d_", "inference_", "llmd_", "process_")) and not name.endswith("_bucket"):
            key = name + ("{" + ",".join(f"{k}={v}" for k, v in sorted(labels.items())) + "}" if labels else "")
            keep[key] = value
    return {"metrics": keep}


async def run(pods: dict[str, str], epp_url: str, out_path: Path, stop: asyncio.Event,
              pod_interval: float) -> None:
    """Poll every pod every `pod_interval` s and EPP every EPP_INTERVAL_S, until `stop` is set."""
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session:
        with out_path.open("w") as out:
            tasks = [_poll(session, name, f"{base}/metrics", pod_interval, stop, out, pod_sample)
                     for name, base in pods.items()]
            tasks.append(_poll(session, "epp", epp_url, EPP_INTERVAL_S, stop, out, epp_sample))
            await asyncio.gather(*tasks)
