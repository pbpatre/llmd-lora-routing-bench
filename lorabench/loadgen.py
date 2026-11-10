# SPDX-License-Identifier: Apache-2.0
"""Open-loop streaming load client, sharded across processes.

Each request is sent at its scheduled offset regardless of earlier requests
(no coordinated omission). The schedule is split round-robin across worker
processes that share one wall-clock start time. Every record carries:

- t_sched: the scheduled offset, the reference for primary TTFT;
- t_send: when the request headers reached the socket (aiohttp trace hook);
- t_first_token: the first chunk with non-empty delta.content;
- t_end.

Chunks are scanned as bytes; only the usage chunk is JSON-parsed, so a
shard keeps up with tens of thousands of chunks per second.
"""

from __future__ import annotations

import asyncio
import itertools
import json
import multiprocessing as mp
import queue
import time
from pathlib import Path
from typing import Any

import aiohttp

from .arms import ORACLE_HEADER
from .schedule import BASE, Request
from .topology import BASE_MODEL, SERVED_BY_HEADER, raise_nofile

REQUEST_ID_HEADER = "x-request-id"
START_DELAY_S = 1.0  # between the last shard reporting ready and t0

_CONTENT_EMPTY = b'"content":""'
_CONTENT = b'"content":"'
_USAGE = b'"usage":{'


def body_for(req: Request, backend: str) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": BASE_MODEL if req.adapter == BASE else req.adapter,
        "messages": [{"role": "user", "content": req.prompt}],
        "max_tokens": req.max_tokens,
        "min_tokens": req.max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
        "temperature": 0.0,
    }
    if backend == "vllm":
        body["ignore_eos"] = True
    return body


def chunk_has_token(data: bytes) -> bool:
    """True when an SSE data payload carries generated text, not just the role header."""
    i = data.find(_CONTENT)
    while i >= 0:
        if not data.startswith(_CONTENT_EMPTY, i):
            return True
        i = data.find(_CONTENT, i + 1)
    return False


def usage_tokens(data: bytes) -> int | None:
    if _USAGE not in data:
        return None
    try:
        usage = json.loads(data).get("usage") or {}
    except json.JSONDecodeError:
        return None
    return usage.get("completion_tokens")


def oracle_pods(reqs: list[Request], placement: dict[str, list[str]]) -> dict[int, str]:
    """Request index -> oracle pod, round-robin over each adapter's pods in schedule order.

    Assigned once for the whole schedule, so it doesn't depend on sharding.
    """
    cycles = {a: itertools.cycle(pods) for a, pods in placement.items()}
    return {r.idx: next(cycles[r.adapter]) for r in reqs if r.adapter in cycles}


async def _one(session, url, req, t0, backend, timeout_s, run_tag, pod_by_idx):
    sched = t0 + req.t
    delay = sched - time.monotonic()
    if delay > 0:
        await asyncio.sleep(delay)
    rid = f"{run_tag}-{req.idx}"
    rec: dict[str, Any] = {
        "idx": req.idx, "rid": rid, "adapter": req.adapter, "cls": req.cls,
        "t_sched": req.t, "t_send": None, "t_first_byte": None, "t_first_token": None,
        "t_end": None, "chunks": 0, "completion_tokens": None, "status": None,
        "http": None, "served_by": None, "oracle_pod": pod_by_idx.get(req.idx), "error": None,
    }
    headers = {REQUEST_ID_HEADER: rid}
    if rec["oracle_pod"]:
        headers[ORACLE_HEADER] = rec["oracle_pod"]
    ctx = {"rec": rec, "t0": t0}

    async def stream() -> None:
        async with session.post(url, json=body_for(req, backend), headers=headers,
                                trace_request_ctx=ctx) as resp:
            rec["http"] = resp.status
            rec["served_by"] = resp.headers.get(SERVED_BY_HEADER)
            if resp.status != 200:
                rec["status"] = "http_error"
                rec["error"] = (await resp.text())[:300]
                return
            async for raw in resp.content:
                if not raw.startswith(b"data:"):
                    continue
                data = raw[5:].strip()
                if data == b"[DONE]":
                    break
                now = time.monotonic() - t0
                if rec["t_first_byte"] is None:
                    rec["t_first_byte"] = now
                if chunk_has_token(data):
                    rec["chunks"] += 1
                    if rec["t_first_token"] is None:
                        rec["t_first_token"] = now
                n = usage_tokens(data)
                if n is not None:
                    rec["completion_tokens"] = n
            rec["status"] = "ok"

    try:
        await asyncio.wait_for(stream(), timeout=timeout_s)
    except aiohttp.ClientError as e:
        # Before TimeoutError: aiohttp's connect timeouts subclass both.
        rec["status"] = "client_error"
        rec["error"] = repr(e)[:300]
    except asyncio.TimeoutError:
        rec["status"] = "timeout"
    except Exception as e:  # one bad response must not kill the shard
        rec["status"] = "client_error"
        rec["error"] = repr(e)[:300]
    if rec["status"] == "ok" and rec["t_first_token"] is None:
        rec["status"] = "empty_response"
    if rec["t_send"] is None:
        rec["t_send"] = time.monotonic() - t0
    rec["send_lag"] = rec["t_send"] - req.t
    rec["t_end"] = time.monotonic() - t0
    return rec


async def _on_headers_sent(session, ctx, params):
    c = ctx.trace_request_ctx
    if c and c["rec"]["t_send"] is None:
        c["rec"]["t_send"] = time.monotonic() - c["t0"]


async def _lag_probe(stop: asyncio.Event, out: list[float], period: float = 0.01):
    nxt = time.monotonic() + period
    while not stop.is_set():
        await asyncio.sleep(max(0.0, nxt - time.monotonic()))
        out.append(time.monotonic() - nxt)
        nxt += period


async def _shard_main(url, reqs, out_path, backend, timeout_s, run_tag, pod_by_idx, wall_t0):
    trace = aiohttp.TraceConfig()
    trace.on_request_headers_sent.append(_on_headers_sent)
    connector = aiohttp.TCPConnector(limit=0, limit_per_host=0)
    timeout = aiohttp.ClientTimeout(total=None, sock_connect=10)
    # Map the shared wall-clock start onto this process's monotonic clock.
    t0 = time.monotonic() + (wall_t0 - time.time())
    lags: list[float] = []
    stop = asyncio.Event()
    async with aiohttp.ClientSession(connector=connector, timeout=timeout, trace_configs=[trace]) as session:
        probe = asyncio.create_task(_lag_probe(stop, lags))
        tasks = [asyncio.create_task(_one(session, url, r, t0, backend, timeout_s, run_tag, pod_by_idx))
                 for r in reqs]
        with out_path.open("w") as f:
            for fut in asyncio.as_completed(tasks):
                f.write(json.dumps(await fut) + "\n")
        stop.set()
        await probe
    lags.sort()
    return {"loop_lag_p99_s": lags[int(0.99 * (len(lags) - 1))] if lags else None,
            "loop_lag_max_s": lags[-1] if lags else None}


def _shard_entry(args, ready, go, wall_t0, q):
    """Shard process: report ready, wait for the shared t0, replay, report stats or the error."""
    raise_nofile()
    ready.set()
    go.wait()
    try:
        q.put(("ok", asyncio.run(_shard_main(*args, wall_t0.value))))
    except BaseException as e:  # report, don't hang the parent
        q.put(("err", repr(e)))


def run(url: str, reqs: list[Request], out_dir: Path, backend: str, timeout_s: float, run_tag: str,
        shards: int = 4, oracle_placement: dict[str, list[str]] | None = None) -> dict:
    """Replay a schedule across `shards` processes and write requests.jsonl in out_dir.

    t0 is fixed only after every shard has started, so process spawn time never
    shows up as send lag.
    """
    pod_by_idx = oracle_pods(reqs, oracle_placement or {})
    ctx = mp.get_context("spawn")
    q, go, wall_t0 = ctx.Queue(), ctx.Event(), ctx.Value("d", 0.0)
    procs, parts, readies = [], [], []
    for k in range(shards):
        mine = reqs[k::shards]
        part = out_dir / f"requests.shard{k}.jsonl"
        ready = ctx.Event()
        mine_pods = {r.idx: pod_by_idx[r.idx] for r in mine if r.idx in pod_by_idx}
        args = (url, mine, part, backend, timeout_s, run_tag, mine_pods)
        procs.append(ctx.Process(target=_shard_entry, args=(args, ready, go, wall_t0, q), daemon=True))
        parts.append(part)
        readies.append(ready)
    for p in procs:
        p.start()
    for ready, p in zip(readies, procs):
        while not ready.wait(timeout=1.0):
            if p.exitcode is not None:
                raise RuntimeError(f"load shard exited during start with code {p.exitcode}")
    wall_t0.value = time.time() + START_DELAY_S
    go.set()

    results = []
    last_t = max((r.t for r in reqs), default=0.0)
    deadline = time.monotonic() + START_DELAY_S + last_t + timeout_s + 120
    while len(results) < len(procs):
        try:
            results.append(q.get(timeout=2.0))
        except queue.Empty:
            dead = [p.exitcode for p in procs if p.exitcode not in (None, 0)]
            if dead or time.monotonic() > deadline:
                for p in procs:
                    p.terminate()
                raise RuntimeError(f"load shard died or hung (exit codes {dead})")
    for p in procs:
        p.join()
    errs = [r for status, r in results if status == "err"]
    if errs:
        raise RuntimeError(f"load shard failed: {errs}")
    with (out_dir / "requests.jsonl").open("w") as out:
        for part in parts:
            out.write(part.read_text())
            part.unlink()
    stats = [r for _, r in results]
    return {
        "wall_t0": wall_t0.value,
        "shards": shards,
        "loop_lag_p99_s": max(s["loop_lag_p99_s"] or 0 for s in stats),
        "loop_lag_max_s": max(s["loop_lag_max_s"] or 0 for s in stats),
    }
