# SPDX-License-Identifier: Apache-2.0
"""Seeded open-loop request schedules shared by every arm of a scenario.

A schedule is a pure function of (scenario, seed): every arm replays the same
requests at the same offsets, so arm comparisons are paired by seed.
"""

from __future__ import annotations

import dataclasses
import hashlib
import zlib
from pathlib import Path

import numpy as np

from . import jsonl
from .scenario import Scenario

BASE = "__base__"

# A fixed vocabulary keeps prompts tokenizer-stable across engines; each
# request gets a unique random prompt so prefix-cache hits stay negligible.
_VOCAB = (
    "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima "
    "mike november oscar papa quebec romeo sierra tango uniform victor whiskey "
    "xray yankee zulu red orange yellow green blue indigo violet black white "
    "north south east west river mountain forest desert ocean island valley"
).split()


@dataclasses.dataclass(frozen=True)
class Request:
    idx: int
    t: float  # offset from run start, seconds
    adapter: str  # BASE for base-model requests
    cls: str
    prompt: str
    max_tokens: int


def _weights(n: int, zipf: float) -> np.ndarray:
    if zipf <= 0:
        return np.full(n, 1.0 / n)
    w = 1.0 / np.arange(1, n + 1) ** zipf
    return w / w.sum()


def target_probabilities(sc: Scenario) -> dict[str, float]:
    """Probability that one request targets each adapter (and the base model)."""
    probs = {BASE: sc.base_share} if sc.base_share > 0 else {}
    lora = 1.0 - sc.base_share
    for c in sc.classes:
        names = sc.group(c.group).names()
        for name, w in zip(names, _weights(len(names), c.zipf)):
            probs[name] = lora * c.share * float(w)
    return probs


def generate(sc: Scenario, seed: int) -> list[Request]:
    rng = np.random.default_rng([seed, zlib.crc32(sc.name.encode())])
    probs = target_probabilities(sc)
    targets = list(probs)
    p = np.array([probs[k] for k in targets])
    p /= p.sum()
    cls_of = {k: "base" if k == BASE else sc.class_of(k) for k in targets}

    out: list[Request] = []
    t = rng.exponential(1.0 / sc.rate)
    while t < sc.duration_s:
        target = targets[rng.choice(len(targets), p=p)]
        words = rng.choice(_VOCAB, size=sc.prompt_tokens)
        idx = len(out)
        out.append(Request(idx=idx, t=round(t, 6), adapter=target, cls=cls_of[target],
                           prompt=f"[{idx}] " + " ".join(words), max_tokens=sc.output_tokens))
        t += rng.exponential(1.0 / sc.rate)
    return out


def write(reqs: list[Request], path: Path) -> str:
    """Write a schedule and return its sha256, recorded per run."""
    path.parent.mkdir(parents=True, exist_ok=True)
    body = jsonl.write(path, [dataclasses.asdict(r) for r in reqs])
    return hashlib.sha256(body.encode()).hexdigest()
