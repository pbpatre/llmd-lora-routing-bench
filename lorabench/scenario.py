# SPDX-License-Identifier: Apache-2.0
"""Scenario files: topology, engine flags, adapters and traffic mix."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import yaml


@dataclasses.dataclass(frozen=True)
class AdapterGroup:
    prefix: str
    count: int
    rank: int | tuple[int, ...] = 16  # a list cycles over the group's adapters

    def __post_init__(self):
        if isinstance(self.rank, list):
            object.__setattr__(self, "rank", tuple(self.rank))

    def names(self) -> list[str]:
        return [f"{self.prefix}-{i}" for i in range(self.count)]

    def rank_at(self, i: int) -> int:
        return self.rank[i % len(self.rank)] if isinstance(self.rank, tuple) else self.rank

    def max_rank(self) -> int:
        return max(self.rank) if isinstance(self.rank, tuple) else self.rank


@dataclasses.dataclass(frozen=True)
class TrafficClass:
    """A named slice of LoRA traffic spread over one adapter group."""

    name: str
    group: str
    share: float  # fraction of LoRA (non-base) requests
    zipf: float = 0.0  # 0 means uniform within the group


@dataclasses.dataclass(frozen=True)
class Scenario:
    name: str
    description: str
    pods: int
    max_loras: int
    max_cpu_loras: int
    max_lora_rank: int
    max_num_seqs: int
    groups: tuple[AdapterGroup, ...]
    classes: tuple[TrafficClass, ...]
    rate: float  # total offered req/s, base model included
    duration_s: float
    base_share: float
    prompt_tokens: int
    output_tokens: int
    timeout_s: float
    steady_after_s: float
    base_check: bool = True  # DESIGN_LLMD_709.md §6.4 check 6; off where base TTFT is slow by design
    reference: str = ""
    assumed: tuple[str, ...] = ()
    oracle: dict[str, tuple[str, ...]] = dataclasses.field(default_factory=dict)
    capacity_rps: float | None = None  # one-pod capacity used to resolve rate_c
    sim: dict[str, Any] = dataclasses.field(default_factory=dict)

    def adapters(self) -> list[str]:
        return [n for g in self.groups for n in g.names()]

    def group(self, name: str) -> AdapterGroup:
        for g in self.groups:
            if g.prefix == name:
                return g
        raise KeyError(f"scenario {self.name}: unknown adapter group {name!r}")

    def rank_of(self, adapter: str) -> int:
        for g in self.groups:
            names = g.names()
            if adapter in names:
                return g.rank_at(names.index(adapter))
        raise KeyError(adapter)

    def class_of(self, adapter: str) -> str:
        for c in self.classes:
            if adapter in self.group(c.group).names():
                return c.name
        raise KeyError(adapter)


def load(path: str | Path, capacity_rps: float | None = None) -> Scenario:
    """Load and validate a scenario.

    Traffic gives either `rate` (req/s) or `rate_c`, a multiple of the one-pod
    capacity. `rate_c` needs `capacity_rps`: from calibration on GPU, from the
    scenario's `sim.capacity_rps` on the simulator.
    """
    raw = yaml.safe_load(Path(path).read_text())
    eng, tr = raw["engine"], raw["traffic"]
    groups = tuple(AdapterGroup(**g) for g in raw["adapters"])
    if "rate_c" in tr:
        if capacity_rps is None:
            raise ValueError(f"{path}: rate_c needs a one-pod capacity (run calibrate)")
        rate = tr["rate_c"] * capacity_rps
    else:
        rate = tr["rate"]
    sc = Scenario(
        name=raw["name"],
        description=raw.get("description", ""),
        reference=raw.get("reference", ""),
        pods=int(raw["pods"]),
        max_loras=int(eng["max_loras"]),
        max_cpu_loras=int(eng.get("max_cpu_loras", eng["max_loras"])),
        max_lora_rank=int(eng.get("max_lora_rank", max(g.max_rank() for g in groups))),
        max_num_seqs=int(eng.get("max_num_seqs", 256)),
        groups=groups,
        classes=tuple(TrafficClass(**c) for c in tr["classes"]),
        rate=float(rate),
        duration_s=float(tr["duration_s"]),
        base_share=float(tr.get("base_share", 0.0)),
        prompt_tokens=int(tr.get("prompt_tokens", 64)),
        output_tokens=int(tr["output_tokens"]),
        timeout_s=float(tr.get("timeout_s", 120)),
        steady_after_s=float(tr.get("steady_after_s", 60)),
        base_check=bool(raw.get("base_check", True)),
        assumed=tuple(raw.get("assumed", [])),
        oracle={k: tuple(v) for k, v in (raw.get("oracle") or {}).items()},
        capacity_rps=capacity_rps,
        sim=raw.get("sim", {}),
    )
    validate(sc)
    return sc


def validate(sc: Scenario) -> None:
    def fail(msg: str):
        raise ValueError(f"{sc.name}: {msg}")

    total = sum(c.share for c in sc.classes)
    if abs(total - 1.0) > 1e-9:
        fail(f"class shares sum to {total}, want 1")
    if not 0.0 <= sc.base_share < 1.0:
        fail("base_share must be in [0, 1)")
    groups = [c.group for c in sc.classes]
    if len(set(groups)) != len(groups):
        fail("an adapter group is used by two classes")
    for c in sc.classes:
        if c.share <= 0 or sc.group(c.group).count <= 0:
            fail(f"class {c.name} has no traffic or adapters")
    if sc.max_cpu_loras < sc.max_loras:
        fail("max_cpu_loras < max_loras")
    if any(g.max_rank() > sc.max_lora_rank for g in sc.groups):
        fail("adapter rank above max_lora_rank")
    if sc.steady_after_s >= sc.duration_s:
        fail("steady window is empty")
    if sc.oracle:
        pods = {f"pod-{i}" for i in range(sc.pods)}
        if set(sc.oracle) != set(sc.adapters()):
            fail("oracle must place every adapter, and only known adapters")
        for a, placed in sc.oracle.items():
            if not placed or not set(placed) <= pods:
                fail(f"oracle placement for {a} must be a non-empty subset of {sorted(pods)}")
