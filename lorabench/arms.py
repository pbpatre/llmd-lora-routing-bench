# SPDX-License-Identifier: Apache-2.0
"""Router arms: one EPP EndpointPickerConfig per arm.

Every arm except `random` starts from the default router (`stock`), as in
#3392, and adds its LoRA plugin on top.
"""

from __future__ import annotations

from typing import Any, Callable

import yaml

from .scenario import Scenario

ORACLE_HEADER = "x-bench-pod"

# EPP injects utilization-detector as a Filter into every profile when no
# saturation detector is configured (config/loader/defaults.go). It drops pods
# whose waiting queue exceeds 5, whose KV use exceeds 0.8, or whose metrics are
# older than 200 ms. It is part of the default router, so every arm keeps it;
# declaring it pins and records those values.
SATURATION_DETECTOR = {
    "name": "utilization-detector",
    "type": "utilization-detector",
    "parameters": {"queueDepthThreshold": 5, "kvCacheUtilThreshold": 0.8,
                   "metricsStalenessThreshold": "200ms", "headroom": 0.0},
}
POD_LABEL = "bench/pod"

Plugin = dict[str, Any]
Arm = tuple[list[Plugin], list[Plugin]]  # (plugin definitions, profile references)


def _with(arm: Arm, plugin: Plugin, ref: Plugin) -> Arm:
    return arm[0] + [plugin], arm[1] + [ref]


def stock(sc: Scenario, base_model: str, prefix: bool = True) -> Arm:
    """The default scorers: queue 2, KV-utilization 2, prefix 3."""
    plugins = [{"type": "queue-scorer"}, {"type": "kv-cache-utilization-scorer"}]
    refs = [{"pluginRef": "queue-scorer", "weight": 2}, {"pluginRef": "kv-cache-utilization-scorer", "weight": 2}]
    if prefix:
        plugins.append({"type": "prefix-cache-scorer"})
        refs.append({"pluginRef": "prefix-cache-scorer", "weight": 3})
    return plugins, refs


def stock_noprefix(sc: Scenario, base_model: str) -> Arm:
    return stock(sc, base_model, prefix=False)


def affinity_w2(sc: Scenario, base_model: str) -> Arm:
    return _with(stock(sc, base_model), {"type": "lora-affinity-scorer"},
                 {"pluginRef": "lora-affinity-scorer", "weight": 2})


def residency_w4(sc: Scenario, base_model: str) -> Arm:
    plugin = {"type": "lora-residency-scorer",
              "parameters": {"baseModels": [base_model], "maxCpuLoras": sc.max_cpu_loras}}
    return _with(stock(sc, base_model), plugin, {"pluginRef": "lora-residency-scorer", "weight": 4})


def residency_w4_rr2(sc: Scenario, base_model: str) -> Arm:
    """The #3392 README pairing for overloaded hot pods."""
    return _with(residency_w4(sc, base_model), {"type": "running-requests-size-scorer"},
                 {"pluginRef": "running-requests-size-scorer", "weight": 2})


def oracle(sc: Scenario, base_model: str) -> Arm:
    """The scenario's hand placement as a dominant preference on top of residency.

    The load client names the pod in ORACLE_HEADER; the utilization filter can
    still drop it, so this is a strong preference, not a pin.
    """
    if not sc.oracle:
        raise ValueError(f"{sc.name}: arm 'oracle' needs an 'oracle' placement in the scenario")
    plugin = {"type": "header-label-affinity-scorer", "name": "oracle",
              "parameters": {"headerName": ORACLE_HEADER, "labelKey": POD_LABEL, "stampResponseHeader": False}}
    return _with(residency_w4(sc, base_model), plugin, {"pluginRef": "oracle", "weight": 100})


def random_floor(sc: Scenario, base_model: str) -> Arm:
    """No scorers, random picker: the floor."""
    return [{"type": "random-picker"}], [{"pluginRef": "random-picker"}]


ARMS: dict[str, Callable[[Scenario, str], Arm]] = {
    "stock": stock,
    "stock-noprefix": stock_noprefix,
    "affinity-w2": affinity_w2,
    "residency-w4": residency_w4,
    "residency-w4-rr2": residency_w4_rr2,
    "oracle": oracle,
    "random": random_floor,
}

# Alpha plugins need --allow-experimental-plugins.
EXPERIMENTAL = {"residency-w4", "residency-w4-rr2", "oracle"}


def epp_config(arm: str, sc: Scenario, base_model: str, endpoints_path: str) -> str:
    plugins, refs = ARMS[arm](sc, base_model)
    discovery = {"name": "file-disc", "type": "file-discovery",
                 "parameters": {"path": endpoints_path, "watchFile": False}}
    if not any(p["type"].endswith("-picker") for p in plugins):
        plugins, refs = _with((plugins, refs), {"type": "max-score-picker"}, {"pluginRef": "max-score-picker"})
    plugins = [discovery, *plugins, SATURATION_DETECTOR]
    refs = [{"pluginRef": SATURATION_DETECTOR["name"]}, *refs]
    cfg = {
        "apiVersion": "llm-d.ai/v1",
        "kind": "EndpointPickerConfig",
        "plugins": plugins,
        "schedulingProfiles": [{"name": "default", "plugins": refs}],
        "flowControl": {"saturationDetector": {"pluginRef": SATURATION_DETECTOR["name"]}},
        "dataLayer": {"discovery": {"endpoints": {"pluginRef": "file-disc"}}},
    }
    return yaml.safe_dump(cfg, sort_keys=False)


def endpoints_file(pods: list[tuple[str, str, int]]) -> str:
    """File-discovery endpoints from (name, ip, port); each pod is labelled with its name."""
    return yaml.safe_dump({"endpoints": [
        {"name": n, "address": ip, "port": str(port), "labels": {POD_LABEL: n}} for n, ip, port in pods
    ]}, sort_keys=False)
