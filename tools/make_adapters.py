#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Write synthetic PEFT LoRA adapters (random weights, fp16, all linear layers).

The weights are random: routing and slot behaviour don't depend on what an
adapter has learned, and every report says the adapters are synthetic.

  python3 tools/make_adapters.py --base-config /path/to/model/config.json \
      --scenario scenarios/s1-fits.yaml --out /data/adapters [--seed 0]

Needs numpy and safetensors (both ship with vLLM).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lorabench.scenario import load  # noqa: E402

TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def shapes(cfg: dict) -> dict[str, tuple[int, int]]:
    """(in_features, out_features) per target module for a Llama-style model."""
    h = cfg["hidden_size"]
    inter = cfg["intermediate_size"]
    heads = cfg["num_attention_heads"]
    kv = cfg.get("num_key_value_heads", heads)
    hd = cfg.get("head_dim", h // heads)
    return {
        "q_proj": (h, heads * hd), "k_proj": (h, kv * hd), "v_proj": (h, kv * hd), "o_proj": (heads * hd, h),
        "gate_proj": (h, inter), "up_proj": (h, inter), "down_proj": (inter, h),
    }


def write_adapter(out: Path, cfg: dict, rank: int, base_name: str, rng: np.random.Generator) -> int:
    from safetensors.numpy import save_file

    out.mkdir(parents=True, exist_ok=True)
    tensors = {}
    for layer in range(cfg["num_hidden_layers"]):
        for mod, (fin, fout) in shapes(cfg).items():
            block = "self_attn" if mod.endswith(("q_proj", "k_proj", "v_proj", "o_proj")) else "mlp"
            key = f"base_model.model.model.layers.{layer}.{block}.{mod}"
            tensors[f"{key}.lora_A.weight"] = (rng.standard_normal((rank, fin)) / np.sqrt(fin)).astype(np.float16)
            tensors[f"{key}.lora_B.weight"] = (rng.standard_normal((fout, rank)) * 1e-3).astype(np.float16)
    save_file(tensors, str(out / "adapter_model.safetensors"))
    (out / "adapter_config.json").write_text(json.dumps({
        "peft_type": "LORA", "task_type": "CAUSAL_LM", "base_model_name_or_path": base_name,
        "r": rank, "lora_alpha": 2 * rank, "lora_dropout": 0.0, "bias": "none",
        "target_modules": list(TARGETS), "fan_in_fan_out": False, "inference_mode": True,
    }, indent=1))
    return sum(t.nbytes for t in tensors.values())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-config", type=Path, required=True, help="the base model's config.json")
    ap.add_argument("--base-name", default="unsloth/Llama-3.1-8B-Instruct")
    ap.add_argument("--scenario", type=Path, action="append", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    cfg = json.loads(a.base_config.read_text())
    done = set()
    for path in a.scenario:
        sc = load(path, capacity_rps=1.0)
        for name in sc.adapters():
            if name in done:
                continue
            rank = sc.rank_of(name)
            dest = a.out / name
            marker = dest / "adapter_config.json"
            if marker.exists() and json.loads(marker.read_text())["r"] == rank:
                done.add(name)
                continue
            rng = np.random.default_rng([a.seed, sum(map(ord, name))])
            n = write_adapter(dest, cfg, rank, a.base_name, rng)
            print(f"{name}: rank {rank}, {n / 1e6:.0f} MB")
            done.add(name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
