# llmd-lora-routing-bench

[![tests](https://github.com/pbpatre/llmd-lora-routing-bench/actions/workflows/test.yml/badge.svg)](https://github.com/pbpatre/llmd-lora-routing-bench/actions/workflows/test.yml)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

> Independent project; not affiliated with or endorsed by the llm-d project.

A reproducible benchmark for LoRA-aware request routing on real vLLM pods, driven through [llm-d-router](https://github.com/llm-d/llm-d-router)'s endpoint picker. It compares routing policies (arms) on the same seeded traffic and reports not only latency per traffic class but the mechanism behind it: adapter slot contention, swaps, and packing.

Its first use is [llm-d-router#709](https://github.com/llm-d/llm-d-router/issues/709) (adaptive LoRA placement): it produces the baselines a placement filter must beat, then evaluates filters against the same scenarios. [DESIGN_LLMD_709.md](DESIGN_LLMD_709.md) states that problem, the scenarios and the pre-registered gate.

**First result** ([S1, 4× B200](results/s1-v1.md)): with 4 pods and 2 adapter slots each, stock routing leaves cold adapters at 2–3 s TTFT p99, queued behind hot adapters holding both slots. `lora-residency-scorer` brings that to 0.03 s with 4 adapter swaps per run instead of about 400.

Each run starts 4 engine pods, the router's endpoint picker (EPP) and Envoy, with no Kubernetes needed. It then replays a seeded open-loop schedule. Each run reports:
- TTFT per traffic class;
- what happened underneath: requests blocked at arrival, slot holding, pods per adapter, and adapter loads;
- validity checks.

New to the topic? Start with [PRIMER.md](PRIMER.md), a primer on LoRA routing in llm-d and vLLM and how this benchmark measures it. [DESIGN_LLMD_709.md](DESIGN_LLMD_709.md) is the precise specification.

```
loadgen ─► Envoy ─ext_proc─► EPP (file discovery, one config per arm)
             │ ORIGINAL_DST
             ▼
       pod-0 … pod-3   (vLLM, one GPU each; or llm-d-inference-sim)
```

## Status

- The harness is unit-tested (`make test`).
- S0 calibration and the S1 baseline are complete on 4× B200: [results/s1-v1.md](results/s1-v1.md).
- The S2–S5 baselines are running.

## Requirements

- **GPU runs:** Linux, 4 free GPUs, and Python 3.10+. `tools/setup-node.sh` installs the rest:
  - vLLM 0.31.0, in a venv;
  - Llama-3.1-8B-Instruct;
  - synthetic adapters;
  - Envoy 1.33.2;
  - `epp`, built from llm-d-router `1f7f0d90`;
  - `llm-d-inference-sim` v1.0.0.
- **Simulator runs:** the `epp`, `envoy` and `llm-d-inference-sim` binaries, in `.cache/bin/` or set with `LORABENCH_EPP`, `LORABENCH_ENVOY` and `LORABENCH_SIM`.

## Quick start (GPU node)

```bash
WORK=$HOME/llmd-lora-routing-bench-work ./tools/setup-node.sh   # about 1 h; uses Docker if present
source $HOME/llmd-lora-routing-bench-work/env.sh
export LORABENCH_GPUS=0,1,2,3                       # optional; defaults to 0-3

make test             # unit tests
make gpu-smoke        # S1 on simulator pods: checks the plumbing, no GPU time (runs-smoke/)
make quick-s1         # S1 on vLLM: stock vs residency-w4, one seed (runs-quick/)
make calibrate        # S0: one-pod capacity per output length, adapter load costs
make s1-set CALIBRATION=runs/s0-calibration/<dir>/calibration.json
make first-set CALIBRATION=...   # S2, S3, S5; with s1-set, the first set (DESIGN_LLMD_709.md §9)
make report           # per-seed table over runs/
make gate-baselines CALIBRATION=...      # the gate's reference arms, seeds 1-5
make gate-runs ARM=<arm> SEEDS="1 2" ...  # tune a candidate; then SEEDS="3 4 5" with settings frozen
make gate ARM=<arm>   # #709 verdict on gate seeds 3-5 (DESIGN_LLMD_709.md §8)
```

**Calibrate on the GPU you benchmark.** `make calibrate` measures one pod's capacity C (the highest rate before TTFT p99 jumps) and its unloaded request time at 256 and 1,024 output tokens. S3 and S4 give their rate as `rate_c`, a multiple of C, with class shares solved so that each hot adapter needs 1.15 pods and the giant 2.25. On B200, C₁₀₂₄ = 14 req/s, so S3 runs at 44.8 req/s with 16.1 req/s per hot adapter. S1 and S2 use #3392's fixed rates, so for them calibration feeds only the checks and the feasibility bound. [PRIMER.md §9](PRIMER.md#9-calibration) explains how the loads follow from C.

To run two jobs on one node, give the second its own GPUs, ports and run directory:

```bash
LORABENCH_GPUS=7 LORABENCH_PORT_OFFSET=1000 make calibrate RUNS=runs-cal
```

## Simulator only

```bash
make smoke-sim        # S1, S2, S3, S4 x 3 arms x 1 seed, shortened; writes runs-smoke/
```

The simulator checks the harness and the qualitative regimes. Its latencies never count as results (see [Known limits](#known-limits)).

## Layout

```
scenarios/         S0 calibration, S1 fits, S2 oversubscribed, S3 two hot, S4 one giant, S5 Zipf
lorabench/
  scenario.py      scenario files: topology, engine flags, adapters, traffic mix
  schedule.py      seeded open-loop schedule; every arm replays the same one
  jsonl.py         JSON-lines read/write
  arms.py          one EPP config per arm (stock, affinity, residency, oracle, random)
  topology.py      starts and stops pods, EPP and Envoy
  loadgen.py       sharded open-loop streaming client
  scrape.py        pod /metrics every 100 ms, EPP every 5 s
  collect.py       vLLM adapter loads from logs; Envoy access-log join
  analyze.py       per-run summary: latency per class, mechanism diagnostics, validity
  compare.py       cross-run table and the #709 gate
  calibrate.py     S0 capacity sweep and load-cost probes
  runner.py        one run end to end; balanced arm order for matrices
tools/
  setup-node.sh    one-time GPU node setup
  make_adapters.py synthetic PEFT adapters (random weights)
tests/             unit tests on synthetic data
```

## Outputs

Each run writes `<runs>/<scenario>/<run-id>/`. Results go to `runs/`; `runs-smoke/` and `runs-quick/` are cleared on every smoke or quick run. A report refuses two valid runs of the same scenario, arm, seed and replicate.

| File | Contents |
| --- | --- |
| `meta.json` | scenario, arm, seed, schedule sha256, binary hashes, offered rates |
| `schedule.jsonl` | the replayed schedule |
| `requests.jsonl` | one record per request: timestamps, status, serving pod |
| `scrape.jsonl` | pod and EPP metrics, every 100 ms |
| `access.jsonl` | Envoy access log |
| `loads.jsonl` | vLLM adapter loads and activations (vLLM only) |
| `logs/`, `config/` | process logs and rendered configs |
| `summary.json` | the analysis, including `validity` |

## Known limits

- **The simulator** has no adapter load latency and no CPU tier. It evicts randomly, and its dequeue order differs from vLLM's. Use it to validate the harness, never to compare TTFT across arms.
- **Parameters marked `assumed:`** in each scenario aren't stated in #3392 yet.
- **Not checked automatically yet:**
  - that the residency scorer's response hook fires;
  - the EPP plugin-latency bound;
  - GPU throttling.

  Each run lists these under `validity.not_checked`.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). The Python package is `lorabench`; run it with `python -m lorabench`.

## License

Apache 2.0, see [LICENSE](LICENSE).
