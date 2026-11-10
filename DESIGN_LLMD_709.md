# LoRA routing baseline benchmark: design

**Status:** v2.3, 10 Nov 2026. v2.3 changes, made before any candidate filter ran, are listed under §8 and §12.

**Context:** [llm-d/llm-d-router#709](https://github.com/llm-d/llm-d-router/issues/709) (adaptive LoRA placement). This benchmark produces the baselines that a placement filter must beat. The gate proposed on #709: the filter must beat `lora-residency-scorer` (#3392, by atantawi) in at least one mix without losing in the others. §8 makes that rule exact.

**Pinned router:** llm-d-router `1f7f0d90`, the #3392 merge commit (25 Oct 2026), which contains `lora-residency-scorer`. Local checkouts older than this lack the scorer.

## 1. Questions

1. **Do #3392's regimes reproduce on an independent rig?** atantawi's rig was 4×H100 with vLLM v0.31.0. Our GPUs are H200/B200, so we claim the regimes reproduce, not the numbers.
2. **For each in-tree arm and each regime,** how large are starvation, overload and churn, per class?
3. **Mechanism.** Report measured values, not labels:
   - each class's arrival rate × service time;
   - the time fraction each adapter has zero running requests on each pod;
   - pods per adapter;
   - blocked-at-arrival;
   - whether a safe layout was *feasible* (computed offline) and whether the arm *reached* one.
4. **Headroom.** What does a hand-written oracle placement achieve relative to residency? If the oracle can't beat residency, a placement filter can't either.
5. **Noise,** so the #709 gate is decided by a rule registered before any filter exists.

**Non-goals:** weight tuning, prefill/decode disaggregation, multiple EPP replicas, autoscaling, engine changes.

## 2. Topology (no Kubernetes)

```
loadgen (P processes, sharded seeded schedule) ─► Envoy :8080 ─ext_proc─► EPP (file discovery)
                                                  │ ORIGINAL_DST via x-gateway-destination-endpoint
                                                  ▼
                        pod-0 … pod-3  (vLLM, one GPU each; llm-d-inference-sim locally)
```

**EPP:**
- File discovery (`runner.go:289`).
- Flags: `--secure-serving=false`, `--tracing=false` (no trace collector), `--refresh-metrics-interval=50ms`, and `--allow-experimental-plugins` for arms that use alpha plugins.
- The EPP binary is built from the pin.

**Envoy:**
- Copied from the `llm-d-router-standalone` chart: v1.33.2, `FULL_DUPLEX_STREAMED` request and response bodies, `response_header_mode: SEND`, `message_timeout: 1000s`.
- Explicit settings: route `timeout: 0`, `stream_idle_timeout: 300s`, retries off, circuit breakers raised to 100k, h2 `max_concurrent_streams` raised.
- Adds response header `x-served-by: %UPSTREAM_HOST%`.
- Writes a JSON access log keyed by the client's `x-request-id`, with `%UPSTREAM_HOST%`, `%RESPONSE_FLAGS%` and upstream start time.
- **The access log is the authoritative attribution;** the response header is a cross-check.

**Addresses:**
- On Linux, each pod binds its own loopback IP (127.0.0.2–.5), because EPP's subset matching is IP-only (`requestcontrol/candidates.go:142-147`).
- On macOS, all pods share 127.0.0.1 with distinct ports, since extra loopback IPs need root there.

**Readiness (default, "primed"):**
1. `/health` returns 200 on every pod.
2. `/v1/models` lists every adapter.
3. One base-model request has gone directly to each pod, so `vllm:lora_requests_info` exists (#3361).
4. EPP has scraped every pod for longer than its staleness threshold.

A "cold" variant (no priming), to match #3392's S1 exactly, is not implemented yet (§12).

**Per-run hygiene (GPU):**
- Restart all pods and EPP.
- Pre-read every adapter file into the page cache. "Storage" then means page cache, as it partly did in #3392. True storage cost is measured once, in S0, with caches dropped.
- Persist `TRITON_CACHE_DIR` and the vLLM compile cache, and prewarm once per pin. That removes the first-use LoRA kernel compile from runs.
- Set `max_num_seqs` and `max_num_batched_tokens` explicitly.
- Pin the frontend to Python (`VLLM_USE_RUST_FRONTEND` unset).
- Turn on GPU persistence mode.
- Log clocks, throttle reasons, power and temperature at 1 s, and flag throttled runs.
- Gate runs execute one at a time, never two 4-GPU runs in parallel.

**Backends:**

| Backend | Fidelity | Allowed use |
| --- | --- | --- |
| `vllm` | real scheduler, LRU GPU/CPU tiers, load latency | all numbers |
| `sim` (llm-d-inference-sim v1.0.0) | real `max_loras` slot gating, but: no load latency, random eviction, no CPU tier, loaded-adapter-first dequeue (vLLM is FCFS-with-skip), Rust-frontend-style LoRA labels | harness validation and regime smoke tests only; **never compare sim TTFT across arms** |

## 3. Scenarios

Model: Llama-3.1-8B-Instruct, `unsloth/Llama-3.1-8B-Instruct@4699cc75` as in #3392. vLLM v0.31.0 for comparability.

Adapters: synthetic fp16, all linear layers, random weights (disclosed), registered with `--lora-modules`. Every request decodes exactly L tokens (`ignore_eos`). Base share is 20% unless noted.

| ID | Regime | Pods × `max_loras` | `max_cpu_loras` | Adapters | Traffic | Replicates |
| --- | --- | --- | --- | --- | --- | --- |
| **S0** | calibration | 1 × 1 | 2 | 1 + 20 probes (5 per rank 8/16/32/64) | rate sweep at L = 256 and 1024 → one-pod capacity `C_L` = the highest rate with TTFT p99 ≤ 4× unloaded and ≥ 98% complete (the TTFT knee); load cost storage→CPU and CPU→GPU per rank (caches dropped and warm) | #3392 probes |
| **S1** | partition exists | 4 × 2 | 16 | 2 hot (90% of LoRA) + 4 cold, rank 16* | 12 req/s, L = 256, 180 s | starvation study, comment 6182873062 |
| **S2** | more never-idle than slots | 4 × 2 | 16* | 8 warm (90%) + 4 cold, rank 16* | 20 req/s, L = 1024 | follow-up 1b, comment 6218627904 |
| **S3** | two hot adapters at ~1.15 pods each | 4 × 2 | 16* | 2 hot + 4 cold* | 0.8 × 4 × `C_1024` total; hot adapters at 1.15 × `C_1024` each; cold timeout 120 s | follow-up 1a, rate rescaled to our GPU |
| **S4** | one giant | 4 × 2 | 16 | 1 giant at 2.25 × `C_1024` + 5 cold | ≤ 0.8 of cluster capacity, L = 1024 | **new**: the case named on #709, not yet measured |
| **S5** | Zipf long tail (cross-check) | 4 × 4 | 4 | 32 adapters, ranks 8/16/32/64, Zipf(1.0) | 8 req/s, L = 256, `--max-lora-rank 64` | one cell of #3392's main grid |

In S2, "cold" is a label carried over from #3392, not a regime: at 0.4 req/s × ~10 s E2E, each cold adapter averages ~4 requests in flight. So all 12 adapters hold slots against 8, and the feasibility bound reports that.

\* These parameters are not stated in the #3392 comments for those studies. They are **assumed**, listed in each scenario's `assumed:` field, and asked about on #709.
- S1 and S2 keep #3392's absolute mixes, because their regime depends on slots, not GPU speed.
- S3 and S4 are scaled to our measured `C_1024`.
- **Deferred** until after the first numbers: the full S5 grid, an 8-pod scale-out, and a second vLLM pin.

## 4. Arms

Every arm uses the default base scorers (queue 2, KV-utilization 2, prefix 3) and the max-score picker.

| Arm | Added | Scenarios |
| --- | --- | --- |
| `stock` | none | all |
| `stock-noprefix` | prefix scorer removed (control, see §6.2) | S1 |
| `affinity-w2` | `lora-affinity-scorer` w2 | all |
| `residency-w4` | `lora-residency-scorer` w4, `baseModels` set, `maxCpuLoras` = pod value | all |
| `residency-w4-rr2` | + `running-requests-size-scorer` w2 (the #3392 README pairing for hot-pod overload) | S3, S4 |
| `oracle` | a hand-written placement per scenario, applied through `header-label-affinity-scorer` at weight 100 (pod label `bench/pod`, request header `x-bench-pod` chosen by the load client), on top of residency-w4 | S2, S3, S4 |
| `random` | random picker only (floor) | sim smoke only |
| *(later)* `placement-*` | the #709 filter, with an allowed-set source (`hrw` or `solver`) | all |

**Notes on the oracle:**
- It is a measurement oracle, not a feature.
- For adapters placed on k > 1 pods, the client picks among the k pods round-robin.
- Oracle placements are written down per scenario before any run.

## 5. Workload

- **Schedule:** one per (scenario, seed): a seeded Poisson arrival time, class, adapter, a random-word prompt that starts with a unique nonce, and L. Every arm replays the identical file, so comparisons are paired by seed.
- **Requests:** streamed chat completions, as in #3392, with `stream_options.include_usage`.
- **Client timeout:** 120 s for the whole request. Every request still in flight is drained until last arrival + 120 s, and windows are assigned by **arrival** time.
- **Seeds and order:** 3 seeds. Arm order within a seed follows a Latin square across seeds.

## 6. Measurement

### 6.1 Client

The load client shards the schedule across 4 processes, with a shared wall-clock t0.

**Timestamps per request:**
- scheduled time;
- send, stamped when the request headers reach the socket (an aiohttp trace hook);
- first content token (the first chunk with non-empty `delta.content`);
- end.

**Token counts** come from `usage`.

**SSE handling:** chunks are scanned as bytes; only the usage chunk is JSON-parsed.

**Validity probes:**
- Event-loop lag, sampled by a 10 ms ticker. The p99 must be < 5 ms.
- Send lag (send − scheduled). The p99 must be < 50 ms.

**TTFT:**
- **Primary:** first token − *scheduled* time, which includes any client-side queueing.
- **Secondary:** first token − send.
- ITL = (end − first token) / (L − 1).

**Generator self-test** (not implemented yet): run the client against the simulator with near-zero latency at 2× the S3 rate, and require both probes to pass. Until then, every run checks both probes.

### 6.2 Router and data plane

- **EPP:** metrics every 5 s, including prefix-cache hit metrics and plugin latency. Also EPP CPU and RSS.
- **Prefix-cache scorer:** `hashing.go:121` seeds its hash with `TargetModel`. If EPP tokenizes requests, the shared chat-template header makes the prefix scorer adapter-affine in `stock`. The `stock-noprefix` control and the hit metric quantify this.
- **Envoy:** `/stats` before and after each run (pending overflow, `rq_timeout`, ext_proc errors).
- **Residency scorer:** check that its response hook fires. This is a validity check (§6.4).

### 6.3 Engine

- **Pods:** scraped every 100 ms. Record `vllm:lora_requests_info`, using the series with the largest value; that value is the engine's wall clock, so use it as the timestamp. Also record `num_requests_running`/`waiting`, KV usage, and `/metrics` size and scrape latency.
- **vLLM loads:**
  - `VLLM_LOGGING_CONFIG_PATH` sets the root logger to INFO and `vllm.lora.model_manager` to DEBUG, with ISO timestamps.
  - `"Adding lora. Model id: …, int id: …"` is a storage→CPU load. It is logged only on a CPU-cache miss.
  - `"Activating LoRA. int id: …, slot index: …"` is a CPU→GPU activation. It is logged only on a GPU-slot miss.
  - Map int ids to names from `--lora-modules` order. Verify the strings and the mapping for each pin, and drop pre-ready lines.
- **Simulator:** an activation proxy instead: an adapter entering a pod's running set, at 100 ms resolution.
- **GPU:** nvidia-smi at 1 s.

### 6.4 Validity (a failing run is excluded and re-run)

1. Loop-lag p99 < 5 ms and send-lag p99 < 50 ms.
2. Every successful request decoded L tokens.
3. Every request is attributed through the access log, its upstream is a pod in the pod set, and it agrees with `x-served-by`.
4. No errors other than client timeouts. Envoy shows no overflow and no ext_proc errors.
5. EPP was up for the whole run, and its plugin latency p99 is below a fixed bound (EPP not saturated).
6. Base-model TTFT p99 ≤ 3 × the S0 unloaded p99, so the pool is not overloaded. It is off (`base_check: false`) in S3 and S4, which are overloaded by design, and in S5, where storage loads stall every request in a vLLM batch, so base TTFT measures thrash rather than overload. Those runs report it instead.
7. The residency scorer's response hook fired (residency arms only).
8. No GPU throttling.

## 7. Metrics (per class; per adapter as supporting detail only)

| Metric | Definition |
| --- | --- |
| TTFT p50 / p99 | nearest-rank. A request that times out before its first token counts as +inf. |
| TTFT mean | over completed requests, reported with the timeout count |
| > 5 s share | share of the class's requests with TTFT > 5 s |
| timeouts, errors | counts per class |
| ITL p50 / p99 | per request (end − first) / (L − 1) |
| work completed | completed requests and output tokens per class. Throughput can't compare open-loop arms, but starving arms shed work. |
| loads, activations | vLLM log counts; sim proxy |
| pods per adapter | distinct pods serving the adapter in the steady window |
| imbalance | busiest pod's mean in-flight ÷ the pod mean (Little's law, #3392) |
| slot-holding | per (pod, adapter): the share of time with ≥ 1 *running* request. Running means [first token − prefill estimate, end] as seen by the client. The scraped running set is recorded but not yet cross-checked. Also counts the gaps with zero running requests; a single gap frees a slot. Windows are 30 s. |
| blocked-at-arrival | share of cold requests that, when they reach the pod (Envoy upstream start), find ≥ `max_loras` *other* adapters holding slots there. Base-model requests are excluded. |
| feasibility | offline, from the schedule and S0 service times. Does an assignment exist in which slots (`pods × slots − slot-holding adapters ≥ tail concurrency`) **and** compute (per-pod load ≤ `C_L`) both fit? |
| queue trend | median vLLM waiting queue per pod in the first and last 30 s of the steady window. A pod whose queue grows is past capacity, so the run has no steady state and its p99 depends on run length. |
| reached layout | per 30 s window: does every pod that hosts a non-slot-holding adapter have fewer than `max_loras` slot-holding adapters? |
| worst adapter | the adapter with the highest TTFT p99. Jain's index was dropped: dividing by a constant doesn't change it, and it was not informative. |

## 8. Statistics and the gate (pre-registered)

**Runs:**
- 3 seeds per (scenario, arm), paired.
- **A/A:** `residency-w4` is run twice on every seed in each gate scenario. One copy is at the start of the block and one at the end.
- **The noise band** for each class's p99 is the larger of:
  - that class's observed A/A spread;
  - #3392's replicate spread: 0.70–1.41× for p99, 0.92–1.09× for loads.
- **p99 differences below 0.25 s never fail a guard or count as a win.** Starvation is measured in seconds; a ratio test on 30 ms latencies flags noise.
- **Proportions and counts use absolute margins,** not ratios.

v2.2 widened every band by the largest A/A spread of *any* class. With the bimodal outcomes seen in S2–S4 (§12), that made bands of 0–800×, so v2.3 makes them per class.

**One primary metric per scenario:**

| Scenario | Primary | Win | Non-inferiority guard |
| --- | --- | --- | --- |
| S1 | cold TTFT p99 | not a win target: residency already ≈ 0.2 s | cold p99 ≤ residency × 1.41; cold > 5 s ≤ +1 pp |
| S2 | cold > 5 s share | ≥ 10 pp lower than residency in every seed | warm > 5 s ≤ +5 pp; total timeouts ≤ residency + 1% of requests |
| S3 | hot TTFT p99 | non-inferiority only | cold > 5 s ≤ +1 pp; timeouts no higher |
| S4 | giant TTFT p99 | outside the band in every seed | cold > 5 s ≤ +1 pp |

**Gate for a placement arm:** a win in at least one of S2 or S4, and every guard holds in every scenario, against each scenario's references: `residency-w4` in S1 and S2, plus `residency-w4-rr2` in S3 and S4, where it runs. Any missing or invalid run for a required (scenario, arm, seed) fails the gate.

**Tuning is kept separate:** filter parameters (k sizing, cooldown) are tuned on seeds 1–3, and the gate is decided on seeds 4–6.

## 9. First credible set

- **S0.**
- **S1, S2 and S3:** {stock, affinity-w2, residency-w4} × 3 seeds, plus 3 A/A runs each, plus `stock-noprefix` on S1. That's 3 × (9 + 3) + 3 = 39 runs.
- **One S5 cell:** {affinity-w2, residency-w4} × 3 = 6 runs.
- **Total:** about 45 runs × ~8 min (180 s of load, up to 120 s of drain, and restart) ≈ 6 GPU-hours, run serially.

**Second set, before any filter code:** S4, `residency-w4-rr2`, and the oracle arm on S2, S3 and S4. If the oracle can't beat residency, report that on #709 before building the filter.

## 10. Simulator validation

Simulator flags are set from the GPU S0 knee (`--max-num-seqs`, `--time-factor-under-load`), so compute contention can appear. Until S0 exists, they use stand-in values.

**Pass criteria:**
1. Every scenario and arm runs end to end, and validity checks 1–5 pass.
2. Schedules are byte-identical across arms (sha256 recorded).
3. **S1:** `stock` shows high blocked-at-arrival for cold requests; `residency-w4` shows little.
4. **S2:** cold starvation appears under every arm.
5. **S3:** residency packs (1 pod per hot adapter).
6. The analysis reproduces hand-computed values on synthetic fixtures (unit tests).

## 11. Deliverables

1. **This repo,** public, linked from #709. It holds scenarios, arms, the schedule generator, the load client, launchers, scrapers, analysis, the gate evaluator, and `make` targets.
2. **Per-scenario reports** on #709: per-seed tables, mechanism diagnostics, and the list of assumed parameters.
3. **The `test/perf` scenario promised on #709,** built on `inference-perf` and the simulator so it fits the nightly suite. It follows the first numbers; that sequencing will be stated on #709.

## 12. Implementation notes

**The default saturation filter is pinned, not removed.** When no saturation detector is configured, EPP injects `utilization-detector` as a *Filter* into every profile (`config/loader/defaults.go`, `ensureSaturationDetector`). It drops pods whose:
- waiting queue exceeds 5,
- KV use exceeds 0.8, or
- metrics are older than 200 ms.

It is part of the default router, and #3392 ran with it, so every arm keeps it. `arms.py` declares it with those values, so they are pinned and recorded. Two consequences:
- **The oracle is a strong preference, not a hard pin.** The filter can drop the oracle's pod.
- **Requests that queue for a LoRA slot raise a pod's waiting count.** So a pod where cold requests are starving can be filtered out. That interaction is part of what the baselines measure.

**Adapters are registered statically.** vLLM preloads every `--lora-modules` adapter into the CPU tier at startup (`init_static_loras` → `add_lora`).
- So when `max_cpu_loras` ≥ the adapter count (S1, S2, S3, S4), storage→CPU loads during a run are about 0 by construction. The churn metric there is CPU→GPU activations.
- Storage loads appear only where the CPU tier is smaller than the catalogue (S5).
- #3392's starvation study reported "loads" 24/17/6 with 6 adapters and a 16-adapter CPU tier. So it either registered adapters lazily (resolver) or counted activations. **Asked on #709.** A lazy-resolver mode is not implemented yet.

**Log parsing.** vLLM V1 logs LoRA loads from the EngineCore process, whose lines carry a `(EngineCore pid=N)` prefix. The parser allows that prefix and strips ANSI colour codes. A vLLM run with zero parsed activations is invalid.

**Calibration feeds the runs.** `calibration.json` (from `calibrate`) supplies, for the scenario's output length:
- one-pod capacity, for `rate_c` scenarios;
- unloaded E2E, for the feasibility bound;
- the base-model TTFT limit, 3 × the unloaded p99, for validity check 6. S0 measures one adapter, whose unloaded TTFT is no lower than the base model's, so the limit is conservative.

**S0 pod configuration.**
- The load-cost probes need `--max-loras 1 --max-cpu-loras 2`, so the capacity sweep runs on the same pod.
- One adapter is active in the sweep, so `max_loras` does not bind there.
- If LoRA kernel cost turns out to depend on `max_loras`, the sweep should be repeated with `max_loras 2`.

**Not implemented yet:**
- the cold (unprimed) S1 variant;
- the generator self-test (§6.1);
- an explicit compile-cache prewarm run (the caches persist through `TRITON_CACHE_DIR` and `VLLM_CACHE_ROOT`);
- S0 with page caches dropped (needs root);
- cross-checking slot holding against the scraped running set;
- the residency response-hook check, the EPP plugin-latency bound, and GPU throttle parsing. Each run lists these three under `validity.not_checked`.

**v2.3 notes (10 Nov 2026, from the first GPU baselines, before any candidate filter).**
- **Capacity is the TTFT knee:** the highest swept rate with TTFT p99 ≤ 4× unloaded and ≥ 98% complete. The v2.2 rule also required ITL p50 ≤ 1.3× unloaded. On B200 that fired at 6 req/s (L = 1024), because batching raises ITL smoothly, while TTFT broke only between 16 and 20 req/s. S3 and S4 scale with this capacity.
- **Runs are serial.** Two 4-GPU streams running at once on one host made overloaded scenarios measurably worse and noisier: S4 residency giant p99 6.9–17.5 s alongside another stream vs 7.3–9.4 s alone. Every reported baseline is serial.
- **Overloaded scenarios are bimodal.** In S3 and S4, residency-based arms either settle into a balanced layout (queues stay empty) or tip one or two packed pods past capacity for the rest of the run. Which one happens is decided early. The queue-trend diagnostic shows which, per run.
