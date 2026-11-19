# Baseline v1: S1–S5 and the oracle headroom check (10 Nov 2026)

## Setup

- 4× NVIDIA B200, one vLLM 0.31.0 pod each (`unsloth/Llama-3.1-8B-Instruct@4699cc75`, synthetic adapters).
- llm-d-router `1f7f0d90` (#3392 merge), file discovery behind Envoy 1.33.2.
- Seeds 1–5 for the gate references (residency, rr2) and the oracle in S1, S3, S4 and S5; seeds 1–3 for the other arms. All runs **serial**, one 4-GPU run on the node at a time (DESIGN_LLMD_709.md §12, v2.3).
- Capacity from calibration (highest rate with TTFT p99 ≤ 4× unloaded, DESIGN_LLMD_709.md §12): 48 req/s at L = 256 and 14 req/s at L = 1024 per pod.
- 110 valid runs. One earlier S1 A/A run is invalid (one 503 during an Envoy connection race, since fixed) and was replaced by a valid re-run.

The evaluation agreed for the #709 gate is in [Evaluation](#evaluation-gate-seeds-35) at the end. The per-scenario sections before it describe seeds 1–3. Their values are steady-window TTFT p99 in seconds, per seed (1 / 2 / 3). "> 5 s" is the share of the class's requests with TTFT above 5 s. "inf" means more than 1% of the class timed out at 120 s.

## S1: adapters fit (2 hot + 4 cold, 8 slots, 12 req/s)

| arm | cold p99 | hot p99 | GPU activations |
| --- | --- | --- | --- |
| stock | 3.29 / 2.53 / 2.06 | 1.36 / 1.41 / 1.41 | 415 / 436 / 398 |
| stock-noprefix | 2.08 / 2.46 / 1.49 | 1.36 / 1.40 / 1.37 | 401 / 415 / 381 |
| affinity-w2 | 0.04 / 0.04 / 0.04 | 0.03 | 77 / 70 / 86 |
| residency-w4 | 0.03 / 0.03 / 0.03 | 0.03–0.04 | 4 / 4 / 4 |
| oracle | 0.03 / 0.03 / 0.03 | 0.03 | 4 / 4 / 4 |

Residency already matches the hand placement. Details in [s1-v1.md](s1-v1.md).

## S2: more busy adapters than slots (8 warm + 4 cold, 8 slots, L = 1024)

| arm | cold p99 | cold > 5 s | warm p99 | warm > 5 s | timeouts (cold / warm) |
| --- | --- | --- | --- | --- | --- |
| stock | 72.6 / 62.7 / 75.0 | 86–88% | 62.4 / 60.5 / 60.7 | 75–80% | 0 / 0 |
| affinity-w2 | inf | 79–100% | inf / 118 / 92.8 | 4–18% | 29–36 / 6–36 |
| residency-w4 | inf | 100% | 0.05 / 9.38 / 0.04 | 0–1% | 21–29 / 0–13 |
| oracle | 43.0 / 49.2 / 47.6 | 43–48% | inf | 24–26% | 0 / 53–67 |

**No layout fits here.** All 12 adapters hold slots against 8 (feasibility bound), so every arm starves somebody:
- residency starves the cold adapters completely and keeps warm ones fast;
- the oracle isolates the cold adapters on one pod and starves warm ones instead.

A placement filter can only move the starvation between classes. Removing it needs time-slicing in the engine, as #3392 already noted.

## S3: two hot adapters, each ~1.15 pods of compute (L = 1024, 44.8 req/s)

| arm | hot p99 | cold p99 | base p99 | pods with a growing queue |
| --- | --- | --- | --- | --- |
| stock | 0.17 / 0.41 / 0.67 | inf (100% > 5 s) | 0.19–0.74 | 0 |
| affinity-w2 | 72.4 / 44.3 / 13.8 | 0.05 / 40.1 / 42.7 | 12.8–22.4 | 2 / 0 / 0 |
| residency-w4 | 7.91 / 10.6 / 8.13 | 0.06 / 53.2 / 0.08 | 0.07 / 2.25 / 0.07 | 0 |
| residency-w4 (A/A) | 7.07 / 6.39 / 13.4 | 21.2 / 0.05 / 91.7 | 3.63 / 0.05 / 11.9 | 0 / 0 / 3 |
| residency-w4-rr2 | 0.59 / 0.13 / 8.77 | 0.60 / 0.12 / 116 | 0.06 / 0.13 / 1.52 | 0 / 0 / 3 |
| oracle | 0.08 / 0.09 / 0.09 | 0.10 / 0.09 / 0.10 | 0.08–0.09 | 0 |

- **Stock spreads the hot adapters,** so hot traffic is fine, but the four cold adapters never get a slot: every cold request is over 5 s.
- **Residency packs each hot adapter onto too few pods.** Hot p99 sits at 6–13 s in every run, and cold p99 swings from 0.05 s to 92 s depending on the run.
- **residency + running-requests (rr2) is bimodal:** fast on seeds 1 and 2, tipped past capacity on seed 3.
- **The hand placement** (each hot adapter on 2 pods, cold adapters spread) is at about 0.1 s for every class on every seed.

## S4: one giant adapter, ~2.25 pods of compute (L = 1024, 44.8 req/s)

| arm | giant p99 | cold p99 | cold > 5 s | base p99 |
| --- | --- | --- | --- | --- |
| stock | 9.10 / 13.8 / 9.58 | 82.8 / 72.4 / 59.1 | 78–79% | 1.86–8.47 |
| affinity-w2 | 6.31 / 10.7 / 30.2 | inf / 118 / 81.7 | 26–29% | 4.13–10.7 |
| residency-w4 | 8.84 / 8.95 / 7.46 | 66.4 / 29.7 / 59.8 | 13–14% | 1.60–3.57 |
| residency-w4 (A/A) | 9.01 / 9.39 / 7.31 | 55.1 / 39.8 / 1.37 | 0–15% | 0.08–8.89 |
| residency-w4-rr2 | 0.15 / 4.05 / 0.11 | 0.12 / 91.4 / 0.09 | 0–9% | 0.07–2.24 |
| oracle | 0.11 / 0.10 / 0.09 | 0.12 / 0.09 / 0.10 | 0% | 0.07–0.11 |

- **Residency keeps the giant adapter on 2 pods** (about 225 requests in flight each, against about 60 on the others), so giant p99 stays at 7–9 s and cold adapters behind it wait up to a minute.
- **rr2 is bimodal again:** fast on seeds 1 and 3, tipped on seed 2.
- **The hand placement** (giant on 3 pods, cold adapters on the 4th) is at about 0.1 s on every seed.

## S5: Zipf long tail (32 adapters, 16 slots, CPU tier 4, 8 req/s)

| arm | LoRA p99 | base p99 | storage loads | GPU activations |
| --- | --- | --- | --- | --- |
| stock | 4.53 / 3.78 / 3.42 | 0.84–1.03 | 1002 / 945 / 774 | same |
| affinity-w2 | 3.12 / 3.18 / 2.84 | 0.84–1.06 | 572 / 552 / 516 | same |
| residency-w4 | 0.97 / 2.63 / 0.85 | 0.29–0.47 | 279 / 316 / 254 | same |
| residency-w4-rr2 | 1.86 / 1.69 / 0.91 | 0.55–0.86 | 372 / 323 / 272 | same |

Residency halves the adapter loads against affinity (a third of stock's) and cuts p99 roughly 1.2–3×. Adding rr2 costs a little: about 15% more loads, and p99 similar or slightly higher. Every load comes from storage, because the CPU tier (4) is far smaller than the catalogue (32). Each one stalls a vLLM batch, which is also why base-model TTFT rises.

## Evaluation (gate seeds 3–5)

Proposed gate for #709, replacing the per-seed p99 rules (DESIGN_LLMD_709.md §8). It was fixed before any candidate filter ran. It needs agreement on #709 before it judges one.

**Seeds.** A filter's settings are tuned on seeds 1–2 and frozen before it runs on gate seeds 3–5. The baselines ran all five seeds; only the filter's own runs remain.

**Three metrics, all with one threshold of 0.5 s.** That threshold is the unloaded TTFT p99 (0.03 s) plus one storage load of a rank-16 adapter (0.43 s), rounded up. So a request that only had to load its adapter is not a miss; one that queued is.

| Metric | Definition |
| --- | --- |
| **Miss rate** (primary) | Share of all steady-window requests, base and every adapter, with TTFT > 0.5 s. Timeouts count as misses. Pooled over the gate seeds. |
| **Bad seeds** (reliability) | Gate seeds whose miss rate is above 5%. Win scenarios only. |
| **Worst adapter** (fairness) | The highest per-adapter miss rate, counting only adapters with ≥ 20 steady-window requests on that seed, on the worst gate seed. |

**Roles:**
- **S3 and S4 are win scenarios, and a filter must win both.** A win means, against both residency-w4 and residency-w4-rr2:
  - a pooled miss rate at least 2 points lower;
  - no more bad seeds;
  - a worst adapter no more than 5 points worse.
- **S1 and S5 are guards:** miss rate within +1 point and worst adapter within +5 points of both references.
- **S2 is report-only.** No layout fits there (above).

**Why these and not p99:**
- In S3 and S4, the residency-based arms are bimodal: a run either settles into a balanced layout or tips one or two pods past capacity for the rest of the run. A/A repeats of residency differ by up to 15 points of miss rate on the same seed. So only pooled and per-seed-count metrics are meaningful.
- Per-class p99 over 30–60 cold requests is effectively one request.
- The worst-adapter metric catches what pooled numbers hide. S3 stock misses only 10% of all requests, yet its cold adapters miss 100%.

### Results (cells: miss rate % · bad seeds · worst adapter %)

| Scenario | Role | residency-w4 | residency-w4-rr2 | oracle |
| --- | --- | --- | --- | --- |
| S3 two hot | win | 27.2 · 2/3 · 66.5 | 8.9 · 2/3 · 91.5 | **0.0 · 0/3 · 0.0** |
| S4 one giant | win | 36.5 · 3/3 · 57.0 | 8.8 · 1/3 · 46.7 | **0.0 · 0/3 · 0.0** |
| S1 fits | guard | 0.0 · 0/3 · 0.0 | 0.1 · 0/3 · 8.3 | 0.0 (seed 3 only) |
| S5 Zipf | guard | 3.9 · 1/3 · 18.2 | 4.5 · 1/3 · 13.5 | not written |

Per-seed miss rate (%), seeds 1–5:

| | S3 | S4 |
| --- | --- | --- |
| residency-w4 | 35.6 / 34.9 / 46.8 / 4.9 / 30.4 | 32.9 / 51.8 / 27.2 / 46.3 / 35.6 |
| residency-w4-rr2 | 2.2 / 0.0 / 9.5 / 16.9 / 0.0 | 0.0 / 18.8 / 0.0 / 26.0 / 0.0 |
| oracle | 0 on every seed | 0 on every seed |

Per-adapter TTFT p90 / p99 (s), gate seeds pooled:

| S3 | hot-0 | hot-1 | cold-0 | cold-1 | cold-2 | cold-3 | base |
| --- | --- | --- | --- | --- | --- | --- | --- |
| residency-w4 | 2.05 / 8.0 | 1.95 / 7.6 | 0.06 / 110 | 0.06 / 98 | 0.07 / 0.43 | 0.08 / 0.27 | 0.06 / 0.14 |
| residency-w4-rr2 | 0.78 / 1.8 | 0.09 / 0.61 | 78 / 111 | 0.07 / 16 | 72 / 106 | 0.08 / 15 | 0.09 / 0.83 |
| oracle | 0.06 / 0.10 | 0.06 / 0.09 | 0.07 / 0.11 | 0.06 / 0.10 | 0.07 / 0.08 | 0.07 / 0.09 | 0.06 / 0.10 |

| S4 | giant | cold-0 | cold-1 | cold-2 | cold-3 | cold-4 | base |
| --- | --- | --- | --- | --- | --- | --- | --- |
| residency-w4 | 3.5 / 7.9 | 4.3 / 25 | 0.13 / 86 | 10 / 73 | 23 / 69 | 19 / 89 | 0.33 / 7.0 |
| residency-w4-rr2 | 0.57 / 2.3 | 0.06 / 71 | 0.66 / 38 | 12 / 71 | 0.06 / 69 | 33 / 67 | 0.06 / 0.64 |
| oracle | 0.06 / 0.09 | 0.07 / 0.10 | 0.06 / 0.10 | 0.07 / 0.10 | 0.05 / 0.07 | 0.05 / 0.07 | 0.05 / 0.07 |

### Reading

1. **There is placement headroom in both win scenarios.** The oracle has a 0% miss rate on all five seeds of S3 and S4. Under this gate it passes against both references.
2. **rr2's weakness is reliability.** It tips over on 2 of 5 seeds in both S3 and S4. When it does, specific cold adapters starve (S3 cold-0 and cold-2 at p90 72–78 s), while its pooled miss rate stays under 10%.
3. **Residency's weakness is packing.** In S3 it slows the hot adapters on every seed (p90 2 s). In S4 it starves cold adapters behind the giant one.
4. **S5 and S1 show no room to lose.** Residency and rr2 are within noise of each other there, so the guards only check that a filter does no harm.

## Questions for #709 (before any filter runs)

1. **Adopt the evaluation above**, including S2 as report-only, S3 and S4 as win scenarios, and the 0.5 s threshold.
2. **How did #3392 define one-pod capacity?** S3 and S4 scale from it; this report uses the highest rate with TTFT p99 ≤ 4× unloaded (DESIGN_LLMD_709.md §12).

## Limits

- One model and adapter rank, and synthetic adapters.
- The oracle placements are hand-written per scenario. They show what is achievable, not the best possible layout.
- Each run is 180 s. Bimodal outcomes might settle differently over longer runs.
- Three gate seeds detect an arm that tips one run in three only about 70% of the time. The worst-adapter metric partly compensates, because a tipped seed starves specific adapters.
- Not checked automatically: the residency response hook, the EPP plugin-latency bound (EPP uses about 3.5 cores at 45 req/s), and GPU throttling.
