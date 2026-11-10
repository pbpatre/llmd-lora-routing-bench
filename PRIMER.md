# A primer on LoRA routing in llm-d and vLLM

This chapter explains, from first principles, what llmd-lora-routing-bench measures and why it measures it this way. It assumes you know roughly what an LLM is and what an HTTP request is, and nothing more.

When you finish, you should be able to read a llmd-lora-routing-bench report, say what each number means, and judge whether a result is trustworthy. [DESIGN_LLMD_709.md](DESIGN_LLMD_709.md) is the precise specification.

**Contents**

1. [The problem: many adapters, few slots](#1-the-problem-many-adapters-few-slots)
2. [Where the router fits](#2-where-the-router-fits)
3. [The question this benchmark answers](#3-the-question-this-benchmark-answers)
4. [Vocabulary](#4-vocabulary)
5. [Arms: the routers being compared](#5-arms-the-routers-being-compared)
6. [Scenarios: the traffic mixes, and what each proves](#6-scenarios-the-traffic-mixes-and-what-each-proves)
7. [Why this works without Kubernetes](#7-why-this-works-without-kubernetes)
8. [Simulator runs and real runs](#8-simulator-runs-and-real-runs)
9. [Calibration](#9-calibration)
10. [How one measurement is taken](#10-how-one-measurement-is-taken)
11. [Seeing the mechanism, not just the latency](#11-seeing-the-mechanism-not-just-the-latency)
12. [Validity: when to throw a run away](#12-validity-when-to-throw-a-run-away)
13. [Statistics and the gate](#13-statistics-and-the-gate)
14. [Worked example: reading a real result](#14-worked-example-reading-a-real-result)
15. [Summary and glossary](#15-summary-and-glossary)

---

## 1. The problem: many adapters, few slots

### 1.1 Base models and LoRA adapters

A **base model** such as Llama-3.1-8B has about 16 GB of weights. Companies often need many variants of it: one tuned for legal text, one for support tickets, one per customer. Training and serving a full copy for each would cost 16 GB per variant.

**LoRA** (Low-Rank Adaptation) avoids that. A variant is stored as a small **adapter**: a pair of thin matrices added to each layer of the base model. An adapter is typically tens to hundreds of megabytes. Its size depends on its **rank**: rank 8 is small and rank 64 is larger.

To answer a request for adapter `legal`, the server runs the base model with `legal`'s matrices added in. Modern engines, vLLM among them, can do this for several different adapters **in the same batch**. That is what makes serving thousands of adapters from one base model economical.

### 1.2 Slots: the GPU only holds a few adapters at once

The engine reserves room on the GPU for a fixed number of adapters at a time. In vLLM this is `--max-loras`, and this chapter calls those places **slots**. With `--max-loras 2`, at most 2 distinct adapters can be in the running batch on that GPU.

Adapters that aren't in a slot live further away:

```
   fast, tiny                                          slow, huge
 ┌───────────────┐   activate     ┌──────────────┐   load     ┌──────────────┐
 │  GPU slots    │ ◄───────────── │   CPU memory │ ◄───────── │   storage    │
 │ (max_loras)   │   (faster)     │(max_cpu_loras)│  (slower) │  (disk/net)  │
 └───────────────┘                └──────────────┘            └──────────────┘
```

Moving an adapter from CPU memory into a slot is an **activation**. Moving it from storage into CPU memory is a **load**. Both take time; calibration (§9) measures how long on the hardware at hand. While an adapter waits for a slot, its requests wait too.

### 1.3 How vLLM decides who gets a slot, and how that starves rare adapters

vLLM's scheduler works in steps. At each step it keeps the requests that are already running, then tries to add waiting requests. Three rules matter here:

1. **A running request keeps its adapter's slot.** An adapter can't be evicted while any of its requests is still generating.
2. **A waiting request whose adapter would need a new slot, when all slots are taken, is skipped.** The scheduler moves on to the next waiting request.
3. **There is no aging.** A skipped request gets no priority for having waited.

Now picture a pod with 2 slots and two busy ("hot") adapters, A and B. Each always has at least one request running. A request arrives for a rarely used ("cold") adapter C:

```
time ─────────────────────────────────────────────────────►
slot 1:  A A A A A A A A A A A A A A A A A A A A A A A A ...   (A never drains)
slot 2:  B B B B B B B B B B B B B B B B B B B B B B B B ...   (B never drains)
C:       waiting ... skipped ... skipped ... skipped ...       (starves)
```

C runs only if A or B happens to have zero running requests for a moment, and with steady traffic that may take seconds or never happen. This is **starvation**. C's users see a time-to-first-token of many seconds while everyone else sees milliseconds.

Notice that C isn't slow because the GPU is busy. It's slow because **of the adapters it was placed next to**. That makes it a routing problem.

### 1.4 Thrash

The opposite failure is **thrash**. If requests for many adapters are spread evenly over every pod, each pod keeps swapping adapters in and out of its few slots. Every swap is an activation, and activations cost time.

The goal is a layout where busy adapters stay put and rare adapters always find a free slot somewhere.

---

## 2. Where the router fits

### 2.1 Several pods, one entry point

A real deployment runs several copies of the engine, called **pods**. Each has its own GPU and slots, and all of them serve the same base model and adapters. A **router** sits in front of them and decides which pod handles each request.

That decision also decides **which adapters end up sharing a pod**, so the router controls both failures from §1: starvation and thrash.

### 2.2 llm-d, the EPP and Envoy

[llm-d](https://github.com/llm-d) is an open-source stack for serving LLMs on Kubernetes. Its routing component is **llm-d-router**, and the part that makes the decision is the **Endpoint Picker (EPP)**.

The EPP doesn't carry request traffic itself. It works with **Envoy**, a widely used proxy:

1. A client sends a request to Envoy.
2. Envoy pauses the request and asks the EPP where to send it. It does this over a gRPC side channel called **ext_proc**, short for external processing.
3. The EPP picks a pod and answers with a header: `x-gateway-destination-endpoint: 10.0.0.7:8000`.
4. Envoy forwards the request to that address. This is the `ORIGINAL_DST` cluster type: "send it where the header says".

So **Envoy moves the bytes and the EPP makes the choice**.

### 2.3 How the EPP chooses: filters, scorers, picker

For every request, the EPP runs a small pipeline over the candidate pods:

```
all pods ─► filters ─► scorers (weighted) ─► picker ─► one pod
            drop some   give each pod a       highest total
                        number in [0, 1]      score wins
```

- **Filters** remove pods. For example, the default `utilization-detector` drops pods whose waiting queue is longer than 5, whose KV cache is over 80% full, or whose metrics are stale.
- **Scorers** rate each remaining pod between 0 and 1. Each score is multiplied by its **weight**, and the results are added up. The default scorers are:
  - `queue-scorer` (weight 2): prefers short queues;
  - `kv-cache-utilization-scorer` (weight 2): prefers free memory;
  - `prefix-cache-scorer` (weight 3): prefers pods that have probably already processed the start of this prompt.
- **The picker** takes the pod with the highest total. Ties are broken round-robin.

The EPP learns each pod's state, such as queue length and which adapters are running or waiting, by scraping the pod's `/metrics` endpoint regularly (llmd-lora-routing-bench sets every 50 ms).

**The default router has no LoRA-aware scorer at all.** It doesn't consider which adapters a pod holds. LoRA awareness is added by plugins, and comparing those plugins is what this benchmark does (§5).

---

## 3. The question this benchmark answers

Issue [llm-d-router#709](https://github.com/llm-d/llm-d-router/issues/709) proposes **adaptive LoRA placement**: a router filter that gives each adapter a stable set of pods. The set grows when the adapter gets busy and shrinks when it cools, and a slot is kept free for rare adapters.

Before building it, two things must be known:

1. **How good are today's routers?** A new filter is only worth adding if it beats the best existing option. Here that's `lora-residency-scorer`, added in [#3392](https://github.com/llm-d/llm-d-router/pull/3392).
2. **What counts as "better"?** The rule is agreed on #709: the filter must win clearly in at least one traffic mix and lose in none. It must be fixed *before* the filter exists, so nobody can bend it afterwards.

llmd-lora-routing-bench produces the first answer (the **baselines**) and implements the second (the **gate**). Later, it will run the new filter through exactly the same tests.

---

## 4. Vocabulary

| Term | Meaning |
| --- | --- |
| **Scenario** | One workload: how many pods, how many slots, which adapters, how much traffic, and how it's split. It's a YAML file in `scenarios/`. |
| **Traffic class** | A named group of adapters with a share of the traffic, such as "hot: 90% over 2 adapters". Results are reported per class. |
| **Base requests** | Requests for the base model with no adapter (20% in most scenarios). They show whether the pool as a whole is overloaded. |
| **Arm** | One router configuration being compared, such as `stock` or `residency-w4`. The term comes from clinical trials, where each "arm" receives a different treatment. |
| **Schedule** | The exact list of requests in a run: when each is sent, to which adapter, with which prompt. It's generated from the scenario and a seed. |
| **Seed** | The number that makes a schedule reproducible. The same scenario and seed always give the identical schedule. |
| **Run** | One (scenario, arm, seed, replicate): fresh pods, a fresh router, the schedule replayed, the results analysed. |
| **Paired comparison** | Every arm replays the *same* schedule for a given seed. Differences between arms then come from the router, not from different random traffic. |
| **Replicate, A/A** | Running the same arm twice on the same seed. The two copies should agree, and their difference measures the **noise**. |
| **Open-loop load** | Requests are sent at their scheduled times whether or not earlier requests have finished, the way real users behave. See §10.2. |
| **Steady window** | The part of a run after the warm-up (by default, from 60 s on). Gate decisions use it. |
| **TTFT** | Time to first token: how long until the user sees the first word of the answer. This is the latency users feel. |
| **p99** | The 99th percentile. 99% of requests were at least this fast. It describes the worst 1%, which is where starvation shows. |

---

## 5. Arms: the routers being compared

Every arm except `random` starts from the default router and adds at most one LoRA-related plugin. All of them keep the default `utilization-detector` filter, because a real deployment has it. Its parameters are written out explicitly in each config, so they're recorded.

| Arm | What it is | Why it's here |
| --- | --- | --- |
| `stock` | The default router: queue 2, KV 2, prefix 3. No LoRA awareness. | The starting point: what you get out of the box. |
| `stock-noprefix` | `stock` without the prefix scorer. | A **control**. The prefix scorer's hash includes the model name, so it may accidentally keep each adapter on the same pod. Removing it shows how much of `stock`'s behaviour comes from that accident. |
| `affinity-w2` | `stock` + `lora-affinity-scorer` at weight 2. | The older LoRA plugin. It prefers pods where the adapter is already active, then pods with a free slot. |
| `residency-w4` | `stock` + `lora-residency-scorer` at weight 4. | **The reference to beat.** It estimates the cost of serving the adapter on each pod. The cost is low if the adapter is already in a slot, higher for a CPU copy, and highest for a full load. It adds a penalty when a request would be blocked or would evict a busy adapter. |
| `residency-w4-rr2` | `residency-w4` + `running-requests-size-scorer` at weight 2. | The pairing #3392's README recommends when a hot adapter overloads its pod. Residency alone keeps a hot adapter on its pod *even when that pod is overloaded*, so this arm adds load-spreading. It's a second reference in the overload scenarios. |
| `oracle` | `residency-w4` + a hand-written placement, applied at weight 100. | **A ceiling estimate.** A person writes down, per scenario, which pods each adapter should use. The load client tags each request with its oracle pod, and the router strongly prefers it. If even this can't beat residency, no placement filter can, and the #709 filter isn't worth building. |
| `random` | No scorers; the random picker. | **A floor.** Any sensible router should beat it. It's used only in simulator checks. |
| *(later)* `placement-*` | The #709 filter. | The candidate the gate will judge. |

Weights matter because scores are added up. Residency at weight 4 against the default scorers' 7 in total means LoRA cost matters a lot, but not absolutely. The oracle's weight of 100 means it almost always wins.

---

## 6. Scenarios: the traffic mixes, and what each proves

Each scenario is built to create **one situation**, so its result has one clear interpretation. All except S0 and S5 use 4 pods with 2 slots each, which makes 8 slots in total. The 20% base-model traffic isn't repeated in each description.

### S0: calibration (1 pod)

**What it does:** sweeps the request rate on a single pod to find its capacity, and measures how long loads and activations take for each adapter rank. See §9.

**What it proves:** nothing about routing. It produces the reference numbers that the other scenarios and checks need.

### S1 "fits": a good layout exists; does the router find it?

**Setup:** 2 hot adapters (90% of LoRA traffic), 4 cold ones (10%), 12 req/s, 256 output tokens. This replicates #3392's starvation study.

**Why it's interesting:** each hot adapter has a request running practically all the time, so it holds a slot permanently. The cold adapters together need only a little capacity. A good layout clearly exists, for example:

```
pod-0: [hot-0, ·]     pod-1: [hot-1, ·]     pod-2: [cold-0, cold-1]     pod-3: [cold-2, cold-3]
```

No cold adapter ever has to wait behind a hot one.

**What it proves:** whether a router **finds** a layout that exists.
- `stock` spreads everything everywhere. Hot adapters end up holding both slots on every pod, and cold requests starve.
- `residency-w4` should keep things apart.
- Because residency already does well here, S1 isn't where a new filter can win. It's a **guard**: the new filter must not do worse.

### S2 "oversubscribed": more busy adapters than slots

**Setup:** 8 "warm" adapters (90%) and 4 "cold" ones (10%), 20 req/s, 1,024 output tokens. This replicates #3392's follow-up 1b.

**Why it's interesting:** with long responses, even the "cold" adapters have about 4 requests in flight on average (0.4 req/s × ~10 s per request). So all 12 adapters want a slot nearly all the time, and there are only 8 slots. **No layout fits.** Someone must wait, and no layout can prevent it. Only *time-sharing* can: deliberately letting slots drain so others get a turn.

**What it proves:**
- **For today's routers:** how badly each fails when no layout fits.
- **For a new filter:** this is one of two places it can win. The gate's win condition is that the cold > 5 s share drops by at least 10 percentage points against residency, on every gate seed, without hurting warm adapters.

### S3 "two hot": an adapter too busy for one pod

**Setup:** 2 hot adapters, each needing about 1.15 pods of compute, plus 4 cold ones. The total is 80% of cluster capacity. The rate is set relative to the measured one-pod capacity (`rate_c`). This replicates #3392's follow-up 1a, rescaled to our GPUs.

**Why it's interesting:** slots are fine, but a hot adapter kept on a single pod overloads that pod. Residency tends to do exactly that.

**What it proves:** whether a router can spread a busy adapter over *just enough* pods. This is a guard scenario, where the new filter must not lose. It's also where `residency-w4-rr2` matters.

### S4 "one giant": the case named on #709

**Setup:** 1 "giant" adapter needing about 2.25 pods of compute, plus 5 cold adapters. Not measured in #3392.

**Why it's interesting:** the giant must occupy about 3 pods to keep up. But if it's on 3 pods, its requests hold slots on all 3, and the 5 cold adapters must squeeze into what's left. This is the situation that motivates *adaptive* placement: the right number of pods for an adapter depends on its traffic.

**What it proves:** the second place a new filter can win. The win condition is a giant p99 below residency's by more than the noise band, on every gate seed, without hurting the cold adapters.

### S5 "Zipf": a long tail

**Setup:** 32 adapters with mixed ranks (8, 16, 32 and 64), 4 slots per pod, a CPU tier of only 4. Popularity follows a Zipf(1.0) curve: the first adapter is twice as popular as the second, three times as popular as the third, and so on. This is one cell of #3392's main grid.

**Why it's interesting:** this is the most realistic shape, with a few popular adapters and a long tail of rare ones. It's also the only scenario where the CPU tier is smaller than the catalogue, so real storage-to-CPU loads happen.

**What it proves:** a cross-check that the conclusions from the clean scenarios still hold in a messy, realistic mix. It's not part of the gate.

### Summary

| Scenario | Situation | Gate role |
| --- | --- | --- |
| S0 | one pod, sweep | produces reference numbers |
| S1 | a layout exists | guard (must not lose) |
| S2 | no layout exists, too many busy adapters | **win possible** |
| S3 | one adapter needs more than a pod | guard |
| S4 | one adapter needs several pods | **win possible** |
| S5 | realistic long tail | cross-check |

---

## 7. Why this works without Kubernetes

In production, llm-d runs on Kubernetes. Kubernetes provides four things that matter here:
1. it starts the pods;
2. it tells the EPP where the pods are (service discovery);
3. it gives each pod its own IP address;
4. it runs the Envoy gateway in front.

None of the routing logic lives in Kubernetes. The EPP's filters, scorers and picker are the same Go code either way. So llmd-lora-routing-bench replaces each Kubernetes piece with something simpler, and keeps everything that makes the decision:

| Kubernetes provides | llmd-lora-routing-bench uses instead |
| --- | --- |
| pods | plain processes: `vllm serve …` or `llm-d-inference-sim …`, each pinned to one GPU with `CUDA_VISIBLE_DEVICES` |
| service discovery (the EPP watches the Kubernetes API) | the EPP's built-in **file discovery** plugin, which reads the pod list from `endpoints.yaml` |
| one IP per pod | one **loopback address** per pod on Linux (127.0.0.2, .3, .4, .5). The EPP matches pods by IP, so pods must not share one. |
| the gateway | an Envoy process whose config copies llm-d's standalone Helm chart: same version, ext_proc settings, timeouts and `ORIGINAL_DST` routing |

**What this keeps:** the real EPP binary, built from the pinned commit, with its real plugins, metric scraping and decision path; the real Envoy data path; and the real vLLM engine.

**What it loses:** Kubernetes-only behaviour such as pod restarts, autoscaling, and network latency between machines. None of it is what #709 is about.

**What it gains:**
- one command starts the whole stack;
- every config is written into the run directory;
- each run restarts everything from scratch;
- it runs on any Linux box with GPUs.

That last point is what makes it reproducible by other people.

---

## 8. Simulator runs and real runs

llmd-lora-routing-bench has two **backends** for the pods.

### 8.1 The real backend: vLLM on GPUs

Each pod is a real `vllm serve` process on its own GPU, serving the real base model with real (synthetic) adapters. This is the **only** source of numbers that count.

The adapters have random weights. That's deliberate and disclosed: routing and slot behaviour depend on adapter *size*, not on what an adapter has learned.

### 8.2 The simulator backend: llm-d-inference-sim

[llm-d-inference-sim](https://github.com/llm-d/llm-d-inference-sim) is a small program that **pretends** to be vLLM. It:
- speaks the same HTTP API;
- exposes the same metric names, including the LoRA running and waiting lists;
- enforces the `max_loras` slot limit;
- produces tokens on a timer instead of computing them.

It needs no GPU and starts in a second.

**What it gets right:** the slot limit, so the router sees the same kind of signals and starvation can appear.

**What it gets wrong:**
- there's no load or activation latency;
- eviction is random, not least-recently-used;
- there's no CPU tier;
- it orders waiting requests differently from vLLM.

**What it's used for:** checking the harness, quickly and for free.
- Do all the processes start and talk to each other?
- Do the configs load?
- Are requests counted and attributed correctly?
- Do the qualitative regimes appear (§10 of DESIGN_LLMD_709.md)?

**What it's never used for:** comparing latency between arms. A simulator TTFT measures the simulator's timer, not a GPU.

### 8.3 Smoke runs and quick runs

- **`make gpu-smoke`:** S1, shortened, on the simulator. It takes a few minutes and uses no GPU. Run it first on any new machine.
- **`make quick-s1`:** S1 on real vLLM, stock against residency, one seed. It's the first real number, but it's preliminary: one seed and no noise estimate.
- **`make s1-set` and `make first-set`:** the full sets, with 3 seeds and A/A replicates. These are the numbers worth reporting.

---

## 9. Calibration

### 9.1 Why calibrate

Some questions can't be answered without knowing how fast *this* hardware is:
- **How much traffic is "80% of capacity"?** S3 and S4 are defined that way (`rate_c`), so they mean the same thing on an H100 as on a B200.
- **How long is a request with no queueing?** The feasibility check (§11) needs this.
- **What base-model TTFT means "not overloaded"?** A validity check needs it (§12).

### 9.2 Capacity

Calibration (scenario S0) runs a single pod with one adapter. It steps the request rate up: 1, 2, 4, 6, … req/s, for 60 s at each step. Then it applies a fixed rule:

> **Capacity** is the highest rate at which, compared with the lowest rate:
> - TTFT p99 is at most 2× higher,
> - the time between tokens (ITL p50) is at most 1.3× higher, and
> - at least 98% of requests complete.

That rate is the **knee** of the curve: beyond it, latency climbs fast. It's measured separately for 256 and 1,024 output tokens, because longer answers occupy the GPU longer.

### 9.3 Load and activation cost

With `--max-loras 1 --max-cpu-loras 2`, the pod can hold one adapter in a slot and two in CPU memory. Then:
- **First request to a new adapter:** pays load plus activation.
- **Second request:** pays nothing extra, because the adapter is already in the slot. The difference between the two is the **load cost**.
- **Alternating between two adapters already in CPU memory:** each switch pays an activation only. The difference is the **activation cost**.

Each measurement is repeated per rank (8, 16, 32, 64), and the results go into `calibration.json`.

---

## 10. How one measurement is taken

### 10.1 The life of a run

```
1. resolve scenario  → request rate fixed (from calibration if rate_c)
2. generate schedule → e.g. 2,209 requests for S1 seed 1, saved with a checksum
3. start 4 pods      → wait until healthy, all adapters listed, one priming request each
4. start EPP         → wait until it reports all 4 pods ready
5. start Envoy       → snapshot its counters
6. start the scraper → pod metrics every 100 ms, EPP metrics every 5 s
7. replay schedule   → 4 sender processes, shared start time t0
8. stop everything   → snapshot Envoy counters, kill all processes
9. collect           → join Envoy's access log; parse vLLM's adapter-activation log lines
10. analyse          → summary.json
```

Every run starts from **fresh processes**, so no arm inherits a warm cache from the previous one. Arm order rotates from seed to seed, so no arm always runs first.

### 10.2 Open-loop sending, and why it matters

A naive load tester waits for one answer before sending the next request. That's called **closed-loop**. When the server slows down, a closed-loop tester also slows down, and the slow period is under-measured. This error has a name: **coordinated omission**.

Real users don't wait for each other. llmd-lora-routing-bench is **open-loop**: each request is sent at its scheduled time no matter what. Arrival times follow a Poisson process, the standard model for independent users. It's generated from the seed, so it's identical across arms.

The client itself must keep up, or its own delay would look like server latency. So it is split over 4 processes and monitors itself. Every run records how late each request was sent (**send lag**) and how late the client's event loop woke up (**loop lag**). Too much lag invalidates the run.

### 10.3 Timestamps per request

```
t_sched        t_send         t_upstream        t_first_token                t_end
   │  client     │   Envoy +     │   engine queue   │         streaming          │
   │  delay      │   EPP choice  │   + prefill      │         tokens             │
   ▼             ▼               ▼                  ▼                            ▼
   ├─────────────┼───────────────┼──────────────────┼────────────────────────────┤
   └──────────────────── TTFT (primary) ────────────┘
```

- **Primary TTFT** = first token − *scheduled* time. Any delay anywhere counts against the system, including the client's.
- **The serving pod** comes from Envoy's access log, the authoritative record of where Envoy sent the request. A response header is used only as a cross-check.
- **Every request decodes exactly L tokens** (`ignore_eos`), so all requests are the same amount of work and the arms are comparable.

### 10.4 Percentiles with timeouts

A request that times out (120 s) before its first token never gets a TTFT. Dropping it would make a starving router look *better*. Instead, its TTFT counts as **infinity**. A p99 of `inf` therefore means "more than 1% of this class timed out", not a bug.

---

## 11. Seeing the mechanism, not just the latency

Latency says *that* something went wrong. These measures say *why*, and they're what make a result convincing to a reviewer.

**Slot holding.** For each pod and adapter, the share of time the adapter had at least one request running, and so held a slot. It's built from each request's [first token − prefill estimate, end] interval.

**Blocked at arrival.** For each LoRA request: when it reached its pod, were at least `max_loras` *other* adapters holding slots there? If so, it had to wait for one to drain. This is the direct measure of the starvation in §1.3. A blocked share of 70% for cold requests means 7 in 10 arrived at a pod already full of other adapters.

**Pods per adapter.** How many distinct pods served each adapter. A value of 4 means "spread everywhere"; 1 means "kept in one place".

**Activations and loads.** Counted from vLLM's own log lines:
- `Activating LoRA …` is a CPU→GPU move;
- `Adding lora …` is a storage→CPU load.

Many activations mean thrash. The simulator has no such logs, so it uses a proxy: how often an adapter newly appears in a pod's running set.

**Feasibility bound.** Computed offline from the offered rates and calibration. Could *any* placement fit? The steps:
1. Each adapter needs enough pods for its compute.
2. An adapter with an average of 3 or more requests in flight per pod is idle less than 5% of the time, so it effectively holds a slot permanently.
3. If the slot-holding adapters leave fewer free slots than the rare adapters need at once, **no layout works**.

The bound reports `slots_fit` and `compute_fits`. It's how S2 is shown to be oversubscribed by arithmetic, not by assertion.

**Reached layout.** Per 30-second window: did every pod hosting a rare adapter have a free slot? It compares what the router *achieved* with what was *possible*.

**Imbalance.** The busiest pod's average number of in-flight requests divided by the average over all pods. A value of 1.0 is perfectly even.

---

## 12. Validity: when to throw a run away

A benchmark that never rejects its own runs can't be trusted. Each run is checked, and **an invalid run is excluded and re-run**, never "adjusted". A run is valid only if all of the following hold:

1. **The client kept up:** loop lag p99 below 5 ms, send lag p99 below 50 ms.
2. **Every completed request produced exactly L tokens**, and reported its usage.
3. **Every completed request is in Envoy's access log**, served by one of the 4 pods, and the response header agrees.
4. **No errors** other than timeouts, and Envoy reports no overflow or ext_proc failures.
5. **On vLLM, adapter activations were found in the logs.** Zero would mean the log parsing broke.
6. **The base model wasn't overloaded:** base TTFT p99 is at most 3× its unloaded value from calibration. This is skipped for S3 and S4, which are overloaded by design, and for S5, where adapter loads stall every request (DESIGN_LLMD_709.md §6.4).

Checks not automated yet are listed in each run's `validity.not_checked`, so their absence is explicit.

---

## 13. Statistics and the gate

### 13.1 Why one run isn't enough

Run the same arm twice on the same schedule and the p99 differs a little, because of timing jitter, GPU clocks and thread scheduling. Before claiming arm X beats arm Y, you need to know how big that noise is.

- **Seeds:** each comparison uses 3 seeds (3 different schedules). A win must hold on **every** seed, not on average.
- **A/A runs:** on every seed, `residency-w4` runs twice, once at the start of the seed's block and once at the end. The difference between the two copies is measured noise.
- **The noise band** for p99 is the wider of the A/A spread and #3392's own replicate spread (0.70×–1.41×). A p99 "win" must fall outside this band.
- **Shares and counts** (such as "% over 5 s") use fixed absolute margins instead, such as "at least 10 percentage points".

### 13.2 Pre-registration

The rules are written into code (`compare.py`, `GATE`) **before** any candidate filter exists. Three habits keep the process honest:
1. **Tuning seeds and gate seeds are separate.** The filter's parameters may be tuned on seeds 1–3. The verdict uses only seeds 4–6, which the tuning never saw.
2. **Missing or invalid runs fail the gate.** They're never skipped.
3. **The verdict is mechanical.** `make gate ARM=…` prints pass or fail with every comparison shown.

### 13.3 The gate

A candidate arm passes if:
- it **wins** in S2 (cold > 5 s share at least 10 pp below residency) **or** in S4 (giant p99 below residency by more than the noise band), on every gate seed;
- **and** it **loses nowhere**: every guard holds in S1, S2, S3 and S4. The guards are "cold p99 no worse than the band", "warm or cold > 5 s share at most 1–5 pp worse", and "timeouts no higher".

The comparison is against `residency-w4`, and also against `residency-w4-rr2` in S3 and S4.

---

## 14. Worked example: reading a real result

The first real run, `make quick-s1` on 4× NVIDIA B200 with vLLM 0.31.0, produced this. It's one seed, so it's preliminary:

| arm | cold TTFT p99 | cold blocked at arrival | hot TTFT p99 | base TTFT p99 | pods per adapter | activations |
| --- | --- | --- | --- | --- | --- | --- |
| stock | 2.19 s | 69.2% | 1.35 s | 0.04 s | 4 | 410 |
| residency-w4 | 0.03 s | 0% | 0.03 s | 0.03 s | 1 | 4 |

Read it in this order:

1. **Is it valid?** Both runs passed every check: client lag about 1.5 ms, all requests attributed, no errors. The numbers can be read.
2. **Is the pool overloaded?** Base TTFT is 0.03–0.04 s in both arms. The cluster as a whole is fine, so any slowness is about *adapters*, not capacity.
3. **What happened to cold requests under stock?** Their p99 is 2.19 s, about 50 times the base model's. The mechanism column explains why: **69% of cold requests arrived at a pod where both slots were held by other adapters**. That's the starvation from §1.3, measured directly.
4. **Why?** Stock served every adapter on all 4 pods (pods per adapter = 4), so hot adapters held slots everywhere and adapters were swapped onto GPUs 410 times in 3 minutes. That's thrash and starvation together.
5. **Residency** kept each adapter on one pod. Nothing was ever blocked, and only 4 activations happened after the start. Every class sees about 30 ms.
6. **How does it compare with #3392?** atantawi's 4×H100 study reported stock cold p99 of 6.95–7.73 s and residency about 0.2 s. On B200 the starvation is smaller, but **the regime reproduces**: same direction, same mechanism. Numbers aren't expected to match across hardware, so a result like this is reported as "regime reproduced", never "numbers match".
7. **What it doesn't show yet:**
   - whether this holds on other seeds;
   - how big the run-to-run noise is;
   - how residency behaves when no good layout exists (S2) or when an adapter outgrows a pod (S3, S4).

   That's what the full sets are for.

This example is also why S1 is a *guard* and not a *win* scenario. Residency already reaches about 30 ms here, so a new filter has nothing to win. It can only fail to lose.

---

## 15. Summary and glossary

**In one paragraph.** GPUs serving many LoRA adapters can hold only a few at a time. Whichever adapters a router puts together on a pod decide whether rare adapters starve or the GPU thrashes. llmd-lora-routing-bench runs the real llm-d router (EPP) and Envoy in front of real vLLM pods, without Kubernetes, and replays identical seeded traffic against each router configuration ("arm"). It reports latency per traffic class together with the mechanism behind it, throws away any run that fails its validity checks, and applies a rule fixed in advance to decide whether a new placement filter beats the best existing router.

| Term | Short definition |
| --- | --- |
| A/A run | the same arm run twice on the same seed, to measure noise |
| activation | moving an adapter from CPU memory into a GPU slot |
| adapter (LoRA) | a small add-on to a base model that specialises it |
| arm | one router configuration under test |
| blocked at arrival | a request reaching a pod whose slots are all held by other adapters |
| calibration | measuring one pod's capacity and load costs on this hardware |
| closed / open loop | waiting for answers before sending, versus sending on schedule |
| coordinated omission | the under-measurement caused by closed-loop testing |
| EPP | Endpoint Picker: the llm-d component that picks a pod per request |
| Envoy, ext_proc | the proxy carrying requests, and its side channel for asking the EPP |
| feasibility bound | an offline answer to "could any placement fit this traffic?" |
| file discovery | the EPP reading the pod list from a file instead of the Kubernetes API |
| gate | the pre-registered rule a new filter must pass |
| knee | the rate past which latency rises sharply; it defines capacity |
| load | moving an adapter from storage into CPU memory |
| noise band | how much a metric moves between identical runs |
| oracle | a hand-written placement, used to estimate how much better routing could do |
| p99 | the latency that 99% of requests beat |
| pod | one engine process with its own GPU |
| scenario | one workload: pods, slots, adapters, traffic |
| schedule, seed | the exact request list for a run, and the number that reproduces it |
| simulator | llm-d-inference-sim, a GPU-free stand-in for vLLM used to check the harness |
| slot | room on the GPU for one adapter (`--max-loras` of them per pod) |
| slot holding | an adapter keeping its slot because it has a request running |
| starvation | a request waiting indefinitely because others always hold the slots |
| steady window | the part of a run after warm-up, used for decisions |
| thrash | adapters constantly swapped in and out of slots |
| TTFT | time to first token |

**Where to go next:**
- [README.md](README.md) to run it;
- [DESIGN_LLMD_709.md](DESIGN_LLMD_709.md) for the exact specification;
- [llm-d-router#709](https://github.com/llm-d/llm-d-router/issues/709) for the discussion this benchmark serves.
