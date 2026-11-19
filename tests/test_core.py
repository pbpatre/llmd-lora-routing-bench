# SPDX-License-Identifier: Apache-2.0
import datetime as dt
import json
import math
from pathlib import Path

import pytest
import yaml

from lorabench import analyze, arms, calibrate, collect, compare, runner, schedule, topology
from lorabench.loadgen import chunk_has_token, oracle_pods, usage_tokens
from lorabench.scenario import load
from lorabench.scrape import pod_sample

ROOT = Path(__file__).resolve().parents[1]
SCENARIOS = sorted((ROOT / "scenarios").glob("*.yaml"))


@pytest.mark.parametrize("path", SCENARIOS, ids=lambda p: p.stem)
def test_scenario_loads(path):
    sc = load(path, capacity_rps=4.0)
    assert sc.adapters()
    assert sum(schedule.target_probabilities(sc).values()) == pytest.approx(1.0)


def test_rate_c_requires_capacity():
    with pytest.raises(ValueError, match="rate_c"):
        load(ROOT / "scenarios" / "s3-two-hot.yaml")


def test_s4_shares_match_follow_up_1a():
    # Each hot adapter gets 1.15 x one-pod capacity at 0.8 of cluster capacity.
    sc = load(ROOT / "scenarios" / "s3-two-hot.yaml", capacity_rps=10.0)
    p = schedule.target_probabilities(sc)
    assert sc.rate * p["hot-0"] == pytest.approx(11.5, rel=0.01)


def test_schedule_is_deterministic_and_matches_mix():
    sc = load(ROOT / "scenarios" / "s1-fits.yaml")
    a, b, c = schedule.generate(sc, 1), schedule.generate(sc, 1), schedule.generate(sc, 2)
    assert a == b and a != c
    n = len(a)
    expected = sc.rate * sc.duration_s
    assert abs(n - expected) < 5 * math.sqrt(expected)
    for target, p in schedule.target_probabilities(sc).items():
        got = sum(r.adapter == target for r in a) / n
        assert abs(got - p) < 4 * math.sqrt(p * (1 - p) / n) + 1e-9, target
    assert len({r.prompt for r in a}) == n
    assert all(r.prompt.startswith(f"[{r.idx}]") for r in a)  # unique nonce first


def test_zipf_weights():
    w = schedule._weights(4, 1.0)
    assert w[0] / w[1] == pytest.approx(2.0)


@pytest.mark.parametrize("arm", sorted(arms.ARMS))
@pytest.mark.parametrize("path", SCENARIOS, ids=lambda p: p.stem)
def test_epp_config_renders(arm, path):
    sc = load(path, capacity_rps=4.0)
    if arm == "oracle" and not sc.oracle:
        with pytest.raises(ValueError):
            arms.epp_config(arm, sc, "base", "/x/endpoints.yaml")
        return
    cfg = yaml.safe_load(arms.epp_config(arm, sc, "base", "/x/endpoints.yaml"))
    names = {p.get("name", p["type"]) for p in cfg["plugins"]}
    refs = [r["pluginRef"] for r in cfg["schedulingProfiles"][0]["plugins"]]
    assert set(refs) <= names
    assert cfg["dataLayer"]["discovery"]["endpoints"]["pluginRef"] == "file-disc"
    if arm.startswith("residency"):
        res = next(p for p in cfg["plugins"] if p["type"] == "lora-residency-scorer")
        assert res["parameters"]["maxCpuLoras"] == sc.max_cpu_loras
        assert res["parameters"]["baseModels"] == ["base"]


def test_endpoints_and_envoy_config(tmp_path):
    eps = yaml.safe_load(arms.endpoints_file([("pod-0", "127.0.0.2", 8100)]))
    assert eps["endpoints"][0]["labels"][arms.POD_LABEL] == "pod-0"
    cfg = topology.envoy_config(topology.Ports(), tmp_path / "a.jsonl")
    hcm = cfg["static_resources"]["listeners"][0]["filter_chains"][0]["filters"][0]["typed_config"]
    ext = hcm["http_filters"][0]["typed_config"]
    assert ext["processing_mode"]["response_body_mode"] == "FULL_DUPLEX_STREAMED"
    route = hcm["route_config"]["virtual_hosts"][0]["routes"][0]["route"]
    assert route["timeout"] == "0s"
    upstream = next(c for c in cfg["static_resources"]["clusters"] if c["name"] == "original_destination_cluster")
    opts = upstream["typed_extension_protocol_options"]["envoy.extensions.upstreams.http.v3.HttpProtocolOptions"]
    assert opts["common_http_protocol_options"]["idle_timeout"] == "3s"  # below vLLM's 5 s keep-alive


def test_latin_order_balances_first_arm():
    order = runner.latin_order(["a", "b", "c"], [1, 2, 3])
    firsts = [order[i * 3][1] for i in range(3)]
    assert sorted(firsts) == ["a", "b", "c"]


def test_latin_order_brackets_each_seed_with_aa():
    order = runner.latin_order(["stock", "residency-w4", "affinity-w2"], [1, 2], aa_arm="residency-w4")
    block1 = [x for x in order if x[0] == 1]
    assert block1[0] == (1, "residency-w4", 0) and block1[-1] == (1, "residency-w4", 1)
    assert sum(1 for x in block1 if x[1] == "residency-w4") == 2


def test_epp_config_pins_saturation_filter():
    sc = load(ROOT / "scenarios" / "s1-fits.yaml")
    cfg = yaml.safe_load(arms.epp_config("stock", sc, "base", "/x"))
    assert cfg["flowControl"]["saturationDetector"]["pluginRef"] == "utilization-detector"
    assert cfg["schedulingProfiles"][0]["plugins"][0]["pluginRef"] == "utilization-detector"


def test_calibration_lookup_sets_limits():
    sc = load(ROOT / "scenarios" / "s1-fits.yaml")
    cal = {"lengths": {"256": {"capacity_rps": 9.0, "e2e_unloaded_s": 2.5, "ttft_p99_unloaded_s": 0.05}}}
    got = runner.calibration_for(cal, sc)
    assert got == {"capacity_rps": 9.0, "e2e_unloaded_s": 2.5, "base_ttft_p99_limit_s": pytest.approx(0.15)}
    with pytest.raises(ValueError):
        runner.calibration_for({"lengths": {}}, sc)
    assert runner._seconds("1s") == 1.0 and runner._seconds("30ms") == pytest.approx(0.03)


def test_base_check_off_skips_the_base_limit():
    sc = load(ROOT / "scenarios" / "s3-two-hot.yaml", capacity_rps=4.0)
    cal = {"lengths": {"1024": {"capacity_rps": 4.0, "e2e_unloaded_s": 10.0, "ttft_p99_unloaded_s": 0.05}}}
    assert "base_ttft_p99_limit_s" not in runner.calibration_for(cal, sc)
    assert "base_ttft_p99_limit_s" not in runner.sim_calibration(sc)


def test_oracle_round_robin():
    reqs = [schedule.Request(i, float(i), a, "c", "p", 4) for i, a in enumerate(["g", "x", "g", "g", "g"])]
    got = oracle_pods(reqs, {"g": ["pod-0", "pod-1", "pod-2"]})
    assert got == {0: "pod-0", 2: "pod-1", 3: "pod-2", 4: "pod-0"}


def test_chunk_scanning():
    assert not chunk_has_token(b'{"choices":[{"delta":{"role":"assistant","content":""}}]}')
    assert chunk_has_token(b'{"choices":[{"delta":{"content":"hi"}}]}')
    assert not chunk_has_token(b'{"choices":[],"usage":{"completion_tokens":3}}')
    assert usage_tokens(b'{"choices":[],"usage":{"prompt_tokens":5,"completion_tokens":3}}') == 3
    assert usage_tokens(b'{"choices":[{"delta":{"content":"x"}}]}') is None


def test_lora_info_newest_series_wins():
    text = "\n".join([
        'vllm:lora_requests_info{max_lora="2",running_lora_adapters="a,b",waiting_lora_adapters=""} 100.0',
        'vllm:lora_requests_info{max_lora="2",running_lora_adapters="a",waiting_lora_adapters="c"} 200.0',
        'vllm:num_requests_running{model_name="m"} 3',
        'vllm:num_requests_waiting{model_name="m"} 1',
    ])
    s = pod_sample(text)
    assert s["lora"] == {"running": ["a"], "waiting": ["c"], "max_lora": 2, "ts": 200.0}
    assert s["running"] == 3 and s["waiting"] == 1


def test_vllm_load_log_parsing(tmp_path):
    log = tmp_path / "pod-0.log"
    log.write_text("\n".join([
        "2026-10-26T10:00:00.000 DEBUG vllm.lora.model_manager Adding lora. Model id: 7, int id: 1",
        # vLLM V1: the LoRA manager logs from EngineCore, with a prefix and maybe colour codes.
        "(EngineCore pid=4242) 2026-10-26T10:00:05.000 DEBUG vllm.lora.model_manager Adding lora. Model id: 8, int id: 2",
        "\x1b[1;36m(EngineCore pid=4242)\x1b[0m 2026-10-26T10:00:05.100 DEBUG vllm.lora.model_manager Activating LoRA. int id: 2, slot index: 0",
        "2026-10-26T10:00:06.000 INFO vllm.engine noise",
    ]))
    after = dt.datetime.fromisoformat("2026-10-26T10:00:01").timestamp()
    loads = collect.vllm_loads(log, "pod-0", {1: "a", 2: "b"}, after)
    assert [(e["kind"], e["adapter"]) for e in loads] == [("storage_to_cpu", "b"), ("cpu_to_gpu", "b")]


def test_access_log_join():
    reqs = [{"rid": "r-1"}, {"rid": "r-2"}]
    acc = {"r-1": {"rid": "r-1", "upstream": "127.0.0.1:8100", "start": "1000.5", "us_tx_begin_us": "2500",
                   "flags": "-"}}
    collect.join_access(reqs, acc, wall_t0=1000.0)
    assert reqs[0]["upstream"] == "127.0.0.1:8100"
    assert reqs[0]["t_upstream"] == pytest.approx(0.5025)
    assert "upstream" not in reqs[1]


# ------------------------------------------------------------------ analysis

A, B = "10.0.0.1:1", "10.0.0.2:1"


def _req(idx, adapter, cls, pod, send, first, end, status="ok", tokens=8):
    return {"idx": idx, "rid": f"r-{idx}", "adapter": adapter, "cls": cls, "t_sched": send, "t_send": send,
            "send_lag": 0.001, "t_first_byte": first, "t_first_token": first, "t_end": end,
            "chunks": tokens, "completion_tokens": tokens if status == "ok" else None, "status": status,
            "http": 200, "served_by": pod, "upstream": pod, "t_upstream": send, "error": None}


def _write_run(tmp: Path, reqs, **meta_extra):
    meta = {"run_id": "t", "arm": "x", "seed": 0, "backend": "sim",
            "addr_to_pod": {A: "pod-0", B: "pod-1"}, "prefill_estimate_s": 0.0,
            "loadgen": {"loop_lag_p99_s": 0.001},
            "scenario": {"name": "t", "pods": 2, "max_loras": 2, "duration_s": 120.0, "steady_after_s": 0.0,
                         "timeout_s": 120.0, "output_tokens": 8}}
    meta.update(meta_extra)
    (tmp / "meta.json").write_text(json.dumps(meta))
    (tmp / "requests.jsonl").write_text("".join(json.dumps(r) + "\n" for r in reqs))
    (tmp / "access.jsonl").write_text("")  # contents already joined into the requests
    for tag in ("before", "after"):
        (tmp / f"envoy-stats-{tag}.json").write_text(json.dumps({"stats": [
            {"name": "cluster.original_destination_cluster.upstream_rq_pending_overflow", "value": 0}]}))


def _starvation_reqs():
    return [
        _req(0, "hot-0", "hot", A, 0.0, 0.0, 120.0),    # holds a pod-0 slot all run
        _req(1, "hot-1", "hot", A, 0.0, 0.0, 120.0),    # holds the other pod-0 slot
        _req(2, "hot-2", "hot", B, 0.0, 0.0, 120.0),    # alone on pod-1
        _req(3, "cold-0", "cold", A, 20.0, 27.0, 28.0),  # blocked, waits 7 s
        _req(4, "cold-1", "cold", B, 30.0, 30.05, 31.0),  # not blocked
        _req(5, "cold-2", "cold", A, 40.0, None, 99.0, status="timeout"),
    ]


def test_starvation_diagnostics(tmp_path):
    _write_run(tmp_path, _starvation_reqs())
    res = analyze.analyze(tmp_path)
    cold = res["full"]["classes"]["cold"]
    assert cold["timeouts"] == 1 and cold["ttft_p99"] == "inf"
    assert cold["share_over_5s"] == pytest.approx(2 / 3)
    assert res["blocked_at_arrival"]["cold"] == pytest.approx(2 / 3)
    assert res["pods_per_adapter"]["hot-0"] == 1
    lay = res["layout"]
    assert lay["n_windows"] == 4
    w0 = lay["windows"][0]  # 0-30 s: cold-0 runs 27-28 s on pod-0
    assert w0["pods"]["pod-0"]["holding"] == ["hot-0", "hot-1"]
    assert w0["pods"]["pod-0"]["violation"] is True
    assert w0["pods"]["pod-1"]["violation"] is False
    assert w0["free_slots"] == 1
    assert res["slot_holding"]["pod-0/hot-0"]["running_share"] == pytest.approx(1.0)
    v = res["validity"]
    assert v["valid"] is True


def test_validity_requires_the_access_log(tmp_path):
    reqs = _starvation_reqs()
    _write_run(tmp_path, reqs)
    (tmp_path / "access.jsonl").unlink()
    assert analyze.analyze(tmp_path)["validity"]["valid"] is False
    reqs[4]["upstream"] = None  # served (header present) but missing from the access log
    _write_run(tmp_path, reqs)
    v = analyze.analyze(tmp_path)["validity"]
    assert v["unattributed"] == 1 and v["valid"] is False


def test_validity_flags_disagreement_and_lag(tmp_path):
    reqs = _starvation_reqs()
    reqs[4]["served_by"] = A  # header says pod-0, access log says pod-1
    reqs[3]["send_lag"] = 0.2
    _write_run(tmp_path, reqs)
    v = analyze.analyze(tmp_path)["validity"]
    assert v["header_disagrees_with_access_log"] == 1
    assert v["valid"] is False


def test_validity_requires_parsed_loads_on_vllm(tmp_path):
    _write_run(tmp_path, _starvation_reqs(), backend="vllm")
    v = analyze.analyze(tmp_path)["validity"]
    assert v["loads_parsed"] is False and v["valid"] is False


def test_validity_requires_envoy_stats_and_usage(tmp_path):
    reqs = _starvation_reqs()
    reqs[0]["completion_tokens"] = None
    _write_run(tmp_path, reqs)
    (tmp_path / "envoy-stats-after.json").unlink()
    v = analyze.analyze(tmp_path)["validity"]
    assert v["missing_usage"] == 1 and v["envoy_stats_present"] is False and v["valid"] is False


def test_blocked_counts_holders_that_arrived_before_the_window(tmp_path):
    reqs = _starvation_reqs()
    _write_run(tmp_path, reqs)
    meta = json.loads((tmp_path / "meta.json").read_text())
    meta["scenario"]["steady_after_s"] = 10.0  # the hot holders arrived at t=0
    (tmp_path / "meta.json").write_text(json.dumps(meta))
    assert analyze.analyze(tmp_path)["blocked_at_arrival"]["cold"] == pytest.approx(2 / 3)


def test_feasibility_bound():
    meta = {"scenario": {"pods": 4, "max_loras": 2, "capacity_rps": 10.0}, "e2e_unloaded_s": 2.6,
            "offered_rates": {"__base__": 2.4, "hot-0": 4.32, "hot-1": 4.32,
                              "cold-0": 0.24, "cold-1": 0.24, "cold-2": 0.24, "cold-3": 0.24}}
    fb = analyze.feasibility_bound(meta)
    assert fb["holding_slots_needed"] == 2  # S1: two hot adapters, one pod each
    assert fb["slots_fit"] and fb["compute_fits"]
    meta["offered_rates"] = {f"warm-{i}": 1.8 for i in range(8)} | {f"cold-{i}": 0.4 for i in range(4)}
    meta["e2e_unloaded_s"] = 10.3
    fb = analyze.feasibility_bound(meta)
    # S2's "cold" adapters run at 0.4 req/s x 10.3 s ~ 4 concurrent requests, so they
    # hold slots too: 12 slot-holding adapters for 8 slots.
    assert fb["holding_slots_needed"] == 12 and not fb["slots_fit"]


def test_percentile_nearest_rank_with_censoring():
    assert analyze.pct([1.0] * 98 + [2.0, math.inf], 99) == 2.0
    assert analyze.pct([1.0] * 98 + [2.0, math.inf, math.inf], 99) == math.inf


# ------------------------------------------------------------------ gate

def test_misses_counts_timeouts_and_skips_small_adapters():
    reqs = _starvation_reqs()  # hot-0/1/2 fast; cold-0 waits 7 s; cold-1 fast; cold-2 times out
    m = analyze.misses(reqs, threshold=0.5, min_n=1)
    assert (m["n"], m["misses"]) == (6, 2) and m["miss_rate"] == pytest.approx(1 / 3)
    assert m["classes"]["cold"]["miss_rate"] == pytest.approx(2 / 3)
    assert m["worst_adapter"]["miss_rate"] == 1.0
    assert analyze.misses(reqs, threshold=0.5, min_n=20)["worst_adapter"] is None  # every adapter has 1 request


def _summary(sc, arm, seed, misses=0, n=1000, worst=0.0, rep=0, valid=True, backend="vllm", classes=None):
    return {"run_id": f"{sc}-{arm}-{seed}-{rep}", "scenario": sc, "arm": arm, "seed": seed, "replicate": rep,
            "valid": valid, "backend": backend, "validity": {"valid": valid}, "blocked_at_arrival": {},
            "steady": {"classes": classes or {}},
            "misses": {"n": n, "misses": misses, "miss_rate": misses / n,
                       "worst_adapter": {"adapter": "a", "miss_rate": worst}, "classes": {}, "adapters": {}}}


def _gate_runs(cand=None, ref=None):
    """Seeds 3-5 for every gate scenario. `cand` / `ref` map scenario -> per-seed (misses per 1000, worst)."""
    cand, ref = cand or {}, ref or {}
    good, tipped = (0, 0.0), (300, 0.6)
    runs = []
    for sc in compare.GATE:
        default_ref = [tipped, good, tipped] if compare.GATE[sc] == "win" else [good] * 3
        for arm, per in (("cand", cand.get(sc, [good] * 3)),
                         ("residency-w4", ref.get(sc, default_ref)), ("residency-w4-rr2", ref.get(sc, default_ref))):
            for seed, (k, w) in zip(compare.GATE_SEEDS, per):
                runs.append(_summary(sc, arm, seed, misses=k, worst=w))
    return runs


def test_gate_passes_when_the_candidate_never_tips():
    rep = compare.gate(_gate_runs(), "cand")
    assert rep["passes"] is True
    s3 = rep["scenarios"]["s3-two-hot"]
    assert s3["metrics"]["residency-w4"]["bad_seeds"] == 2 and s3["metrics"]["cand"]["bad_seeds"] == 0


def test_gate_needs_wins_in_both_s3_and_s4():
    ref_good = {"s4-one-giant": [(0, 0.0)] * 3}  # nothing to win in S4
    assert compare.gate(_gate_runs(ref=ref_good), "cand")["passes"] is False


def test_gate_fails_on_more_bad_seeds_even_with_a_lower_mean():
    # Candidate: 6% on one seed (tipped), 0 elsewhere -> 2.0% pooled.
    # Reference: 4.5% on every seed (never over the 5% line) -> 4.5% pooled.
    rep = compare.gate(_gate_runs(cand={"s3-two-hot": [(60, 0.3), (0, 0.0), (0, 0.0)]},
                                  ref={"s3-two-hot": [(45, 0.3)] * 3}), "cand")
    vs = rep["scenarios"]["s3-two-hot"]["vs"]["residency-w4"]
    assert vs["miss_rate"] is True and vs["bad_seeds"] is False and rep["passes"] is False


def test_gate_fairness_guard_catches_a_starved_adapter():
    # S1 guard: overall miss rate unchanged, but one adapter starves.
    rep = compare.gate(_gate_runs(cand={"s1-fits": [(5, 0.4), (0, 0.0), (0, 0.0)]}), "cand")
    assert rep["scenarios"]["s1-fits"]["vs"]["residency-w4"] == {"miss_rate": True, "fairness": False}
    assert rep["passes"] is False


def test_gate_guard_allows_its_margin():
    within = compare.gate(_gate_runs(cand={"s5-zipf": [(10, 0.0)] * 3}), "cand")   # +1.0 point
    beyond = compare.gate(_gate_runs(cand={"s5-zipf": [(11, 0.0)] * 3}), "cand")   # +1.1 points
    assert within["passes"] is True and beyond["passes"] is False


def test_gate_fails_on_missing_or_invalid_runs():
    runs = [r for r in _gate_runs() if not (r["scenario"] == "s4-one-giant" and r["arm"] == "residency-w4-rr2")]
    rep = compare.gate(runs, "cand")
    assert rep["scenarios"]["s4-one-giant"]["missing_or_invalid"] and rep["passes"] is False
    runs = _gate_runs()
    next(r for r in runs if r["arm"] == "cand" and r["seed"] == 4)["valid"] = False
    assert compare.gate(runs, "cand")["passes"] is False


def test_gate_ignores_tuning_seeds():
    runs = [dict(r, seed=r["seed"] - 2) for r in _gate_runs()]  # seeds 1-3 only
    assert compare.gate(runs, "cand")["passes"] is False


def test_report_refuses_mixed_backends():
    runs = [_summary("s1-fits", "stock", 1, backend="sim"), _summary("s1-fits", "stock", 1)]
    with pytest.raises(ValueError):
        compare.one_backend(runs, None)
    assert len(compare.one_backend(runs, "sim")) == 1


def test_duplicate_valid_runs_are_an_error():
    runs = [_summary("s1-fits", "stock", 1), _summary("s1-fits", "stock", 1)]
    with pytest.raises(ValueError, match="two valid runs"):
        compare.valid_index(runs)
    runs[0]["valid"] = False  # an invalid run and its valid re-run are fine
    assert len(compare.valid_index(runs)) == 1


def test_gpu_ids():
    gpu_ids = topology.gpu_ids
    assert gpu_ids(4, "") == [0, 1, 2, 3]
    assert gpu_ids(4, "4,5,6,7") == [4, 5, 6, 7]
    assert gpu_ids(2, "6, 7, 0") == [6, 7]
    for bad in ("1,2,3", "1,1,2,3"):
        with pytest.raises(ValueError):
            gpu_ids(4, bad)


def test_port_offset(monkeypatch):
    assert topology.Ports().envoy == 8080
    monkeypatch.setenv("LORABENCH_PORT_OFFSET", "1000")
    p = topology.Ports()
    assert (p.envoy, p.pod_base, p.epp_metrics) == (9080, 9100, 10090)


def test_envoy_idle_connection_closes_are_not_overflow(tmp_path):
    _write_run(tmp_path, _starvation_reqs())
    stats = {"stats": [{"name": "cluster.original_destination_cluster.upstream_cx_idle_timeout", "value": 600},
                       {"name": "http.inference.downstream_rq_idle_timeout", "value": 0}]}
    (tmp_path / "envoy-stats-after.json").write_text(json.dumps(stats))
    assert analyze.envoy_overflow(tmp_path) == {}
    stats["stats"][1]["value"] = 2
    (tmp_path / "envoy-stats-after.json").write_text(json.dumps(stats))
    assert analyze.envoy_overflow(tmp_path) == {"http.inference.downstream_rq_idle_timeout": 2}


def test_capacity_is_the_ttft_knee():
    # Shape of the measured B200 curve at L=1024: TTFT p99 flat, then a knee between 14 and 20 req/s.
    curve = [{"rate": r, "ttft_p99": t, "itl_p50": i, "completed_fraction": 1.0}
             for r, t, i in [(1, 0.030, 0.0059), (8, 0.040, 0.0077), (14, 0.072, 0.0116),
                             (16, 0.124, 0.0126), (20, 3.7, 0.0138)]]
    assert calibrate.capacity(curve, {"ttft_p99_factor": 4.0, "completed_fraction": 0.98}) == 14
    curve[2]["completed_fraction"] = 0.9  # dropped requests end the sweep too
    assert calibrate.capacity(curve, {"ttft_p99_factor": 4.0, "completed_fraction": 0.98}) == 8


def test_queue_trend_flags_a_pod_that_never_drains():
    scrapes = [{"src": p, "t": 1000.0 + t, "waiting": w} for t in range(60, 180)
               for p, w in (("pod-0", 2), ("pod-1", (t - 60)))]  # pod-1's queue grows 1 per second
    q = analyze.queue_trend(scrapes, ["pod-0", "pod-1"], wall_t0=1000.0, start=60, end=180)
    assert q["per_pod"]["pod-0"] == {"waiting_start": 2, "waiting_end": 2}
    assert q["growing_queue_pods"] == ["pod-1"]
