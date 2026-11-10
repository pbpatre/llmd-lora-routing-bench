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

def _summary(sc, arm, seed, classes, rep=0, valid=True, backend="vllm"):
    return {"run_id": f"{sc}-{arm}-{seed}-{rep}", "scenario": sc, "arm": arm, "seed": seed, "replicate": rep,
            "valid": valid, "backend": backend,
            "validity": {"valid": valid}, "blocked_at_arrival": {}, "steady": {"classes": classes}}


def _cls(p99=1.0, over=0.0, n=100, timeouts=0):
    return {"ttft_p99": p99, "share_over_5s": over, "n": n, "timeouts": timeouts}


def _gate_runs(cand_s3_cold_over, cand_s3_warm_over=0.10, cand_giant_p99=2.0):
    runs = []
    for s in (4, 5, 6):
        arms_ = (("residency-w4", (0.80, 0.08), 2.0), ("residency-w4-rr2", None, 2.0),
                 ("cand", (cand_s3_cold_over, cand_s3_warm_over), cand_giant_p99))
        for arm, s3, giant in arms_:
            if s3 is not None:
                runs.append(_summary("s1-fits", arm, s, {"cold": _cls(0.2), "hot": _cls(0.05)}))
                runs.append(_summary("s2-oversubscribed", arm, s,
                                     {"cold": _cls(over=s3[0]), "warm": _cls(over=s3[1]), "base": _cls()}))
            runs.append(_summary("s3-two-hot", arm, s, {"hot": _cls(0.3), "cold": _cls(), "base": _cls()}))
            runs.append(_summary("s4-one-giant", arm, s, {"giant": _cls(giant), "cold": _cls()}))
    return runs


def test_gate_passes_on_clear_s3_win():
    rep = compare.gate(_gate_runs(cand_s3_cold_over=0.30), "cand")
    assert rep["scenarios"]["s2-oversubscribed"]["win"]["won"] is True
    assert rep["passes"] is True


def test_gate_fails_when_win_costs_warm_adapters():
    rep = compare.gate(_gate_runs(cand_s3_cold_over=0.30, cand_s3_warm_over=0.20), "cand")
    assert rep["passes"] is False


def test_gate_fails_without_win_or_with_missing_data():
    assert compare.gate(_gate_runs(cand_s3_cold_over=0.75), "cand")["passes"] is False
    runs = [r for r in _gate_runs(0.30) if not (r["scenario"] == "s4-one-giant" and r["arm"] == "cand")]
    assert compare.gate(runs, "cand")["passes"] is False


def test_gate_fails_when_one_seed_is_invalid():
    runs = _gate_runs(0.30)
    for r in runs:
        if r["scenario"] == "s2-oversubscribed" and r["arm"] == "cand" and r["seed"] == 5:
            r["valid"] = False
    assert compare.gate(runs, "cand")["passes"] is False


def test_gate_s5_win_needs_rr2_present():
    runs = [r for r in _gate_runs(0.75, cand_giant_p99=0.5)
            if not (r["scenario"] == "s4-one-giant" and r["arm"] == "residency-w4-rr2")]
    rep = compare.gate(runs, "cand")
    assert rep["scenarios"]["s4-one-giant"]["missing_or_invalid"]
    assert rep["passes"] is False
    assert compare.gate(_gate_runs(0.75, cand_giant_p99=0.5), "cand")["passes"] is True


def test_gate_ignores_tuning_seeds():
    runs = [dict(r, seed=r["seed"] - 3) for r in _gate_runs(0.30)]  # seeds 1-3 only
    assert compare.gate(runs, "cand")["passes"] is False


def test_report_refuses_mixed_backends():
    runs = [_summary("s1-fits", "stock", 1, {}, backend="sim"), _summary("s1-fits", "stock", 1, {})]
    with pytest.raises(ValueError):
        compare.one_backend(runs, None)
    assert len(compare.one_backend(runs, "sim")) == 1


def test_noise_band_widens_with_aa_spread():
    runs = [_summary("s1-fits", "residency-w4", 1, {"cold": _cls(1.0)}),
            _summary("s1-fits", "residency-w4", 1, {"cold": _cls(2.0)}, rep=1)]
    idx = compare.valid_index(runs)
    lo, hi = compare.noise_band(idx, "s1-fits", "cold")
    assert hi == pytest.approx(2.0) and lo == pytest.approx(0.5)
    assert compare.noise_band(idx, "s1-fits", "hot") == compare.P99_BAND_3392  # no A/A data for that class


def test_gate_timeouts_guard_allows_its_margin():
    # S2 allows the candidate 1% of the steady requests in extra timeouts (n = 100 per class here).
    def s3_runs(extra):
        runs = _gate_runs(cand_s3_cold_over=0.30)
        for r in runs:
            if r["scenario"] == "s2-oversubscribed" and r["arm"] == "cand":
                r["steady"]["classes"]["cold"]["timeouts"] = extra
        return runs
    guard = lambda rep: next(g for g in rep["scenarios"]["s2-oversubscribed"]["guards"] if g["metric"] == "timeouts")
    assert guard(compare.gate(s3_runs(3), "cand"))["ok"] is True    # 3 <= 0 + 1% of 300
    assert guard(compare.gate(s3_runs(4), "cand"))["ok"] is False


def test_duplicate_valid_runs_are_an_error():
    runs = [_summary("s1-fits", "stock", 1, {}), _summary("s1-fits", "stock", 1, {})]
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
    assert calibrate.capacity(curve, {"ttft_p99_factor": 4.0, "itl_p50_factor": 1.3,
                                      "completed_fraction": 0.98}) == 1


def test_p99_rules_ignore_differences_below_the_floor():
    band = (0.7, 1.41)
    assert compare._check("ratio_max", 0.07, 0.03, None, band, 0) is True     # +40 ms is not a regression
    assert compare._check("ratio_max", 0.60, 0.30, None, band, 0) is False    # 2x and +0.3 s is
    assert compare._check("ratio_lower", 0.02, 0.10, None, band, 0) is False  # 5x faster but only 80 ms
    assert compare._check("ratio_lower", 0.2, 12.0, None, band, 0) is True


def test_queue_trend_flags_a_pod_that_never_drains():
    scrapes = [{"src": p, "t": 1000.0 + t, "waiting": w} for t in range(60, 180)
               for p, w in (("pod-0", 2), ("pod-1", (t - 60)))]  # pod-1's queue grows 1 per second
    q = analyze.queue_trend(scrapes, ["pod-0", "pod-1"], wall_t0=1000.0, start=60, end=180)
    assert q["per_pod"]["pod-0"] == {"waiting_start": 2, "waiting_end": 2}
    assert q["growing_queue_pods"] == ["pod-1"]
