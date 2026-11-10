# SPDX-License-Identifier: Apache-2.0
"""Launch and tear down one run's processes: pods, EPP, Envoy.

No Kubernetes. Pods are llm-d-inference-sim or vLLM processes, EPP uses file
discovery, and Envoy forwards to the endpoint EPP picks (ORIGINAL_DST).
"""

from __future__ import annotations

import dataclasses
import json
import os
import resource
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from . import arms
from .scenario import Scenario
from .scrape import parse

ROOT = Path(__file__).resolve().parents[1]
BASE_MODEL = "base"  # served name of the base model on every backend
SERVED_BY_HEADER = "x-served-by"  # set by Envoy to the upstream it picked
SIM_DEFAULTS = {"ttft": "30ms", "itl": "10ms", "time_factor": 3.0, "max_num_seqs": 64}


@dataclasses.dataclass
class Ports:
    """Local ports for one run. LORABENCH_PORT_OFFSET shifts them all, so two runs can share a node."""
    envoy: int = 8080
    envoy_admin: int = 19000
    epp_grpc: int = 9002
    epp_health: int = 9003
    epp_metrics: int = 9090
    pod_base: int = 8100

    def __post_init__(self):
        off = int(os.environ.get("LORABENCH_PORT_OFFSET", "0") or 0)
        if off:
            for f in dataclasses.fields(self):
                setattr(self, f.name, getattr(self, f.name) + off)


@dataclasses.dataclass
class Pod:
    name: str
    ip: str
    port: int
    gpu: int | None = None

    @property
    def addr(self) -> str:
        return f"{self.ip}:{self.port}"

    @property
    def url(self) -> str:
        return f"http://{self.addr}"


def tool(name: str, env: str) -> str:
    """Resolve a binary: $env, then .cache/bin, then PATH."""
    path = os.environ.get(env) or str(ROOT / ".cache" / "bin" / name)
    if Path(path).exists():
        return path
    found = shutil.which(name)
    if not found:
        raise FileNotFoundError(f"{name} not found; set {env} or put it in .cache/bin")
    return found


def raise_nofile() -> None:
    """Envoy and the load shards hold one descriptor per stream; macOS defaults to 256."""
    _, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (min(hard, 65536), hard))


class Procs:
    MONITORS = {"nvidia-smi"}  # may exit on their own without failing the run

    def __init__(self, logdir: Path):
        self.logdir = logdir
        self.procs: list[tuple[str, subprocess.Popen]] = []

    def start(self, name: str, cmd: list[str], env: dict | None = None) -> subprocess.Popen:
        log = (self.logdir / f"{name}.log").open("wb")
        (self.logdir / f"{name}.cmd").write_text(json.dumps(cmd))
        p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env={**os.environ, **(env or {})},
                             start_new_session=True)
        self.procs.append((name, p))
        return p

    def check(self) -> None:
        dead = [(n, p.returncode) for n, p in self.procs if n not in self.MONITORS and p.poll() is not None]
        if dead:
            raise RuntimeError(f"process exited early: {dead}; see {self.logdir}")

    @staticmethod
    def _signal(pgid: int, sig: int) -> bool:
        try:
            os.killpg(pgid, sig)
            return True
        except (ProcessLookupError, PermissionError):
            return False

    def stop(self, grace_s: float = 30.0) -> None:
        """Stop every process group, children included (vLLM's EngineCore holds the GPU),
        and return only once all groups are gone."""
        groups = [p.pid for _, p in reversed(self.procs)]  # start_new_session: pgid == pid
        for g in groups:
            self._signal(g, signal.SIGTERM)
        deadline = time.monotonic() + grace_s
        while time.monotonic() < deadline and any(self._signal(g, 0) for g in groups):
            for _, p in self.procs:
                p.poll()
            time.sleep(0.2)
        for g in groups:
            self._signal(g, signal.SIGKILL)
        for _, p in self.procs:
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
        while any(self._signal(g, 0) for g in groups) and time.monotonic() < deadline + 30:
            time.sleep(0.2)


def http(url: str, body: dict | None = None, timeout: float = 2.0) -> tuple[int, bytes]:
    """GET, or POST `body` as JSON. Returns (status, body); HTTP errors are not raised."""
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, headers={"content-type": "application/json"} if data else {})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def wait_until(pred, timeout: float, what: str, procs: Procs, interval: float = 0.5) -> None:
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        procs.check()
        try:
            if pred():
                return
        except Exception as e:  # not up yet
            last = e
        time.sleep(interval)
    raise TimeoutError(f"timed out waiting for {what} ({last!r})")


# ---------------------------------------------------------------- pods


def sim_cmd(sc: Scenario, pod: Pod, seed: int) -> list[str]:
    p = {**SIM_DEFAULTS, **sc.sim}
    cmd = [
        tool("llm-d-inference-sim", "LORABENCH_SIM"),
        "--port", str(pod.port),
        "--model", BASE_MODEL,
        "--mode", "random",
        "--seed", str(seed),
        "--max-loras", str(sc.max_loras),
        "--max-cpu-loras", str(sc.max_cpu_loras),
        "--max-num-seqs", str(p["max_num_seqs"]),
        "--max-model-len", "8192",
        "--time-to-first-token", p["ttft"],
        "--inter-token-latency", p["itl"],
        "--time-factor-under-load", str(p["time_factor"]),
        "--lora-modules",
    ]
    cmd += [json.dumps({"name": a, "path": f"/adapters/{a}", "base_model_name": BASE_MODEL}) for a in sc.adapters()]
    return cmd


def vllm_cmd(sc: Scenario, pod: Pod, model: str, adapters_dir: Path) -> list[str]:
    cmd = [
        os.environ.get("LORABENCH_VLLM", "vllm"), "serve", model,
        "--host", pod.ip, "--port", str(pod.port),
        "--served-model-name", BASE_MODEL,
        "--enable-lora",
        "--max-loras", str(sc.max_loras),
        "--max-cpu-loras", str(sc.max_cpu_loras),
        "--max-lora-rank", str(sc.max_lora_rank),
        "--max-num-seqs", str(sc.max_num_seqs),
        "--max-num-batched-tokens", "8192",
        "--max-model-len", "4096",
        "--lora-modules",
    ]
    return cmd + [f"{a}={adapters_dir / a}" for a in sc.adapters()]


VLLM_LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {"iso": {"format": "%(asctime)s.%(msecs)03d %(levelname)s %(name)s %(message)s",
                           "datefmt": "%Y-%m-%dT%H:%M:%S"}},
    "handlers": {"out": {"class": "logging.StreamHandler", "formatter": "iso", "stream": "ext://sys.stdout"}},
    "loggers": {
        "vllm": {"handlers": ["out"], "level": "INFO", "propagate": False},
        "vllm.lora.model_manager": {"handlers": ["out"], "level": "DEBUG", "propagate": False},
        "vllm.lora.worker_manager": {"handlers": ["out"], "level": "DEBUG", "propagate": False},
        # Per-request INFO lines would add arm-independent but avoidable overhead.
        "vllm.entrypoints.logger": {"handlers": ["out"], "level": "WARNING", "propagate": False},
    },
}


def gpu_ids(n: int, spec: str | None = None) -> list[int]:
    """GPU index per pod. LORABENCH_GPUS="4,5,6,7" picks GPUs on a shared node; default 0..n-1."""
    spec = os.environ.get("LORABENCH_GPUS", "") if spec is None else spec
    if not spec.strip():
        return list(range(n))
    ids = [int(x) for x in spec.split(",") if x.strip()]
    if len(ids) < n or len(set(ids)) != len(ids):
        raise ValueError(f"LORABENCH_GPUS={spec!r}: need {n} distinct GPU indices")
    return ids[:n]


def make_pods(sc: Scenario, backend: str, ports: Ports) -> list[Pod]:
    """EPP matches pods by IP only, so on Linux each pod gets its own loopback IP
    (127.0.0.2, ...). macOS has only 127.0.0.1 without root; pods share it there."""
    gpus = gpu_ids(sc.pods) if backend == "vllm" else [None] * sc.pods
    linux = sys.platform.startswith("linux")
    return [Pod(name=f"pod-{i}", ip=f"127.0.0.{i + 2}" if linux else "127.0.0.1", port=ports.pod_base + i,
                gpu=gpus[i]) for i in range(sc.pods)]


# ---------------------------------------------------------------- envoy


def envoy_config(ports: Ports, access_log: Path) -> dict:
    """Mirrors the llm-d-router-standalone chart (Envoy v1.33.2) with explicit timeouts and limits."""
    log_format = {
        "rid": "%REQ(X-REQUEST-ID)%",
        "start": "%START_TIME(%s.%6f)%",
        "upstream": "%UPSTREAM_HOST%",
        "flags": "%RESPONSE_FLAGS%",
        "code": "%RESPONSE_CODE%",
        "attempts": "%UPSTREAM_REQUEST_ATTEMPT_COUNT%",
        "us_tx_begin_us": "%COMMON_DURATION(DS_RX_BEG:US_TX_BEG:us)%",
        "duration_ms": "%DURATION%",
    }
    return {
        "admin": {"address": {"socket_address": {"address": "127.0.0.1", "port_value": ports.envoy_admin}}},
        "static_resources": {
            "listeners": [{
                "name": "inference",
                "address": {"socket_address": {"address": "127.0.0.1", "port_value": ports.envoy}},
                "filter_chains": [{"filters": [{
                    "name": "envoy.filters.network.http_connection_manager",
                    "typed_config": {
                        "@type": "type.googleapis.com/envoy.extensions.filters.network.http_connection_manager.v3.HttpConnectionManager",
                        "stat_prefix": "inference",
                        "stream_idle_timeout": "300s",
                        "request_timeout": "0s",
                        "http2_protocol_options": {"max_concurrent_streams": 100000},
                        "access_log": [{
                            "name": "envoy.access_loggers.file",
                            "typed_config": {
                                "@type": "type.googleapis.com/envoy.extensions.access_loggers.file.v3.FileAccessLog",
                                "path": str(access_log),
                                "log_format": {"json_format": log_format},
                            },
                        }],
                        "route_config": {"name": "inference", "virtual_hosts": [{
                            "name": "inference", "domains": ["*"],
                            "response_headers_to_add": [{"header": {"key": SERVED_BY_HEADER, "value": "%UPSTREAM_HOST%"},
                                                         "append_action": "OVERWRITE_IF_EXISTS_OR_ADD"}],
                            "routes": [{"match": {"prefix": "/"}, "route": {
                                "cluster": "original_destination_cluster",
                                "timeout": "0s", "idle_timeout": "86400s",
                            }}],
                        }]},
                        "http_filters": [
                            {"name": "envoy.filters.http.ext_proc", "typed_config": {
                                "@type": "type.googleapis.com/envoy.extensions.filters.http.ext_proc.v3.ExternalProcessor",
                                "failure_mode_allow": False,
                                "grpc_service": {"envoy_grpc": {"cluster_name": "epp", "authority": f"localhost:{ports.epp_grpc}"},
                                                 "timeout": "10s"},
                                "processing_mode": {
                                    "request_header_mode": "SEND", "response_header_mode": "SEND",
                                    "request_body_mode": "FULL_DUPLEX_STREAMED", "response_body_mode": "FULL_DUPLEX_STREAMED",
                                    "request_trailer_mode": "SEND", "response_trailer_mode": "SEND",
                                },
                                "message_timeout": "1000s",
                            }},
                            {"name": "envoy.filters.http.router", "typed_config": {
                                "@type": "type.googleapis.com/envoy.extensions.filters.http.router.v3.Router",
                                "suppress_envoy_headers": True,
                            }},
                        ],
                    },
                }]}],
            }],
            "clusters": [
                {
                    "name": "epp", "type": "STATIC", "connect_timeout": "10s", "lb_policy": "LEAST_REQUEST",
                    "typed_extension_protocol_options": {"envoy.extensions.upstreams.http.v3.HttpProtocolOptions": {
                        "@type": "type.googleapis.com/envoy.extensions.upstreams.http.v3.HttpProtocolOptions",
                        "explicit_http_config": {"http2_protocol_options": {"max_concurrent_streams": 100000}},
                    }},
                    "circuit_breakers": {"thresholds": [{"max_connections": 100000, "max_pending_requests": 100000,
                                                         "max_requests": 100000, "max_retries": 0}]},
                    "load_assignment": {"cluster_name": "epp", "endpoints": [{"lb_endpoints": [{"endpoint": {
                        "address": {"socket_address": {"address": "127.0.0.1", "port_value": ports.epp_grpc}}}}]}]},
                },
                {
                    "name": "original_destination_cluster", "type": "ORIGINAL_DST", "connect_timeout": "10s",
                    "lb_policy": "CLUSTER_PROVIDED",
                    # vLLM (uvicorn) closes idle keep-alive connections after 5 s. Envoy must drop them
                    # first, or a request can race the close and fail with 503 UC.
                    "typed_extension_protocol_options": {"envoy.extensions.upstreams.http.v3.HttpProtocolOptions": {
                        "@type": "type.googleapis.com/envoy.extensions.upstreams.http.v3.HttpProtocolOptions",
                        "common_http_protocol_options": {"idle_timeout": "3s"},
                        "explicit_http_config": {"http_protocol_options": {}},
                    }},
                    "circuit_breakers": {"thresholds": [{"max_connections": 100000, "max_pending_requests": 100000,
                                                         "max_requests": 100000, "max_retries": 0}]},
                    "original_dst_lb_config": {"use_http_header": True,
                                               "http_header_name": "x-gateway-destination-endpoint"},
                },
            ],
        },
    }


# ---------------------------------------------------------------- run


@dataclasses.dataclass
class Topology:
    sc: Scenario
    arm: str
    backend: str
    run_dir: Path
    seed: int
    model: str = ""
    adapters_dir: Path | None = None
    ports: Ports = dataclasses.field(default_factory=Ports)

    def __post_init__(self):
        self.pods = make_pods(self.sc, self.backend, self.ports)
        self.procs = Procs(self.run_dir / "logs")
        (self.run_dir / "logs").mkdir(parents=True, exist_ok=True)
        (self.run_dir / "config").mkdir(parents=True, exist_ok=True)

    @property
    def gateway(self) -> str:
        return f"http://127.0.0.1:{self.ports.envoy}/v1/chat/completions"

    @property
    def epp_metrics(self) -> str:
        return f"http://127.0.0.1:{self.ports.epp_metrics}/metrics"

    def addr_to_pod(self) -> dict[str, str]:
        return {p.addr: p.name for p in self.pods}

    def start(self) -> None:
        self.start_pods()
        self.start_router()

    def start_pods(self) -> None:
        cfg = self.run_dir / "config"
        if self.backend == "vllm":
            self._preread_adapters()
            self.procs.start("nvidia-smi", [
                "nvidia-smi", "--query-gpu=timestamp,index,clocks.sm,clocks.mem,power.draw,temperature.gpu,"
                "utilization.gpu,memory.used,clocks_throttle_reasons.active", "--format=csv", "-lms", "1000"])
        for pod in self.pods:
            if self.backend == "sim":
                self.procs.start(pod.name, sim_cmd(self.sc, pod, self.seed))
            else:
                logcfg = cfg / "vllm-logging.json"
                logcfg.write_text(json.dumps(VLLM_LOGGING))
                env = {
                    "CUDA_VISIBLE_DEVICES": str(pod.gpu),
                    "VLLM_LOGGING_CONFIG_PATH": str(logcfg),
                    "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "false",
                }
                self.procs.start(pod.name, vllm_cmd(self.sc, pod, self.model, self.adapters_dir), env)

        ready_s = 900 if self.backend == "vllm" else 30
        for pod in self.pods:
            wait_until(lambda: http(pod.url + "/health")[0] == 200, ready_s, f"{pod.name} /health", self.procs)
            wait_until(lambda: self._models_listed(pod), 60, f"{pod.name} /v1/models", self.procs)
        for pod in self.pods:
            self._prime(pod)

    def start_router(self) -> None:
        cfg = self.run_dir / "config"
        endpoints = cfg / "endpoints.yaml"
        endpoints.write_text(arms.endpoints_file([(p.name, p.ip, p.port) for p in self.pods]))
        epp_cfg = cfg / "epp.yaml"
        epp_cfg.write_text(arms.epp_config(self.arm, self.sc, BASE_MODEL, str(endpoints)))
        envoy_cfg = cfg / "envoy.json"
        envoy_cfg.write_text(json.dumps(envoy_config(self.ports, self.run_dir / "access.jsonl"), indent=1))

        epp_args = [
            tool("epp", "LORABENCH_EPP"),
            "--config-file", str(epp_cfg),
            "--pool-name", "llmd-lora-routing-bench", "--pool-namespace", "local",
            "--secure-serving=false",
            "--tracing=false",  # on by default; with no collector it retries exports inside the measured router
            "--grpc-port", str(self.ports.epp_grpc),
            "--grpc-health-port", str(self.ports.epp_health),
            "--metrics-port", str(self.ports.epp_metrics),
            "--refresh-metrics-interval", "50ms",
            "-v", "2",
        ]
        if self.arm in arms.EXPERIMENTAL:
            epp_args.append("--allow-experimental-plugins")
        raise_nofile()
        self.procs.start("epp", epp_args)
        wait_until(self._epp_sees_all_pods, 60, "EPP to discover and scrape every pod", self.procs)
        self.procs.start("envoy", [tool("envoy", "LORABENCH_ENVOY"), "-c", str(envoy_cfg),
                                   "--log-level", "warn", "--concurrency", "4",
                                   "--disable-hot-restart"])  # else two runs on one host collide on its socket
        wait_until(lambda: http(f"http://127.0.0.1:{self.ports.envoy_admin}/ready")[0] == 200,
                   30, "Envoy ready", self.procs)
        self.envoy_stats("before")

    def _epp_sees_all_pods(self) -> bool:
        """EPP reports every pod ready and none stale (DESIGN_LLMD_709.md §2, readiness 4)."""
        code, body = http(self.epp_metrics)
        if code != 200:
            return False
        ready = stale = None
        for name, _, value in parse(body.decode(errors="replace")):
            if name.endswith("ready_endpoints"):
                ready = (ready or 0) + value
            elif name.endswith("flow_control_stale_endpoints"):
                stale = (stale or 0) + value
        return ready == len(self.pods) and not stale

    def _preread_adapters(self) -> None:
        """Read every adapter file once so loads come from page cache in every arm (DESIGN_LLMD_709.md §2)."""
        n = 0
        for name in self.sc.adapters():
            for f in sorted((self.adapters_dir / name).glob("*")):
                with f.open("rb") as fh:
                    while chunk := fh.read(1 << 24):
                        n += len(chunk)
        (self.run_dir / "config" / "preread_bytes").write_text(str(n))

    def _models_listed(self, pod: Pod) -> bool:
        code, body = http(pod.url + "/v1/models")
        if code != 200:
            return False
        ids = {m["id"] for m in json.loads(body).get("data", [])}
        return set(self.sc.adapters()) <= ids

    def _prime(self, pod: Pod) -> None:
        """One base-model request per pod, straight to the pod, so vllm:lora_requests_info exists (#3361)."""
        body = {"model": BASE_MODEL, "messages": [{"role": "user", "content": "prime"}],
                "max_tokens": 4, "stream": False}
        code, _ = http(pod.url + "/v1/chat/completions", body, timeout=60)
        if code != 200:
            raise RuntimeError(f"priming {pod.name} failed with HTTP {code}")

    def envoy_stats(self, tag: str) -> None:
        code, body = http(f"http://127.0.0.1:{self.ports.envoy_admin}/stats?format=json", timeout=5)
        if code == 200:
            (self.run_dir / f"envoy-stats-{tag}.json").write_bytes(body)

    def stop(self) -> None:
        try:
            self.envoy_stats("after")
        except OSError:  # Envoy already gone; validity flags the missing stats
            pass
        self.procs.stop()
