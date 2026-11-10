#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# One-time setup on the GPU node (Linux, x86_64 or arm64). Idempotent.
#   WORK=/data/llmd-lora-routing-bench ./tools/setup-node.sh
# Needs: python3.10+ with venv (Ubuntu: apt install python3.10-venv), curl, network to PyPI, Hugging Face and GitHub (plus
# docker.io if Docker is present). Docker is optional: without it the script
# downloads the Go toolchain from go.dev and Envoy from GitHub releases.
# Run nvidia-smi -pm 1 as root first. Set LORABENCH_GPUS=4,5,6,7 later if
# GPUs 0-3 are busy on a shared node.
set -euo pipefail
cd "$(dirname "$0")/.."
WORK=${WORK:-$HOME/llmd-lora-routing-bench-work}
VLLM_VERSION=${VLLM_VERSION:-0.31.0}
MODEL=${MODEL:-unsloth/Llama-3.1-8B-Instruct}
MODEL_REV=${MODEL_REV:-4699cc75}
ROUTER_COMMIT=${ROUTER_COMMIT:-1f7f0d902e22ce56cd0363fbfbc8fab91653a99e}
ENVOY_IMAGE=${ENVOY_IMAGE:-envoyproxy/envoy:distroless-v1.33.2}
SIM_VERSION=${SIM_VERSION:-v1.0.0}
GO_VERSION=${GO_VERSION:-1.27.3}
mkdir -p "$WORK"/{bin,models,adapters,cache/triton,cache/vllm,src} .cache/bin
# Some GPU containers keep nvidia-smi outside the default PATH.
NVIDIA_BIN=/usr/local/nvidia/bin
[ -x "$NVIDIA_BIN/nvidia-smi" ] && export PATH="$NVIDIA_BIN:$PATH"
case "$(uname -m)" in
  x86_64) GOARCH=amd64; ENVOY_ARCH=x86_64 ;;
  aarch64|arm64) GOARCH=arm64; ENVOY_ARCH=aarch_64 ;;
  *) echo "unsupported arch $(uname -m)"; exit 1 ;;
esac
HAVE_DOCKER=0
if command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then HAVE_DOCKER=1; fi
echo "== docker usable: $HAVE_DOCKER"

echo "== python env (vLLM $VLLM_VERSION)"
if [ ! -x "$WORK/venv/bin/python" ]; then python3 -m venv "$WORK/venv"; fi
"$WORK/venv/bin/pip" install -q "vllm==$VLLM_VERSION" aiohttp numpy pyyaml pytest safetensors huggingface_hub

echo "== model $MODEL@$MODEL_REV"
"$WORK/venv/bin/python" - <<PY
from huggingface_hub import snapshot_download
p = snapshot_download("$MODEL", revision="$MODEL_REV", local_dir="$WORK/models/llama-3.1-8b",
                      allow_patterns=["*.json", "*.safetensors", "tokenizer*"])
print(p)
PY

echo "== adapters"
"$WORK/venv/bin/python" tools/make_adapters.py --base-config "$WORK/models/llama-3.1-8b/config.json" \
  --scenario scenarios/s0-calibration.yaml --scenario scenarios/s1-fits.yaml \
  --scenario scenarios/s2-oversubscribed.yaml --scenario scenarios/s3-two-hot.yaml \
  --scenario scenarios/s4-one-giant.yaml --scenario scenarios/s5-zipf.yaml --out "$WORK/adapters"

ENVOY_VERSION=${ENVOY_IMAGE##*-v}
if [ ! -x "$WORK/bin/envoy" ]; then
  if [ "$HAVE_DOCKER" = 1 ]; then
    echo "== envoy from $ENVOY_IMAGE"
    cid=$(docker create "$ENVOY_IMAGE")
    docker cp "$cid:/usr/local/bin/envoy" "$WORK/bin/envoy"
    docker rm "$cid" >/dev/null
  else
    # Release asset names: envoy-<ver>-linux-x86_64 / envoy-<ver>-linux-aarch_64.
    url="https://github.com/envoyproxy/envoy/releases/download/v$ENVOY_VERSION/envoy-$ENVOY_VERSION-linux-$ENVOY_ARCH"
    echo "== envoy from $url"
    curl -sSfL -o "$WORK/bin/envoy" "$url"
    chmod +x "$WORK/bin/envoy"
  fi
fi
"$WORK/bin/envoy" --version

echo "== epp at $ROUTER_COMMIT and llm-d-inference-sim $SIM_VERSION"
if [ ! -d "$WORK/src/llm-d-router" ]; then
  mkdir -p "$WORK/src/llm-d-router"
  curl -sSfL "https://codeload.github.com/llm-d/llm-d-router/tar.gz/$ROUTER_COMMIT" \
    | tar xz -C "$WORK/src/llm-d-router" --strip-components=1
fi
BUILD_EPP="go build -o /out/epp ./cmd/epp"
BUILD_SIM="go install github.com/llm-d/llm-d-inference-sim/cmd/llm-d-inference-sim@$SIM_VERSION"
build() {  # build <command>, in golang:$GO_VERSION if Docker works, else with a downloaded toolchain
  if [ "$HAVE_DOCKER" = 1 ]; then
    docker run --rm -u "$(id -u):$(id -g)" -e HOME=/tmp -e CGO_ENABLED=0 -e GOBIN=/out \
      -v "$WORK/src/llm-d-router:/src" -v "$WORK/bin:/out" -w /src "golang:$GO_VERSION" sh -c "$1"
  else
    if [ ! -x "$WORK/go/bin/go" ]; then
      curl -sSfL "https://go.dev/dl/go$GO_VERSION.linux-$GOARCH.tar.gz" | tar xz -C "$WORK"
    fi
    (cd "$WORK/src/llm-d-router" && PATH="$WORK/go/bin:$PATH" CGO_ENABLED=0 GOBIN="$WORK/bin" \
      GOPATH="$WORK/gopath" GOCACHE="$WORK/cache/go" sh -c "${1//\/out/$WORK/bin}")
  fi
}
[ -x "$WORK/bin/epp" ] || build "$BUILD_EPP"
[ -x "$WORK/bin/llm-d-inference-sim" ] || build "$BUILD_SIM"
ln -sf "$WORK/bin/epp" .cache/bin/epp
ln -sf "$WORK/bin/envoy" .cache/bin/envoy
ln -sf "$WORK/bin/llm-d-inference-sim" .cache/bin/llm-d-inference-sim

cat > "$WORK/env.sh" <<ENV
export LORABENCH_VLLM="$WORK/venv/bin/vllm"
export LORABENCH_MODEL="$WORK/models/llama-3.1-8b"
export LORABENCH_ADAPTERS="$WORK/adapters"
export LORABENCH_ROUTER_COMMIT="$ROUTER_COMMIT"
export TRITON_CACHE_DIR="$WORK/cache/triton"
export VLLM_CACHE_ROOT="$WORK/cache/vllm"
unset VLLM_USE_RUST_FRONTEND
export PATH="$WORK/venv/bin:$([ -d $NVIDIA_BIN ] && echo $NVIDIA_BIN:)\$PATH"
ENV
{
  echo "vllm $VLLM_VERSION"; echo "model $MODEL@$MODEL_REV"; echo "router $ROUTER_COMMIT"
  echo "envoy $("$WORK/bin/envoy" --version 2>&1 | grep -o "[0-9]*\.[0-9]*\.[0-9]*" | head -1)"; echo "sim $SIM_VERSION"; nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
  sha256sum "$WORK/bin/epp" "$WORK/bin/envoy"
} > "$WORK/provenance.txt"
echo "done. source $WORK/env.sh"
echo "next: make test gpu-smoke quick-s1 (README, Quick start)"
