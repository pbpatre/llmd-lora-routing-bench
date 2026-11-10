# llmd-lora-routing-bench. On the GPU node, `source $WORK/env.sh` first (written by tools/setup-node.sh).
# GPU results go to $(RUNS). Smoke and quick runs go to throwaway trees, cleared on each run,
# so reports never mix them with results (a report refuses duplicate runs).
PY ?= python3
RUNS ?= runs
SEEDS ?= 1 2 3
CALIBRATION ?=
ENGINE = --backend vllm --model $(LORABENCH_MODEL) --adapters-dir $(LORABENCH_ADAPTERS)
VLLM = $(ENGINE) $(if $(CALIBRATION),--calibration $(CALIBRATION))
S1_ARMS = stock stock-noprefix affinity-w2 residency-w4
ARMS = stock affinity-w2 residency-w4

.PHONY: test smoke-sim gpu-smoke quick-s1 calibrate s1-set first-set second-set report gate

test:
	$(PY) -m pytest -q -p no:cacheprovider tests

# Simulator: S1, S2, S3, S4 shortened, for the regime checks in DESIGN_LLMD_709.md §10.
smoke-sim:
	rm -rf runs-smoke
	$(PY) -m lorabench matrix scenarios/s1-fits.yaml scenarios/s2-oversubscribed.yaml \
	  scenarios/s3-two-hot.yaml scenarios/s4-one-giant.yaml \
	  --arms $(ARMS) --seeds 1 --duration-scale 0.34 --runs runs-smoke --keep-going
	$(PY) -m lorabench report runs-smoke

# Simulator, S1 only: a few minutes, no GPU time. The first thing to run on a new node.
gpu-smoke:
	rm -rf runs-smoke
	$(PY) -m lorabench matrix scenarios/s1-fits.yaml --arms stock residency-w4 --seeds 1 \
	  --duration-scale 0.34 --runs runs-smoke
	$(PY) -m lorabench report runs-smoke

# First vLLM number: S1, stock vs residency-w4, seed 1. S1 has a fixed rate, so
# CALIBRATION is optional (without it, the base-model TTFT check is skipped).
quick-s1:
	rm -rf runs-quick
	$(PY) -m lorabench matrix scenarios/s1-fits.yaml --arms stock residency-w4 --seeds 1 $(VLLM) --runs runs-quick
	$(PY) -m lorabench report runs-quick

# S0: one-pod capacity per output length and adapter load costs. Writes calibration.json.
calibrate:
	$(PY) -m lorabench calibrate $(ENGINE) --runs $(RUNS)

# S1 with the A/A replicate (DESIGN_LLMD_709.md §8).
s1-set:
	$(PY) -m lorabench matrix scenarios/s1-fits.yaml --arms $(S1_ARMS) --seeds $(SEEDS) --aa $(VLLM) \
	  --runs $(RUNS) --keep-going
	$(PY) -m lorabench report $(RUNS) --backend vllm

# The rest of the first credible set (DESIGN_LLMD_709.md §9): S2, S3, S5. Run s1-set too.
first-set:
	@test -n "$(CALIBRATION)" || { echo "set CALIBRATION=path/to/calibration.json"; exit 1; }
	$(PY) -m lorabench matrix scenarios/s2-oversubscribed.yaml scenarios/s3-two-hot.yaml --arms $(ARMS) \
	  --seeds $(SEEDS) --aa $(VLLM) --runs $(RUNS) --keep-going
	$(PY) -m lorabench matrix scenarios/s5-zipf.yaml --arms affinity-w2 residency-w4 \
	  --seeds $(SEEDS) $(VLLM) --runs $(RUNS) --keep-going
	$(PY) -m lorabench report $(RUNS) --backend vllm

# Before any filter code (DESIGN_LLMD_709.md §9): S4, the rr2 pairing on S3 and S4, and the oracle
# headroom check on S2, S3, S4. Needs CALIBRATION (S3 and S4 scale with capacity).
second-set:
	@test -n "$(CALIBRATION)" || { echo "set CALIBRATION=path/to/calibration.json"; exit 1; }
	$(PY) -m lorabench matrix scenarios/s4-one-giant.yaml --arms $(ARMS) residency-w4-rr2 oracle \
	  --seeds $(SEEDS) --aa $(VLLM) --runs $(RUNS) --keep-going
	$(PY) -m lorabench matrix scenarios/s3-two-hot.yaml --arms residency-w4-rr2 oracle \
	  --seeds $(SEEDS) $(VLLM) --runs $(RUNS) --keep-going
	$(PY) -m lorabench matrix scenarios/s2-oversubscribed.yaml --arms oracle \
	  --seeds $(SEEDS) $(VLLM) --runs $(RUNS) --keep-going
	$(PY) -m lorabench report $(RUNS) --backend vllm

report:
	$(PY) -m lorabench report $(RUNS) --backend vllm

gate:
	@test -n "$(ARM)" || { echo "set ARM=<candidate arm>"; exit 1; }
	$(PY) -m lorabench report $(RUNS) --backend vllm --gate $(ARM)
