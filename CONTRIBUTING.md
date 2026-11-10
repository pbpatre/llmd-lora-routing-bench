# Contributing

Issues and pull requests are welcome. For anything beyond a small fix, open an issue first so we can agree on the approach.

## Before you open a pull request

```bash
pip install -e '.[dev]'
make test          # unit tests, no GPU or binaries needed
pyflakes lorabench tests tools
```

If you touch the topology, load generator or analysis, also run the simulator smoke test (`make smoke-sim`, needs the `epp`, `envoy` and `llm-d-inference-sim` binaries; see the README).

## Ground rules

- **Results are pre-registered.** Scenario, arm and gate definitions live in `scenarios/` and [DESIGN_LLMD_709.md](DESIGN_LLMD_709.md). A change to them after results exist bumps the DESIGN_LLMD_709.md version and says why.
- **Only valid runs count.** A result in `results/` comes from runs that pass every check in `summary.json` `validity`, with the setup (GPU, vLLM and router versions) stated.
- **Simulator latencies are never results.** The simulator checks the harness, not the arms.
- **Keep it simple.** Small, readable functions; comments say why, not what.

## Commits

Sign off every commit (`git commit -s`) to certify the [Developer Certificate of Origin](https://developercertificate.org/). Contributions are licensed under Apache 2.0.
