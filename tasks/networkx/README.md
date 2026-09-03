# NetworkX: Graph Algorithm Optimization

## Overview

This task optimizes `_single_source_shortest_path_basic`, a BFS helper used by
betweenness centrality in [`networkx/networkx`](https://github.com/networkx/networkx).
The optimization target is `graph-algorithm-optimization` inside `networkx/` — see
`task.yaml` for the exact weighting between correctness and performance.

Only the library source is vendored; the upstream `networkx/tests/` tree (and the
per-module `tests/` directories nested throughout the package) was removed since the
evaluator and benchmark do not depend on it — see `evaluator.py` and `benchmark.py`.

## Source

- **Repository**: https://github.com/networkx/networkx
- **Vendored path**: `networkx/`
- **License**: BSD 3-Clause (see `networkx/LICENSE`, copied verbatim from upstream)

## Files

- `networkx/`: vendored library source containing the `EVOLVE-BLOCK` target
- `task.yaml`: task metadata, evaluation weights, and evolution configuration
- `evaluator.py`: correctness + performance scoring for candidate programs
- `config_handwritten.yaml`, `config_llm_generated.yaml`, `config_llm_judge.yaml`: evaluator configurations
- `benchmark.py`: standalone timing harness comparing initial vs. optimized implementations

## Running

```bash
eb run --task networkx --evaluator handwritten --iteration 25
```

## License

This task vendors source code from `networkx`, licensed under the BSD 3-Clause
License. The upstream license text is included at `networkx/LICENSE`. EvolveBench's
own harness and task code are licensed separately — see the repository root [LICENSE](../../LICENSE).
