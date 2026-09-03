# pymoo: Non-Dominated Sorting

## Overview

This task optimizes non-dominated sorting (`pymoo/functions/standard/non_dominated_sorting.py`)
in [`anyoptimization/pymoo`](https://github.com/anyoptimization/pymoo), a multi-objective
optimization framework, with emphasis on the bi-objective case while preserving exact
Pareto fronts. See `task.yaml` for the exact weighting between correctness and
performance.

## Source

- **Repository**: https://github.com/anyoptimization/pymoo
- **Vendored path**: `pymoo/`
- **License**: Apache License 2.0 (see `pymoo/LICENSE`, copied verbatim from upstream)

## Files

- `pymoo/`: vendored library source containing the `EVOLVE-BLOCK` target
- `task.yaml`: task metadata, evaluation weights, and evolution configuration
- `evaluator.py`: correctness (front equivalence) + performance scoring for candidate programs
- `config_handwritten.yaml`, `config_llm_generated.yaml`, `config_llm_judge.yaml`: evaluator configurations
- `benchmark.py`: standalone timing harness for the optimization target

## Running

```bash
eb run --task pymoo --evaluator handwritten --iteration 40
```

## License

This task vendors source code from `pymoo`, licensed under the Apache License 2.0.
The upstream license text is included at `pymoo/LICENSE`. EvolveBench's own harness
and task code are licensed separately under the same Apache 2.0 terms — see the
repository root [LICENSE](../../LICENSE).
