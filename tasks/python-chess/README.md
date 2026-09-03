# python-chess: Move Generation Optimization

## Overview

This task optimizes chess move-generation/evaluation code in
[`niklasf/python-chess`](https://github.com/niklasf/python-chess). The optimization
target is `chess-engine-optimization` inside `chess/` — see `task.yaml` for the exact
weighting between correctness and performance.

## Source

- **Repository**: https://github.com/niklasf/python-chess
- **Vendored path**: `chess/`
- **License**: GPL-3.0-or-later (see `chess/LICENSE`, copied verbatim from upstream)

> **Note on licensing**: unlike the other vendored tasks in this benchmark,
> `python-chess` is licensed under the **GNU GPL v3**, a copyleft license. The
> `chess/` directory retains its original GPL-3.0 terms independently of the
> Apache-2.0 license covering the rest of this repository (see root
> [LICENSE](../../LICENSE)). If you redistribute a derivative of this task's vendored
> source, GPL-3.0's copyleft obligations apply to that derivative.

## Files

- `chess/`: vendored library source containing the `EVOLVE-BLOCK` target
- `task.yaml`: task metadata, evaluation weights, and evolution configuration
- `evaluator.py`: correctness + performance scoring for candidate programs
- `config_handwritten.yaml`, `config_llm_generated.yaml`, `config_llm_judge.yaml`: evaluator configurations
- `benchmark.py`: standalone timing harness for the optimization target

## Running

```bash
eb run --task python-chess --evaluator handwritten --iteration 30
```

## License

This task vendors source code from `python-chess`, licensed under GPL-3.0-or-later.
The upstream license text is included at `chess/LICENSE`. EvolveBench's own harness
and task code are licensed separately — see the repository root [LICENSE](../../LICENSE).
