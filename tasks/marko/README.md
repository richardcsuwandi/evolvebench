# Marko: Nested Parser Loops

## Overview

This task optimizes nested loops in the `parse_source` method of
[`frostming/marko`](https://github.com/frostming/marko), a CommonMark-compliant
Markdown parser. The optimization target is `performance-nested-loops-parse-source`
inside `marko/`. See `task.yaml` for the exact weighting between correctness and
performance.

## Source

- **Repository**: https://github.com/frostming/marko
- **Vendored path**: `marko/`
- **License**: MIT (see `marko/LICENSE`, copied verbatim from upstream)

## Files

- `marko/`: vendored library source containing the `EVOLVE-BLOCK` target
- `task.yaml`: task metadata, evaluation weights, and evolution configuration
- `evaluator.py`: correctness + performance scoring for candidate programs
- `generated_evaluator.py`: LLM-synthesized evaluator variant
- `config_handwritten.yaml`, `config_llm_generated.yaml`, `config_llm_judge.yaml`: evaluator configurations
- `benchmark.py`: standalone timing harness for the optimization target

## Running

```bash
eb run --task marko --evaluator handwritten --iteration 20
```

## License

This task vendors source code from `marko`, licensed under the MIT License. The
upstream license text is included at `marko/LICENSE`. EvolveBench's own harness and
task code are licensed separately. See the repository root [LICENSE](../../LICENSE).
