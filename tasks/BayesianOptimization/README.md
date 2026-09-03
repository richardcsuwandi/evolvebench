# BayesianOptimization: Acquisition-Function Sampling

## Overview

This task optimizes the random-sampling step of the acquisition function in
[`bayesian-optimization/BayesianOptimization`](https://github.com/bayesian-optimization/BayesianOptimization),
a widely used Python library for Bayesian optimization. The optimization target is
`optimization-acquisition-sampling` inside `bayes_opt/`. See `task.yaml` for the exact
weighting between correctness and performance.

## Source

- **Repository**: https://github.com/bayesian-optimization/BayesianOptimization
- **Vendored path**: `bayes_opt/`
- **License**: MIT (see `bayes_opt/LICENSE`, copied verbatim from upstream)

## Files

- `bayes_opt/`: vendored library source containing the `EVOLVE-BLOCK` target
- `task.yaml`: task metadata, evaluation weights, and evolution configuration
- `evaluator.py`: correctness + performance scoring for candidate programs
- `generated_evaluator.py`: LLM-synthesized evaluator variant
- `config_handwritten.yaml`, `config_llm_generated.yaml`, `config_llm_judge.yaml`: evaluator configurations
- `benchmark.py`: standalone timing harness for the optimization target

## Running

```bash
eb run --task BayesianOptimization --evaluator handwritten --iteration 30
```

## License

This task vendors source code from `BayesianOptimization`, licensed under the MIT
License. The upstream license text is included at `bayes_opt/LICENSE`. EvolveBench's
own harness and task code are licensed separately. See the repository root [LICENSE](../../LICENSE).
