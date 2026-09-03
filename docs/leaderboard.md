# Leaderboard

This page tracks how AI-driven research and optimization systems perform on
EvolveBench, starting with [OpenEvolve](https://github.com/algorithmicsuperintelligence/openevolve)
below. If you've run [ShinkaEvolve](https://github.com/SakanaAI/ShinkaEvolve),
[AlphaEvolve](https://deepmind.google/discover/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/)-style
agents, or anything else against EvolveBench, we'd love to add your numbers here.
See [Submitting a result](#submitting-a-result).

## Overall

Aggregate score is the harmonic mean of per-task speedups (evolved program vs.
baseline) across all tasks, following the aggregation convention used in
[AlgoTune](https://algotune.io/). Per-task and per-evaluator scores are below.

| Rank | System | Evaluator | Aggregate score |
| --- | --- | --- | --- |
| 1 | [OpenEvolve](https://github.com/algorithmicsuperintelligence/openevolve) | LLM-generated | **1.35×** |
| 2 | [OpenEvolve](https://github.com/algorithmicsuperintelligence/openevolve) | Handwritten | 1.12× |

Submit your own system's numbers, see [Submitting a result](#submitting-a-result).

## OpenEvolve

Per-task speedup (evolved program vs. baseline) under each evaluator design. The
aggregate score is the harmonic mean of per-task speedups, following the
aggregation convention used in [AlgoTune](https://algotune.io/). Full methodology
and analysis in the [project write-up](https://richardcsuwandi.github.io/projects/evolvebench/).

| Task | Handwritten evaluator | LLM-generated evaluator |
| --- | --- | --- |
| BayesianOptimization | 1.54× | 1.82× |
| difflib | 0.99× | 3.54× |
| jsonschema | 2.20× | 2.24× |
| lmcache | 0.65× | 1.02× |
| marko | 1.00× | 1.01× |
| networkx | 1.07× | 1.03× |
| pandas_rolling_rank | 1.00× | 2.31× |
| pymoo | 1.00× | 1.28× |
| python-chess | 1.01× | 1.01× |
| python-pathfinding | 1.27× | 1.39× |
| sympy | 2.40× | 0.99× |
| **Aggregate score** | **1.12×** | **1.35×** |

The better evaluator design depended strongly on the task: LLM-generated
evaluators won on broad, workload-diverse tasks like difflib and
pandas_rolling_rank, while the handwritten evaluator won where domain structure
(e.g. SymPy's symbolic comparability rules) mattered more than generic input
diversity.

## Submitting a result

Got numbers for a system on EvolveBench? Adding them is a normal pull request.
Here's how:

1. Add or update an `integrations/<system>/` folder with the exact runner and
   config you used, following the pattern in `integrations/openevolve/`.
2. Report, per task: the evaluator approach, iteration/compute budget, model(s)
   used, and the resulting speedup (or correctness/performance/combined scores,
   whichever your run produces).
3. Open a pull request adding a section and table for your system to this page,
   in the same style as the OpenEvolve table above. First time adding a new
   system's results? Feel free to propose whatever table shape fits best. We'll
   iterate on it together in review.

New tasks are welcome alongside new results. See [Contributing](contributing.md)
for the full checklist.
