# Third-Party Notices

EvolveBench's harness, CLI, scripts, and task definitions
(`evolve_bench/`, `scripts/`, `templates/`, `task_schema.yaml`, `registry.json`) are
licensed under the Apache License 2.0 — see [LICENSE](LICENSE).

Each task under `tasks/<name>/` vendors source code from the upstream project it
benchmarks, under that project's own license. The exact upstream license text is
included alongside the vendored source in every task directory. This table
summarizes the source, license, and vendored path for each task:

| Task | Upstream repository | License | Vendored path |
| --- | --- | --- | --- |
| BayesianOptimization | [bayesian-optimization/BayesianOptimization](https://github.com/bayesian-optimization/BayesianOptimization) | MIT | `tasks/BayesianOptimization/bayes_opt/` |
| difflib | [python/cpython](https://github.com/python/cpython) | PSF License | `tasks/difflib/` |
| jsonschema | [python-jsonschema/jsonschema](https://github.com/python-jsonschema/jsonschema) | MIT | `tasks/jsonschema/` |
| lmcache | [LMCache/LMCache](https://github.com/LMCache/LMCache) | Apache 2.0 | `tasks/lmcache/` |
| marko | [frostming/marko](https://github.com/frostming/marko) | MIT | `tasks/marko/marko/` |
| networkx | [networkx/networkx](https://github.com/networkx/networkx) | BSD 3-Clause | `tasks/networkx/networkx/` |
| pandas_rolling_rank | [pandas-dev/pandas](https://github.com/pandas-dev/pandas) | BSD 3-Clause | `tasks/pandas_rolling_rank/` |
| pymoo | [anyoptimization/pymoo](https://github.com/anyoptimization/pymoo) | Apache 2.0 | `tasks/pymoo/pymoo/` |
| python-chess | [niklasf/python-chess](https://github.com/niklasf/python-chess) | **GPL-3.0-or-later** | `tasks/python-chess/chess/` |
| python-pathfinding | [brean/python-pathfinding](https://github.com/brean/python-pathfinding) | MIT | `tasks/python-pathfinding/` |
| sympy | [sympy/sympy](https://github.com/sympy/sympy) | BSD 3-Clause | `tasks/sympy/` |

**Note on `python-chess`**: this is the only vendored task under a copyleft license
(GPL-3.0-or-later). Its `chess/` directory retains its original GPL-3.0 terms
independently of the Apache-2.0 license covering the rest of this repository.

New task contributions must include the upstream project's license file in the
vendored source directory and document it in this table — see
[CONTRIBUTING.md](CONTRIBUTING.md).
