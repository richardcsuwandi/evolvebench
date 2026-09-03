# EvolveBench

EvolveBench is a benchmark and execution harness for evaluating AI-driven research and optimization systems on real GitHub repositories. It asks a simple question: can an agent improve a real program while preserving the behavior its users depend on?

The benchmark currently contains 11 tasks drawn from projects including CPython, pandas, SymPy, NetworkX, python-jsonschema, and python-chess. Each task packages an editable program, correctness oracle, performance harness, and configurations for comparing evaluator designs.

[Project write-up](https://richardcsuwandi.github.io/projects/evolvebench/)

## Quick start

EvolveBench requires Python 3.8 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Validate the installation and task configuration:

```bash
eb test
```

List or inspect tasks:

```bash
eb tasks --list
eb tasks --info python-pathfinding
```

Run a task with a handwritten evaluator:

```bash
eb run \
  --task python-pathfinding \
  --evaluator handwritten \
  --iteration 50
```

EvolveBench also supports `llm_generated` and `llm_judge` evaluators. Use `all` for either argument to run the complete task or evaluator matrix.

```bash
# Generate evaluator configurations
eb config

# Summarize locally generated runs
eb summarize
```

## Benchmark design

EvolveBench treats the evaluator as part of the system being studied. Tasks can be run with three evaluator approaches:

- **Handwritten** — a task-specific evaluator designed through source inspection.
- **LLM-generated** — an evaluator synthesized from the task definition and baseline program.
- **LLM judge** — a model-based evaluator for objectives that resist a fixed metric.

Evaluators check correctness before rewarding performance. The benchmark covers algorithmic complexity, parser loops, cache policies, numerical routines, graph algorithms, and library-specific regressions.

## Included tasks

| Task | Optimization target |
| --- | --- |
| BayesianOptimization | Acquisition-function random sampling |
| difflib | Redundant similarity computations in `Differ._fancy_replace` |
| jsonschema | Equality checks affected by a Python regression |
| LMCache | LFU cache policy and minimum-frequency tracking |
| Marko | Nested parser loops |
| NetworkX | Graph algorithms |
| pandas rolling rank | Small-window rolling-rank performance |
| pymoo | Non-dominated sorting |
| python-chess | Chess-engine algorithms |
| python-pathfinding | Heap operations |
| SymPy | `Min`/`Max` local-zero discovery |

## Repository structure

```text
evolve_bench/                 # CLI, orchestration, and execution harness
tasks/                        # Task definitions, evaluators, and source fixtures
templates/                    # Evaluator configuration templates
scripts/                      # Setup and configuration utilities
integrations/openevolve/      # OpenEvolve runner and experiment configurations
registry.json                 # Task registry
task_schema.yaml              # Task validation schema
```

Generated programs, checkpoints, summaries, and benchmark outputs are written locally and ignored by Git. They are not committed as source.

## OpenEvolve integration

The harness uses [OpenEvolve](https://github.com/algorithmicsuperintelligence/openevolve) as an external dependency. The conventional development layout places both repositories under one parent directory:

```text
parent/
├── evolvebench/
└── openevolve/
    └── openevolve-run.py
```

EvolveBench-specific runners and configurations live in `integrations/openevolve/`; a complete OpenEvolve installation is still required.

## Adding a task

Contributions that expand the benchmark are welcome. To add a task:

1. Create `tasks/<new_task>/`.
2. Add `task.yaml` with the task metadata, optimization target, and evaluation criteria.
3. Add the baseline program or source fixture and a correctness oracle.
4. Implement a stable performance benchmark and handwritten evaluator.
5. Generate the remaining configurations with `eb config --task <new_task>`.
6. Register the task in `registry.json`.
7. Run `eb test` and the task-specific benchmark before opening a pull request.

A strong task contribution explains why the target matters, what behavior must remain unchanged, how timing noise is controlled, and which inputs are held out from the evolutionary search.

## Documentation site

The task table, leaderboard, and contribution guide are also published as a static
site built with [MkDocs](https://www.mkdocs.org/) (Material theme), generated
directly from `registry.json` and each task's `task.yaml` so it can't drift from the
benchmark itself:

```bash
pip install -e ".[docs]"
python scripts/generate_docs.py   # regenerate docs/tasks.md from the registry
mkdocs serve                      # preview at http://127.0.0.1:8000
```

The site deploys automatically to GitHub Pages on pushes to `main` that touch
`docs/`, `mkdocs.yml`, `registry.json`, or any `task.yaml`.

## Contributing

We welcome new benchmark tasks, stronger correctness tests, evaluator improvements, integrations with other coding agents, reproducibility tooling, and documentation fixes. Please read [CONTRIBUTING.md](CONTRIBUTING.md) for the expected task structure, validation steps, and pull-request checklist.

For proposals or questions, open a GitHub issue so the design can be discussed before substantial implementation work begins.

## License

EvolveBench's harness and task code are licensed under the [Apache License 2.0](LICENSE).
Each task vendors source from the upstream project it benchmarks, under that
project's own license — see [NOTICE.md](NOTICE.md) for the full attribution table.
