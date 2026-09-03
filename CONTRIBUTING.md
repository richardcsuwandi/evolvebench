# Contributing to EvolveBench

Thank you for helping improve EvolveBench. Contributions are welcome across benchmark tasks, evaluators, agent integrations, reproducibility tooling, tests, and documentation.

## Before you start

For a new task or a substantial evaluator change, open an issue first. Describe the optimization target, upstream project, correctness oracle, representative workloads, and expected runtime. Early discussion helps keep tasks comparable and avoids duplicating work.

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Run the repository checks:

```bash
eb test
python -m compileall -q evolve_bench scripts
black --check evolve_bench scripts
flake8 evolve_bench scripts
```

`eb test` also reports whether the API credentials required for model-backed runs are configured. Never commit API keys or `.env` files.

## Adding a benchmark task

Create `tasks/<task_name>/` with the following core files:

```text
tasks/<task_name>/
├── README.md
├── task.yaml
├── config_handwritten.yaml
├── config_llm_generated.yaml
├── config_llm_judge.yaml
├── evaluator.py
├── generated_evaluator.py       # when applicable
├── benchmark.py
└── <source fixtures>
```

Then add the task to `registry.json`.

Every task should provide:

- A narrowly defined edit target.
- A trusted correctness oracle and adversarial edge cases.
- Representative and held-out workloads.
- A deterministic setup where possible, including fixed random seeds.
- A timing protocol that separates setup or compilation from measured execution.
- A continuous performance signal for valid candidates.
- Timeouts and clear diagnostics for invalid or nonterminating candidates.
- A task README documenting dependencies, commands, and known limitations.

Do not include generated best programs, checkpoints, raw trajectories, local environments, or result directories in a pull request.

If the task vendors source code from an upstream project, include only the files
needed to run the evaluator and benchmark (not the upstream project's own test suite,
docs, or CI config), copy that project's license file into the vendored directory
verbatim, and add a row to the attribution table in [NOTICE.md](NOTICE.md) naming the
upstream repository, license, and vendored path. Flag clearly in the task README if
the upstream license is copyleft (e.g. GPL) rather than permissive.

## Pull-request checklist

In the pull-request description, include:

- The problem and why it belongs in EvolveBench.
- The upstream repository and exact source version used for any fixture.
- The correctness properties enforced by the evaluator.
- The commands needed to reproduce validation and timing.
- Hardware, software versions, run budget, and random seeds for reported measurements.
- Known limitations or cases not covered by the evaluator.

Keep unrelated changes in separate pull requests. A contribution should be reviewable without relying on uncommitted result artifacts.

## Reporting problems

When reporting an evaluator bug or a questionable benchmark result, include the task name, evaluator mode, command, environment, relevant logs, and the smallest reproducible example you can provide.
