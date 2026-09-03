# Contributing

EvolveBench welcomes new benchmark tasks, evaluator improvements, integrations with
other AI-driven optimization systems, reproducibility tooling, and documentation
fixes.

The full, canonical guide (expected task structure, validation steps, and the
pull-request checklist) lives in
[CONTRIBUTING.md](https://github.com/richardcsuwandi/evolvebench/blob/main/CONTRIBUTING.md)
in the repository root, so it stays a single source of truth alongside the code it
describes.

## In short

**Adding a task**

1. Open an issue first using the [new task template](https://github.com/richardcsuwandi/evolvebench/issues/new?template=new_task.md)
   to discuss the optimization target, upstream project, and correctness oracle.
2. Create `tasks/<task_name>/` with `task.yaml`, evaluator(s), a benchmark script,
   the baseline source, and a README documenting dependencies and known limitations.
3. Vendor only what the evaluator/benchmark actually use, include the upstream
   project's license file, and register the task in `registry.json`.
4. Run `eb test` and the task-specific benchmark before opening a pull request.

**Adding an agent/system integration**

Follow the [integration request template](https://github.com/richardcsuwandi/evolvebench/issues/new?template=integration_request.md)
and add a new `integrations/<system>/` folder alongside the existing
`integrations/openevolve/`, without modifying other integrations.

**Reporting a result**

See [Leaderboard](leaderboard.md) for how to submit scores for a system you've run
against EvolveBench.
