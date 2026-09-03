## Summary

<!-- What does this PR change and why? -->

## Type of change

- [ ] New benchmark task
- [ ] Evaluator / harness improvement
- [ ] Agent integration (e.g. new `integrations/<system>/`)
- [ ] Reproducibility tooling / docs / CI
- [ ] Bug fix

## If this adds or changes a task

- [ ] Upstream repository and exact source version/commit used are documented in the task README
- [ ] Correctness oracle and adversarial edge cases are described
- [ ] Timing protocol (setup vs. measured execution, seeds, held-out inputs) is documented
- [ ] Upstream license file is vendored alongside the task source and added to the [NOTICE.md](../NOTICE.md) table
- [ ] Task is registered in `registry.json`
- [ ] `eb test` passes locally

## If this adds or changes an agent integration

- [ ] Lives under `integrations/<system>/`, does not modify existing integrations
- [ ] Documents the exact command(s) used to run the benchmark end to end

## Checklist

- [ ] No generated best programs, checkpoints, raw trajectories, or result directories are included
- [ ] Unrelated changes are kept out of this PR
