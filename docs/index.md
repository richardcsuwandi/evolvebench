<p align="center">
  <img src="assets/logo.svg" alt="EvolveBench logo" width="120">
</p>

# EvolveBench

**A benchmark and execution harness for evaluating AI-driven research and
optimization systems on real GitHub repositories.**

EvolveBench asks a simple question: can an agent improve a real program while
preserving the behavior its users depend on? It targets systems like
[OpenEvolve](https://github.com/algorithmicsuperintelligence/openevolve),
ShinkaEvolve, AlphaEvolve-style agents, and other evolutionary or LLM-driven
optimization loops — not just single-shot code generation.

The benchmark draws tasks from real projects, including CPython, pandas, SymPy,
NetworkX, python-jsonschema, and python-chess — see the [full task list](tasks.md)
for the current set. Each task packages an editable program, a correctness oracle,
a performance harness, and configurations for comparing three evaluator designs:
handwritten, LLM-generated, and LLM-judge.

[:material-github: View on GitHub](https://github.com/richardcsuwandi/evolvebench){ .md-button .md-button--primary }
[Browse tasks](tasks.md){ .md-button }

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

```bash
eb test                 # validate installation and task configuration
eb tasks --list         # list available tasks
eb run --task python-pathfinding --evaluator handwritten --iteration 50
```

## Why evaluator design matters

Most benchmarks treat the scoring function as fixed. EvolveBench treats the
**evaluator itself as part of the system being studied**, since how you measure
correctness and performance shapes what an optimization agent will actually find:

- **Handwritten** — a task-specific evaluator designed through source inspection.
- **LLM-generated** — an evaluator synthesized from the task definition and baseline
  program.
- **LLM judge** — a model-based evaluator for objectives that resist a fixed metric.

## Contributing

New tasks and new agent integrations are both welcome — see
[Contributing](contributing.md) for the checklist, or open an issue to discuss a
proposal first.
