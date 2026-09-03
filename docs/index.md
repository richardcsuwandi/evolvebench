<p align="center">
  <img src="assets/logo.svg" alt="EvolveBench logo" width="120">
</p>

# EvolveBench

**A benchmark and execution harness for evaluating AI-driven research and
optimization systems on real GitHub repositories.**

Where SWE-bench-style benchmarks ask an agent to patch a described bug,
EvolveBench targets algorithm discovery and codebase optimization: given a real
function inside a real, actively maintained repository, can the system find a
genuinely better algorithm or implementation, not just a faster constant factor,
while preserving the correctness guarantees the codebase already depends on? It
targets systems that search and iterate rather than emit a single patch, such as
[OpenEvolve](https://github.com/algorithmicsuperintelligence/openevolve),
[ShinkaEvolve](https://github.com/SakanaAI/ShinkaEvolve), and
[AlphaEvolve](https://deepmind.google/discover/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/)-style
agents.

The benchmark draws tasks from real projects, including CPython, pandas, SymPy,
NetworkX, python-jsonschema, and python-chess. See the [full task list](tasks.md)
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

- **Handwritten**: a task-specific evaluator designed through source inspection.
- **LLM-generated**: an evaluator synthesized from the task definition and baseline
  program.
- **LLM judge**: a model-based evaluator for objectives that resist a fixed metric.

## Contributing

New tasks and new agent integrations are both welcome. See
[Contributing](contributing.md) for the checklist, or open an issue to discuss a
proposal first.
