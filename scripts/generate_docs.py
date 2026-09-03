#!/usr/bin/env python3
"""Generate docs/tasks.md from registry.json and each task's task.yaml.

Run after adding or editing a task so the published task table never drifts from
the actual benchmark:

    python scripts/generate_docs.py
"""

import json
from pathlib import Path

import yaml

ROOT = Path(__file__).parent.parent
REGISTRY_PATH = ROOT / "registry.json"
TASKS_DIR = ROOT / "tasks"
OUT_PATH = ROOT / "docs" / "tasks.md"


def load_task(task_id: str) -> dict:
    task_yaml = TASKS_DIR / task_id / "task.yaml"
    with open(task_yaml) as f:
        return yaml.safe_load(f)


def render_row(task_id: str, meta: dict) -> str:
    repo = meta.get("repository", "")
    repo_link = f"[source]({repo})" if repo else ""
    description = meta.get("description", "").replace("|", "\\|")
    category = meta.get("category", "")
    tags = ", ".join(meta.get("tags", []))
    criteria = meta.get("evaluation_criteria", {})
    weights = ", ".join(f"{k}: {v}" for k, v in criteria.items())
    readme_url = (
        f"https://github.com/richardcsuwandi/evolvebench/tree/main/tasks/{task_id}"
    )
    readme_link = f"[README]({readme_url})"
    row = f"| `{task_id}` | {description} | {category} | {weights} |"
    return f"{row} {tags} | {repo_link} · {readme_link} |"


def main() -> None:
    with open(REGISTRY_PATH) as f:
        registry = json.load(f)

    dataset = registry[0]
    task_ids = dataset.get("task_ids", [])

    lines = [
        "# Tasks",
        "",
        f"EvolveBench currently ships **{len(task_ids)} tasks** drawn from real "
        "open-source projects. Each task packages an editable program, a "
        "correctness oracle, a performance harness, and configurations for "
        "comparing evaluator designs.",
        "",
        "Want to add one? See [Contributing](contributing.md).",
        "",
        "| Task | Optimization target | Category | Evaluation weights | Tags | Links |",
        "| --- | --- | --- | --- | --- | --- |",
    ]

    for task_id in sorted(task_ids):
        try:
            meta = load_task(task_id)
        except FileNotFoundError:
            continue
        lines.append(render_row(task_id, meta))

    lines.append("")
    lines.append(
        f"Evaluator approaches available for every task: "
        f"{', '.join(dataset.get('evaluator_approaches', []))}."
    )
    lines.append("")

    OUT_PATH.parent.mkdir(exist_ok=True)
    OUT_PATH.write_text("\n".join(lines))
    print(f"Wrote {OUT_PATH} ({len(task_ids)} tasks)")


if __name__ == "__main__":
    main()
