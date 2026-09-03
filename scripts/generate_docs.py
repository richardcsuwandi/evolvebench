#!/usr/bin/env python3
"""Generate docs/tasks.md from registry.json, each task's task.yaml, and its
evolve_source file.

Run after adding or editing a task so the published task page never drifts from
the actual benchmark:

    python scripts/generate_docs.py
"""

import json
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).parent.parent
REGISTRY_PATH = ROOT / "registry.json"
TASKS_DIR = ROOT / "tasks"
OUT_PATH = ROOT / "docs" / "tasks.md"

EVOLVE_BLOCK_RE = re.compile(
    r"[ \t]*#[ \t]*EVOLVE-BLOCK-START.*?\n(.*?)\n[ \t]*#[ \t]*EVOLVE-BLOCK-END",
    re.DOTALL,
)

LANGUAGE_BY_SUFFIX = {
    ".py": "python",
    ".rs": "rust",
    ".js": "javascript",
    ".ts": "typescript",
    ".go": "go",
    ".c": "c",
    ".cpp": "cpp",
}


def load_task(task_id: str) -> dict:
    task_yaml = TASKS_DIR / task_id / "task.yaml"
    with open(task_yaml) as f:
        return yaml.safe_load(f)


def extract_evolve_blocks(task_id: str, evolve_source: str) -> tuple[str, str]:
    """Return (language, snippet) for the EVOLVE-BLOCK region(s) in evolve_source."""
    source_path = TASKS_DIR / task_id / evolve_source
    text = source_path.read_text()
    blocks = EVOLVE_BLOCK_RE.findall(text)
    if not blocks:
        raise ValueError(f"No EVOLVE-BLOCK found in {source_path}")
    snippet = "\n\n# ...\n\n".join(block.strip("\n") for block in blocks)
    language = LANGUAGE_BY_SUFFIX.get(source_path.suffix, "text")
    return language, snippet


def render_task(task_id: str, meta: dict) -> list[str]:
    repo = meta.get("repository", "")
    description = meta.get("description", "")
    category = meta.get("category", "")
    tags = ", ".join(meta.get("tags", []))
    criteria = meta.get("evaluation_criteria", {})
    weights = ", ".join(f"{k}: {v}" for k, v in criteria.items())
    evolve_source = meta.get("evolve_source", "")

    lines = [
        f"### `{task_id}`",
        "",
        description,
        "",
        f"**Category:** {category} &nbsp;·&nbsp; **Evaluation weights:** {weights} "
        f"&nbsp;·&nbsp; **Tags:** {tags}",
        "",
    ]

    if evolve_source:
        language, snippet = extract_evolve_blocks(task_id, evolve_source)
        lines.append(
            f'??? example "View initial code (`{evolve_source}`)"'
        )
        lines.append("")
        lines.append(f"    ```{language}")
        for line in snippet.splitlines():
            lines.append(f"    {line}" if line else "")
        lines.append("    ```")
        lines.append("")

    if repo:
        lines.append(f"[Source repository]({repo}){{ .md-button }}")
        lines.append("")

    lines.append("---")
    lines.append("")
    return lines


def main() -> None:
    with open(REGISTRY_PATH) as f:
        registry = json.load(f)

    dataset = registry[0]
    task_ids = dataset.get("task_ids", [])

    lines = [
        "# Tasks",
        "",
        "EvolveBench draws tasks from real open-source projects. Each task "
        "packages an editable program, a correctness oracle, a performance "
        "harness, and configurations for comparing evaluator designs. Every "
        "task below shows the actual initial code inside its `EVOLVE-BLOCK`, "
        "the exact region a system is allowed to change.",
        "",
        "Want to add one? See [Contributing](contributing.md).",
        "",
    ]

    for task_id in sorted(task_ids):
        try:
            meta = load_task(task_id)
        except FileNotFoundError:
            continue
        lines.extend(render_task(task_id, meta))

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
