#!/usr/bin/env python3
"""
EvolveBench CLI

Command-line interface for running and comparing evolutionary coding experiments.
"""

import json
import os
import re
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import typer
import yaml
from rich.console import Console
from rich.table import Table

from .harness import BenchmarkHarness

app = typer.Typer(no_args_is_help=True)
console = Console()


# Global options
@app.callback()
def main(
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose output"
    ),
    config_path: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Path to config file"
    ),
):
    """EvolveBench: Benchmark and Execution Harness for AI-Driven Optimization"""
    pass


@app.command()
def run(
    tasks: str = typer.Option(
        "all", "--task", "-t", help="Tasks to run (comma-separated or 'all')"
    ),
    evaluator: str = typer.Option(
        "all",
        "--evaluator",
        "-e",
        help="Evaluator approaches (comma-separated or 'all')",
    ),
    iterations: int = typer.Option(
        50, "--iteration", "-i", help="Number of evolution iterations"
    ),
    output_dir: Path = typer.Option(
        Path("./results"), "--output", "-o", help="Output directory for results"
    ),
    parallel: int = typer.Option(2, "--parallel", "-p", help="Number of parallel runs"),
    timeout: int = typer.Option(3600, "--timeout", help="Timeout per run in seconds"),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose harness output"
    ),
    log_level: Optional[str] = typer.Option(
        None,
        "--log-level",
        "-l",
        help="OpenEvolve log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    ),
):
    """Run benchmark experiments on specified tasks and approaches."""

    # Load registry for dynamic task/evaluator lists
    registry_path = Path(__file__).parent.parent / "registry.json"
    with open(registry_path, "r") as f:
        registry = json.load(f)
    dataset = registry[0] if registry else {"task_ids": [], "evaluator_approaches": []}

    # Parse task list (dynamic from registry when 'all')
    if tasks == "all":
        task_list = list(dataset.get("task_ids", []))
    else:
        task_list = [t.strip() for t in tasks.split(",")]

    # Parse evaluator list (dynamic from registry when 'all')
    if evaluator == "all":
        evaluator_list = list(
            dataset.get(
                "evaluator_approaches", ["handwritten", "llm_generated", "llm_judge"]
            )
        )
    else:
        evaluator_list = [e.strip() for e in evaluator.split(",")]

    console.print("[bold green]Running EvolveBench[/bold green]")
    console.print(f"Tasks: {', '.join(task_list)}")
    console.print(f"Evaluators: {', '.join(evaluator_list)}")
    console.print(f"Iterations: {iterations}")
    console.print(f"Output: {output_dir}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize harness
    harness = BenchmarkHarness(
        output_dir, verbose=verbose, openevolve_log_level=log_level
    )

    # Run experiments
    try:
        harness.run_experiments(
            tasks=task_list,
            evaluators=evaluator_list,
            iterations=iterations,
            parallel=parallel,
            timeout=timeout,
        )

        console.print("[bold green]✓ Experiments completed successfully![/bold green]")
        console.print(f"Results saved to: {output_dir}")

    except Exception as e:
        console.print(f"[bold red]✗ Error running experiments: {e}[/bold red]")
        sys.exit(1)


@app.command()
def tasks(
    list_tasks: bool = typer.Option(False, "--list", "-l", help="List available tasks"),
    info: Optional[str] = typer.Option(
        None, "--info", help="Show info for specific task"
    ),
):
    """Manage and inspect tasks."""

    if list_tasks:
        # Load registry
        registry_path = Path(__file__).parent.parent / "registry.json"
        with open(registry_path, "r") as f:
            registry = json.load(f)

        # Create table
        table = Table(title="Available Tasks")
        table.add_column("Task ID", style="cyan")
        table.add_column("Description", style="magenta")
        table.add_column("Category", style="blue")

        # Load task metadata
        tasks_path = Path(__file__).parent.parent / "tasks"
        for task_id in registry[0]["task_ids"]:
            task_yaml_path = tasks_path / task_id / "task.yaml"
            if task_yaml_path.exists():
                with open(task_yaml_path, "r") as f:
                    task_meta = yaml.safe_load(f)

                table.add_row(
                    task_id,
                    task_meta.get("description", "N/A")[:50] + "...",
                    task_meta.get("category", "N/A"),
                )

        console.print(table)

    elif info:
        # Show detailed info for specific task
        task_path = Path(__file__).parent.parent / "tasks" / info / "task.yaml"
        if not task_path.exists():
            console.print(f"[bold red]Task '{info}' not found[/bold red]")
            sys.exit(1)

        with open(task_path, "r") as f:
            task_meta = yaml.safe_load(f)

        console.print(f"[bold cyan]Task: {info}[/bold cyan]")
        console.print(f"Description: {task_meta.get('description', 'N/A')}")
        console.print(f"Repository: {task_meta.get('repository', 'N/A')}")
        console.print(f"Category: {task_meta.get('category', 'N/A')}")
        console.print(f"Tags: {', '.join(task_meta.get('tags', []))}")
        console.print(
            f"Evaluation Criteria: {task_meta.get('evaluation_criteria', {})}"
        )
        console.print(f"Baseline Metrics: {task_meta.get('baseline_metrics', {})}")


# Helper functions for summarize command
def _parse_percentage_to_x(percentage_str: str) -> float:
    """Convert percentage string to multiplier (x format)."""
    pct = float(percentage_str.replace("%", "").strip())
    return 1.0 + (pct / 100.0)


def _extract_speedup_from_line(line: str) -> Optional[float]:
    """Extract speedup value from a line, handling both x format and percentage."""
    x_match = re.search(r"(\d+\.?\d*)\s*x", line, re.IGNORECASE)
    if x_match:
        return float(x_match.group(1))

    pct_match = re.search(r"([+-]?\d+\.?\d*)\s*%", line)
    if pct_match:
        return _parse_percentage_to_x(pct_match.group(1))

    return None


def _parse_output_txt(file_path: Path) -> Tuple[Optional[float], Optional[float]]:
    """Parse output.txt file and extract handwritten and LLM-generated speedups."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception as e:
        console.print(f"[yellow]Warning: Error reading {file_path}: {e}[/yellow]")
        return None, None

    handwritten_speedup = None
    llm_generated_speedup = None
    lines = content.split("\n")

    handwritten_candidates = []
    llm_candidates = []

    # Pattern 1: Look for summary lines
    for line in lines:
        line_lower = line.lower()
        if "handwritten" in line_lower and (
            "average" in line_lower or "mean" in line_lower or "overall" in line_lower
        ):
            speedup = _extract_speedup_from_line(line)
            if speedup is not None:
                handwritten_candidates.append(("summary", speedup))
        if (
            "llm" in line_lower
            or "llm-generated" in line_lower
            or "llm_generated" in line_lower
        ) and (
            "average" in line_lower or "mean" in line_lower or "overall" in line_lower
        ):
            speedup = _extract_speedup_from_line(line)
            if speedup is not None:
                llm_candidates.append(("summary", speedup))

    # Pattern 2: Look for TOTAL lines
    for line in lines:
        if "total" in line.lower() and "│" in line:
            speedups = re.findall(r"(\d+\.?\d*)\s*x", line, re.IGNORECASE)
            if len(speedups) >= 2:
                handwritten_candidates.append(("total", float(speedups[0])))
                llm_candidates.append(("total", float(speedups[1])))

    # Pattern 3: Look for all speedup lines
    for i, line in enumerate(lines):
        line_lower = line.lower()
        if "🚀" in line and "speedup" in line_lower:
            for j in range(i + 1, min(i + 5, len(lines))):
                next_line = lines[j]
                next_line_lower = next_line.lower()
                if "handwritten" in next_line_lower:
                    speedup = _extract_speedup_from_line(next_line)
                    if speedup is not None:
                        if "x" in next_line_lower:
                            handwritten_candidates.append(("explicit_x", speedup))
                        else:
                            handwritten_candidates.append(("percentage", speedup))
                if "llm" in next_line_lower or "llm-generated" in next_line_lower:
                    speedup = _extract_speedup_from_line(next_line)
                    if speedup is not None:
                        if "x" in next_line_lower:
                            llm_candidates.append(("explicit_x", speedup))
                        else:
                            llm_candidates.append(("percentage", speedup))

        if "handwritten" in line_lower and ("speedup" in line_lower or "🚀" in line):
            speedup = _extract_speedup_from_line(line)
            if speedup is not None:
                if "x" in line_lower:
                    handwritten_candidates.append(("explicit_x", speedup))
                else:
                    handwritten_candidates.append(("percentage", speedup))

        if ("llm" in line_lower or "llm-generated" in line_lower) and (
            "speedup" in line_lower or "🚀" in line
        ):
            speedup = _extract_speedup_from_line(line)
            if speedup is not None:
                if "x" in line_lower:
                    llm_candidates.append(("explicit_x", speedup))
                else:
                    llm_candidates.append(("percentage", speedup))

    priority_order = {"summary": 0, "total": 1, "explicit_x": 2, "percentage": 3}

    if handwritten_candidates:
        best_candidate = None
        best_priority = 99
        for candidate in handwritten_candidates:
            priority = priority_order.get(candidate[0], 99)
            if priority <= best_priority:
                best_priority = priority
                best_candidate = candidate
        handwritten_speedup = best_candidate[1] if best_candidate else None

    if llm_candidates:
        best_candidate = None
        best_priority = 99
        for candidate in llm_candidates:
            priority = priority_order.get(candidate[0], 99)
            if priority <= best_priority:
                best_priority = priority
                best_candidate = candidate
        llm_generated_speedup = best_candidate[1] if best_candidate else None

    return handwritten_speedup, llm_generated_speedup


def _format_speedup(speedup: Optional[float]) -> Optional[str]:
    """Format speedup as string in "X.XXx" format."""
    if speedup is None:
        return None
    return f"{speedup:.2f}x"


def _generate_summaries(tasks_dir: Path):
    """Generate summary.json files for each task from output.txt files."""
    task_dirs = [
        d for d in tasks_dir.iterdir() if d.is_dir() and not d.name.startswith("_")
    ]

    # console.print(f"Found {len(task_dirs)} task directories")

    processed = 0
    skipped = 0
    errors = 0

    for task_dir in sorted(task_dirs):
        output_txt = task_dir / "output.txt"

        if not output_txt.exists():
            console.print(
                f"[yellow]⚠️  Skipping {task_dir.name}: output.txt not found[/yellow]"
            )
            skipped += 1
            continue

        handwritten_speedup, llm_generated_speedup = _parse_output_txt(output_txt)

        if handwritten_speedup is None and llm_generated_speedup is None:
            console.print(
                f"[yellow]⚠️  Could not extract speedups from {task_dir.name}[/yellow]"
            )
            skipped += 1
            continue

        summary = {}
        if handwritten_speedup is not None:
            summary["handwritten_speedup"] = _format_speedup(handwritten_speedup)
        if llm_generated_speedup is not None:
            summary["llm_generated_speedup"] = _format_speedup(llm_generated_speedup)

        summary_json = task_dir / "summary.json"
        try:
            with open(summary_json, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
            processed += 1
        except Exception as e:
            console.print(f"[red]✗ Error writing {summary_json}: {e}[/red]")
            errors += 1


def _parse_speedup(speedup_str: Optional[str]) -> Optional[float]:
    """Parse speedup string like '1.54x' to float like 1.54."""
    if speedup_str is None:
        return None
    try:
        return float(speedup_str.replace("x", "").strip())
    except (ValueError, AttributeError):
        return None


def _evolvebench_score(values: List[float]) -> float:
    """Calculate EvolveBench Score, which is the harmonic mean of a list of values."""
    if not values:
        return 0.0
    positive_values = [v for v in values if v > 0]
    if not positive_values:
        return 0.0
    reciprocal_sum = sum(1.0 / v for v in positive_values)
    return len(positive_values) / reciprocal_sum


def _calculate_scores(tasks_dir: Path):
    """Aggregate summaries and calculate EvolveBench scores."""
    handwritten_speedups = []
    llm_generated_speedups = []
    task_data = []

    task_dirs = [
        d for d in tasks_dir.iterdir() if d.is_dir() and not d.name.startswith("_")
    ]

    for task_dir in sorted(task_dirs):
        summary_json = task_dir / "summary.json"

        if not summary_json.exists():
            continue

        try:
            with open(summary_json, "r", encoding="utf-8") as f:
                summary = json.load(f)

            hw_speedup = _parse_speedup(summary.get("handwritten_speedup"))
            llm_speedup = _parse_speedup(summary.get("llm_generated_speedup"))

            task_info = {
                "task": task_dir.name,
                "handwritten_speedup": hw_speedup,
                "llm_generated_speedup": llm_speedup,
            }

            if hw_speedup is not None:
                handwritten_speedups.append(hw_speedup)
            if llm_speedup is not None:
                llm_generated_speedups.append(llm_speedup)

            task_data.append(task_info)
        except Exception as e:
            console.print(
                f"[yellow]Warning: Error reading {summary_json}: {e}[/yellow]"
            )
            continue

    # Calculate EvolveBench scores
    hw_evolvebench_score = (
        _evolvebench_score(handwritten_speedups) if handwritten_speedups else None
    )
    llm_evolvebench_score = (
        _evolvebench_score(llm_generated_speedups) if llm_generated_speedups else None
    )

    # Display summary table
    summary_table = Table(
        title="Overall EvolveBench Score", show_header=True, header_style="bold cyan"
    )
    summary_table.add_column("Evaluator Type", style="magenta", justify="left")
    summary_table.add_column("EvolveBench Score", style="green", justify="right")

    hw_score = round(hw_evolvebench_score, 2) if hw_evolvebench_score else None
    llm_score = round(llm_evolvebench_score, 2) if llm_evolvebench_score else None
    ratio = (
        round(llm_evolvebench_score / hw_evolvebench_score, 2)
        if (hw_evolvebench_score and llm_evolvebench_score and hw_evolvebench_score > 0)
        else None
    )

    summary_table.add_row(
        "Handwritten", f"{hw_score:.2f}x" if hw_score is not None else "N/A"
    )
    summary_table.add_row(
        "LLM-Generated", f"{llm_score:.2f}x" if llm_score is not None else "N/A"
    )
    summary_table.add_row(
        "Ratio (LLM/HW)", f"{ratio:.2f}x" if ratio is not None else "N/A"
    )

    console.print("\n")
    console.print(summary_table)

    # Per-task comparison table
    task_table = Table(
        title="Per-Task Speedup Comparison", show_header=True, header_style="bold cyan"
    )
    task_table.add_column("Task", style="cyan", justify="left")
    task_table.add_column("Handwritten", style="yellow", justify="right")
    task_table.add_column("LLM-Generated", style="green", justify="right")
    task_table.add_column("LLM/HW Ratio", style="magenta", justify="right")

    for task in task_data:
        hw = task["handwritten_speedup"]
        llm = task["llm_generated_speedup"]

        hw_str = f"{hw:.2f}x" if hw is not None else "N/A"
        llm_str = f"{llm:.2f}x" if llm is not None else "N/A"

        if hw is not None and llm is not None and hw > 0:
            ratio_val = llm / hw
            ratio_str = f"{ratio_val:.2f}x"
            if ratio_val > 1.1:
                ratio_style = "bold green"
            elif ratio_val < 0.9:
                ratio_style = "bold red"
            else:
                ratio_style = "white"
        else:
            ratio_str = "N/A"
            ratio_style = "white"

        task_table.add_row(
            task["task"], hw_str, llm_str, f"[{ratio_style}]{ratio_str}[/]"
        )

    console.print("\n")
    console.print(task_table)

    # Save aggregated results
    aggregated = {
        "tasks": task_data,
        "statistics": {
            "handwritten": {
                "evolvebench_score": hw_score,
                "count": len(handwritten_speedups),
                "values": [round(v, 2) for v in handwritten_speedups],
            },
            "llm_generated": {
                "evolvebench_score": llm_score,
                "count": len(llm_generated_speedups),
                "values": [round(v, 2) for v in llm_generated_speedups],
            },
        },
        "comparison": {
            "evolvebench_score_handwritten": hw_score,
            "evolvebench_score_llm_generated": llm_score,
            "score_ratio": ratio,
        },
    }

    output_file = tasks_dir / "evolvebench_score.json"
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(aggregated, f, indent=2)
        console.print(
            f"\n[bold green]✓[/bold green] Saved aggregated results to: {output_file}"
        )
    except Exception as e:
        console.print(f"\n[bold red]⚠[/bold red] Error saving results: {e}")


@app.command()
def summarize(
    tasks_dir: Optional[Path] = typer.Option(
        None, "--tasks-dir", help="Path to tasks directory (default: ./tasks)"
    ),
    skip_summary: bool = typer.Option(
        False, "--skip-summary", help="Skip summary generation, only calculate scores"
    ),
    skip_score: bool = typer.Option(
        False, "--skip-score", help="Skip score calculation, only generate summaries"
    ),
):
    """
    Summarize benchmark results and calculate EvolveBench scores.

    This command:
    1. Parses output.txt files in each task directory to extract speedup information
    2. Generates summary.json files for each task
    3. Aggregates all summaries and calculates EvolveBench scores (harmonic mean)
    """
    # Determine tasks directory
    if tasks_dir is None:
        tasks_dir = Path(__file__).parent.parent / "tasks"
    else:
        tasks_dir = Path(tasks_dir)

    if not tasks_dir.exists():
        console.print(
            f"[bold red]Error: Tasks directory not found: {tasks_dir}[/bold red]"
        )
        sys.exit(1)

    # Step 1: Generate summaries (unless skipped)
    if not skip_summary:
        console.print("[bold green]Summarizing results...[/bold green]")
        _generate_summaries(tasks_dir)
    else:
        console.print("[yellow]Skipping summary generation[/yellow]")

    # Step 2: Calculate EvolveBench scores (unless skipped)
    if not skip_score:
        console.print("[bold green]Calculating EvolveBench scores...[/bold green]")
        _calculate_scores(tasks_dir)
    else:
        console.print("[yellow]Skipping score calculation[/yellow]")

    console.print("[bold green]✓ Analysis complete![/bold green]")


@app.command()
def config(
    task: Optional[str] = typer.Option(
        None,
        "--task",
        "-t",
        help="Task folder name to generate configs for (default: all tasks)",
    ),
):
    """
    Generate config variants for tasks based on templates and task metadata.

    This command generates config_handwritten.yaml, config_llm_generated.yaml, and
    config_llm_judge.yaml files for tasks by filling in templates with values from
    task.yaml.
    """
    base_path = Path(__file__).parent.parent
    templates_path = base_path / "templates"
    tasks_path = base_path / "tasks"

    # Check templates exist
    template_files = [
        "config_handwritten.yaml",
        "config_llm_generated.yaml",
        "config_llm_judge.yaml",
    ]

    for template_file in template_files:
        template_path = templates_path / template_file
        if not template_path.exists():
            console.print(
                f"[bold red]Error: Template not found: {template_path}[/bold red]"
            )
            sys.exit(1)

    # Load templates
    def load_template(template_path):
        with open(template_path, "r") as f:
            return f.read()

    handwritten_template = load_template(templates_path / "config_handwritten.yaml")
    llm_generated_template = load_template(templates_path / "config_llm_generated.yaml")
    llm_judge_template = load_template(templates_path / "config_llm_judge.yaml")

    # Enumerate task directories
    task_dirs = []
    if task:
        candidate = tasks_path / task
        if candidate.is_dir():
            task_dirs.append(candidate)
        else:
            console.print(
                f"[bold red]Task '{task}' not found under {tasks_path}[/bold red]"
            )
            sys.exit(1)
    else:
        task_dirs = [
            d for d in tasks_path.iterdir() if d.is_dir() and not d.name.startswith(".")
        ]

    if not task_dirs:
        console.print("[yellow]No tasks found[/yellow]")
        return

    # Process tasks
    def load_task_metadata(task_path):
        task_yaml_path = Path(task_path) / "task.yaml"
        if not task_yaml_path.exists():
            return None
        with open(task_yaml_path, "r") as f:
            return yaml.safe_load(f)

    def generate_config_from_template(template_content, task_metadata):
        config = template_content
        replacements = {
            "{max_iterations}": str(task_metadata["task_config"]["max_iterations"]),
            "{diff_based_evolution}": str(
                task_metadata["task_config"]["diff_based_evolution"]
            ).lower(),
            "{max_code_length}": str(task_metadata["task_config"]["max_code_length"]),
            "{language}": task_metadata["task_config"]["language"],
            "{llm_model}": task_metadata["task_config"]["llm_model"],
            "{api_base}": task_metadata["task_config"]["api_base"],
            "{population_size}": str(task_metadata["task_config"]["population_size"]),
            "{archive_size}": str(task_metadata["task_config"]["archive_size"]),
            "{feature_dimensions}": str(
                task_metadata["task_config"]["feature_dimensions"]
            ),
            "{task_description}": task_metadata["description"],
            "{optimization_goal}": (
                f"Improve {task_metadata['optimization_target']} performance"
            ),
            "{evaluation_criteria}": (
                "correctness "
                f"({task_metadata['evaluation_criteria']['correctness']}), "
                "performance "
                f"({task_metadata['evaluation_criteria']['performance']})"
            ),
            "{required_metrics}": str(task_metadata["required_metrics"]),
        }

        if "llm_evaluator_generation_model" in task_metadata:
            replacements["{llm_evaluator_generation_model}"] = task_metadata[
                "llm_evaluator_generation_model"
            ]

        if "llm_judge_config" in task_metadata:
            judge_config = task_metadata["llm_judge_config"]
            replacements.update(
                {
                    "{evaluation_criteria}": judge_config["evaluation_criteria"],
                    "{scoring_rubric}": judge_config["scoring_rubric"],
                    "{task_context}": judge_config["task_context"],
                    "{llm_criteria}": str(judge_config["llm_criteria"]),
                    "{json_format}": judge_config["json_format"],
                }
            )

        for placeholder, value in replacements.items():
            config = config.replace(placeholder, value)

        return config

    console.print("[bold green]Generating config files...[/bold green]")
    processed = 0
    skipped = 0

    for task_dir in task_dirs:
        console.print(f"Processing task: {task_dir.name}")

        task_metadata = load_task_metadata(task_dir)
        if not task_metadata:
            console.print(
                f"  [yellow]No task.yaml found for {task_dir.name}, "
                "skipping...[/yellow]"
            )
            skipped += 1
            continue

        configs = {
            "config_handwritten.yaml": handwritten_template,
            "config_llm_generated.yaml": llm_generated_template,
            "config_llm_judge.yaml": llm_judge_template,
        }

        for config_name, template in configs.items():
            config_content = generate_config_from_template(template, task_metadata)
            config_path = task_dir / config_name

            with open(config_path, "w") as f:
                f.write(config_content)

            console.print(f"  [green]✓[/green] Generated {config_name}")

        processed += 1

    console.print("\n[bold green]✓ Config generation complete![/bold green]")
    console.print(f"Processed: {processed}, Skipped: {skipped}")


@app.command()
def test():
    """
    Test EvolveBench setup to verify configuration.

    This command checks:
    - Templates exist and are valid YAML
    - Tasks have required files and valid task.yaml
    - Registry.json is valid
    - Package structure is correct
    - API credentials are set
    """
    base_path = Path(__file__).parent.parent

    def test_templates():
        templates_dir = base_path / "templates"
        template_files = [
            "config_handwritten.yaml",
            "config_llm_generated.yaml",
            "config_llm_judge.yaml",
        ]

        console.print("🔍 Testing templates...")
        for template_file in template_files:
            template_path = templates_dir / template_file
            if not template_path.exists():
                console.print(f"❌ Template not found: {template_path}")
                return False

            try:
                with open(template_path, "r") as f:
                    yaml.safe_load(f)
                console.print(f"✅ {template_file} is valid YAML")
            except yaml.YAMLError as e:
                console.print(f"❌ {template_file} has YAML error: {e}")
                return False

        return True

    def test_tasks():
        tasks_dir = base_path / "tasks"
        required_files = [
            "task.yaml",
            "config_handwritten.yaml",
            "config_llm_generated.yaml",
            "config_llm_judge.yaml",
        ]

        console.print("\n🔍 Testing tasks...")
        task_dirs = [
            d for d in tasks_dir.iterdir() if d.is_dir() and not d.name.startswith(".")
        ]

        for task_dir in task_dirs:
            console.print(f"  Testing {task_dir.name}...")

            for required_file in required_files:
                file_path = task_dir / required_file
                if not file_path.exists():
                    console.print(f"    ❌ Missing: {required_file}")
                    return False
                console.print(f"    ✅ {required_file}")

            try:
                with open(task_dir / "task.yaml", "r") as f:
                    task_meta = yaml.safe_load(f)

                required_fields = [
                    "task_id",
                    "description",
                    "evaluation_criteria",
                    "task_config",
                ]
                for field in required_fields:
                    if field not in task_meta:
                        console.print(f"    ❌ Missing field in task.yaml: {field}")
                        return False

                console.print("    ✅ task.yaml is valid")

            except yaml.YAMLError as e:
                console.print(f"    ❌ task.yaml has YAML error: {e}")
                return False

        return True

    def test_registry():
        console.print("\n🔍 Testing registry...")

        registry_path = base_path / "registry.json"
        if not registry_path.exists():
            console.print("❌ registry.json not found")
            return False

        try:
            with open(registry_path, "r") as f:
                registry = json.load(f)

            if not isinstance(registry, list) or len(registry) == 0:
                console.print("❌ registry.json should be a non-empty list")
                return False

            required_fields = [
                "name",
                "version",
                "description",
                "task_ids",
                "evaluator_approaches",
            ]
            for field in required_fields:
                if field not in registry[0]:
                    console.print(f"❌ Missing field in registry: {field}")
                    return False

            console.print("✅ registry.json is valid")
            return True

        except json.JSONDecodeError as e:
            console.print(f"❌ registry.json has JSON error: {e}")
            return False

    def test_package():
        console.print("\n🔍 Testing package structure...")

        package_files = [
            "evolve_bench/__init__.py",
            "evolve_bench/cli.py",
            "evolve_bench/harness.py",
            "pyproject.toml",
            "README.md",
        ]

        for file_path in package_files:
            if not (base_path / file_path).exists():
                console.print(f"❌ Missing package file: {file_path}")
                return False
            console.print(f"✅ {file_path}")

        return True

    def test_api_key():
        console.print("\n🔍 Testing API credentials...")
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key or not api_key.strip():
            console.print(
                "❌ OPENAI_API_KEY is not set. "
                "LLM-based evaluators will fail (401/timeout)."
            )
            return False
        console.print("✅ OPENAI_API_KEY is set")
        return True

    console.print("🧪 EvolveBench Setup Test")
    console.print("=" * 30)

    tests = [
        test_templates,
        test_tasks,
        test_registry,
        test_package,
        test_api_key,
    ]

    all_passed = True
    for test_func in tests:
        if not test_func():
            all_passed = False

    console.print("\n" + "=" * 30)
    if all_passed:
        console.print(
            "[bold green]🎉 All tests passed! EvolveBench is ready to use.[/bold green]"
        )
    else:
        console.print(
            "[bold red]❌ Some tests failed. Please check the setup.[/bold red]"
        )
        sys.exit(1)


if __name__ == "__main__":
    app()
