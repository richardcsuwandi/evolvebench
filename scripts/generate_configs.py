#!/usr/bin/env python3
"""
Script to generate config variants for tasks based on templates and task metadata.

Usage:
  python generate_configs.py              # generate for all tasks
  python generate_configs.py --task {task_name} # generate only for specific task
"""

import yaml
import argparse
from pathlib import Path


def load_template(template_path):
    """Load a template file."""
    with open(template_path, "r") as f:
        return f.read()


def load_task_metadata(task_path):
    """Load task metadata from task.yaml."""
    task_yaml_path = Path(task_path) / "task.yaml"
    if not task_yaml_path.exists():
        return None

    with open(task_yaml_path, "r") as f:
        return yaml.safe_load(f)


def generate_config_from_template(template_content, task_metadata):
    """Generate config from template using task metadata."""
    config = template_content

    # Replace template variables
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
        "{feature_dimensions}": str(task_metadata["task_config"]["feature_dimensions"]),
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

    # Add LLM evaluator generation model if specified
    if "llm_evaluator_generation_model" in task_metadata:
        replacements["{llm_evaluator_generation_model}"] = task_metadata[
            "llm_evaluator_generation_model"
        ]

    # Add LLM judge specific replacements
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

    # Apply replacements
    for placeholder, value in replacements.items():
        config = config.replace(placeholder, value)

    return config


def main():
    """Generate config variants for tasks (optionally scoped to one)."""
    parser = argparse.ArgumentParser(
        description="Generate EvolveBench configs from templates and task.yaml"
    )
    parser.add_argument(
        "--task",
        "-t",
        dest="task",
        default=None,
        help="Task folder name to generate (default: all)",
    )
    args = parser.parse_args()

    base_path = Path(__file__).parent.parent
    templates_path = base_path / "templates"
    tasks_path = base_path / "tasks"

    # Load templates
    handwritten_template = load_template(templates_path / "config_handwritten.yaml")
    llm_generated_template = load_template(templates_path / "config_llm_generated.yaml")
    llm_judge_template = load_template(templates_path / "config_llm_judge.yaml")

    # Enumerate task directories (optionally filtered)
    task_dirs = []
    if args.task:
        candidate = tasks_path / args.task
        if candidate.is_dir():
            task_dirs.append(candidate)
        else:
            print(f"Task '{args.task}' not found under {tasks_path}")
            return
    else:
        task_dirs = [
            d for d in tasks_path.iterdir() if d.is_dir() and not d.name.startswith(".")
        ]

    # Process tasks
    for task_dir in task_dirs:
        print(f"Processing task: {task_dir.name}")

        # Load task metadata
        task_metadata = load_task_metadata(task_dir)
        if not task_metadata:
            print(f"  No task.yaml found for {task_dir.name}, skipping...")
            continue

        # Generate config variants
        configs = {
            "config_handwritten.yaml": handwritten_template,
            "config_llm_generated.yaml": llm_generated_template,
            "config_llm_judge.yaml": llm_judge_template,
        }

        for config_name, template in configs.items():
            config_content = generate_config_from_template(template, task_metadata)
            config_path = task_dir / config_name

            # Write config file
            with open(config_path, "w") as f:
                f.write(config_content)

            print(f"  Generated {config_name}")


if __name__ == "__main__":
    main()
