#!/usr/bin/env python3
"""
Run EvolveBench experiments with novelty detection comparison.

This script runs evolvebench tasks with and without novelty detection
to compare performance.

Usage:
    python run_evolvebench.py --task marko --with-novelty
    python run_evolvebench.py --task marko --without-novelty
    python run_evolvebench.py --task all --iterations 20
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

# EvolveBench path (adjust if needed)
EVOLVEBENCH_PATH = Path("/Users/codelion/Documents/GitLab/evolve-bench")
OPENEVOLVE_PATH = Path(__file__).parent.parent.parent
CONFIGS_DIR = Path(__file__).parent / "configs"
RESULTS_DIR = Path(__file__).parent / "results"

# Available tasks
TASKS = [
    "BayesianOptimization",
    "marko",
    "networkx",
    "python-chess",
    "python-pathfinding",
    "difflib",
    "pymoo",
    "sympy",
    "lmcache",
    "pandas_rolling_rank",
    "jsonschema",
    "pyparsing",
]


def run_task(task: str, with_novelty: bool, iterations: int = 20, verbose: bool = False, thompson: bool = False, warmup: bool = False):
    """Run a single evolvebench task with or without novelty, optionally with Thompson sampling or warmup."""

    if warmup:
        novelty_suffix = "warmup"
        config_name = "evolvebench_warmup.yaml"
    elif thompson:
        novelty_suffix = "thompson_sampling"
        config_name = "evolvebench_thompson_sampling.yaml"
    else:
        novelty_suffix = "with_novelty" if with_novelty else "without_novelty"
        config_name = f"evolvebench_{novelty_suffix}.yaml"
    config_path = CONFIGS_DIR / config_name

    if not config_path.exists():
        print(f"Error: Config not found: {config_path}")
        return None

    task_path = EVOLVEBENCH_PATH / "tasks" / task
    if not task_path.exists():
        print(f"Error: Task not found: {task_path}")
        return None

    # Create results directory
    result_dir = RESULTS_DIR / f"{task}_{novelty_suffix}"
    result_dir.mkdir(parents=True, exist_ok=True)

    # Find evaluator and source files
    evaluator_file = task_path / "evaluator.py"
    if not evaluator_file.exists():
        print(f"Error: Evaluator not found: {evaluator_file}")
        return None

    # Use openevolve-run.py with --directory for multi-file evolution
    openevolve_run = OPENEVOLVE_PATH / "openevolve-run.py"

    # First list blocks
    list_cmd = [
        "python", str(openevolve_run),
        "--directory", str(task_path),
        "--list-blocks"
    ]

    try:
        list_result = subprocess.run(list_cmd, capture_output=True, text=True, timeout=30)
        block_ids = []
        for line in list_result.stdout.split('\n'):
            if "ID:" in line and "'" in line:
                block_id = line.split("'")[1]
                block_ids.append(block_id)

        if not block_ids:
            print(f"No EVOLVE blocks found in {task}")
            return None

        block_id = block_ids[0]
        print(f"Using block ID: {block_id}")

    except Exception as e:
        print(f"Error listing blocks: {e}")
        return None

    # Build command
    cmd = [
        "python", str(openevolve_run),
        "--directory", str(task_path),
        "--block-id", block_id,
        "",  # Empty initial_program (not used in multi-file mode)
        str(evaluator_file),
        "--config", str(config_path),
        "--iterations", str(iterations),
        "--output", str(result_dir)
    ]

    print(f"\n{'='*60}")
    print(f"Running: {task} ({novelty_suffix})")
    print(f"Iterations: {iterations}")
    print(f"Config: {config_path}")
    print(f"Output: {result_dir}")
    print(f"{'='*60}\n")

    start_time = time.time()

    try:
        if verbose:
            result = subprocess.run(cmd, cwd=task_path, timeout=3600)
        else:
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=task_path, timeout=3600)

        elapsed = time.time() - start_time

        # Try to load results
        best_info_path = result_dir / "best" / "best_program_info.json"
        best_score = None
        if best_info_path.exists():
            with open(best_info_path) as f:
                info = json.load(f)
                best_score = info.get("metrics", {}).get("combined_score", 0)

        return {
            "task": task,
            "novelty": with_novelty,
            "thompson": thompson,
            "warmup": warmup,
            "iterations": iterations,
            "elapsed_time": elapsed,
            "return_code": result.returncode,
            "best_score": best_score,
            "success": result.returncode == 0
        }

    except subprocess.TimeoutExpired:
        return {
            "task": task,
            "novelty": with_novelty,
            "thompson": thompson,
            "warmup": warmup,
            "iterations": iterations,
            "elapsed_time": 3600,
            "return_code": -1,
            "best_score": None,
            "success": False,
            "error": "timeout"
        }
    except Exception as e:
        return {
            "task": task,
            "novelty": with_novelty,
            "thompson": thompson,
            "warmup": warmup,
            "iterations": iterations,
            "elapsed_time": time.time() - start_time,
            "return_code": -1,
            "best_score": None,
            "success": False,
            "error": str(e)
        }


def main():
    parser = argparse.ArgumentParser(description="Run EvolveBench experiments")
    parser.add_argument("--task", "-t", default="marko", help="Task to run (or 'all')")
    parser.add_argument("--with-novelty", action="store_true", help="Run WITH novelty detection")
    parser.add_argument("--without-novelty", action="store_true", help="Run WITHOUT novelty detection")
    parser.add_argument("--thompson", action="store_true", help="Run WITH Thompson sampling (includes novelty)")
    parser.add_argument("--warmup", action="store_true", help="Run WITH warmup prompt optimization (includes novelty)")
    parser.add_argument("--iterations", "-i", type=int, default=20, help="Number of iterations")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--list-tasks", action="store_true", help="List available tasks")

    args = parser.parse_args()

    if args.list_tasks:
        print("Available tasks:")
        for task in TASKS:
            print(f"  - {task}")
        return

    # Determine which configs to run
    run_thompson = args.thompson
    run_warmup = args.warmup
    if not args.with_novelty and not args.without_novelty and not args.thompson and not args.warmup:
        # Default: run both with and without novelty
        run_with = True
        run_without = True
    else:
        run_with = args.with_novelty
        run_without = args.without_novelty

    # Determine tasks
    if args.task == "all":
        tasks = TASKS
    else:
        tasks = [args.task]

    results = []

    for task in tasks:
        if run_warmup:
            result = run_task(task, with_novelty=True, iterations=args.iterations, verbose=args.verbose, warmup=True)
            if result:
                results.append(result)
                print(f"Result: {task} WARMUP - score={result.get('best_score', 'N/A')}")

        if run_thompson:
            result = run_task(task, with_novelty=True, iterations=args.iterations, verbose=args.verbose, thompson=True)
            if result:
                results.append(result)
                print(f"Result: {task} THOMPSON sampling - score={result.get('best_score', 'N/A')}")

        if run_with:
            result = run_task(task, with_novelty=True, iterations=args.iterations, verbose=args.verbose)
            if result:
                results.append(result)
                print(f"Result: {task} WITH novelty - score={result.get('best_score', 'N/A')}")

        if run_without:
            result = run_task(task, with_novelty=False, iterations=args.iterations, verbose=args.verbose)
            if result:
                results.append(result)
                print(f"Result: {task} WITHOUT novelty - score={result.get('best_score', 'N/A')}")

    # Save results summary
    if results:
        summary_path = RESULTS_DIR / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {summary_path}")

        # Print comparison table
        print("\n" + "="*60)
        print("COMPARISON TABLE")
        print("="*60)
        print(f"{'Task':<25} {'WITH':<12} {'WITHOUT':<12} {'Winner':<10}")
        print("-"*60)

        for task in tasks:
            with_result = next((r for r in results if r["task"] == task and r["novelty"]), None)
            without_result = next((r for r in results if r["task"] == task and not r["novelty"]), None)

            with_score = with_result.get("best_score", 0) if with_result else 0
            without_score = without_result.get("best_score", 0) if without_result else 0

            if with_score and without_score:
                winner = "WITH" if with_score > without_score else "WITHOUT" if without_score > with_score else "TIE"
            else:
                winner = "N/A"

            print(f"{task:<25} {with_score or 'N/A':<12} {without_score or 'N/A':<12} {winner:<10}")


if __name__ == "__main__":
    main()
