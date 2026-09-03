"""
Benchmark Harness for EvolveBench

Orchestrates OpenEvolve runs for different tasks and evaluator modes.
"""

import json
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List

from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
)

console = Console()


class BenchmarkHarness:
    """Orchestrates benchmark experiments."""

    def __init__(
        self, output_dir: Path, verbose: bool = False, openevolve_log_level: str = None
    ):
        self.output_dir = output_dir
        self.base_path = Path(__file__).parent.parent
        self.verbose = verbose
        self.openevolve_log_level = openevolve_log_level

    def run_experiments(
        self,
        tasks: List[str],
        evaluators: List[str],
        iterations: int = 50,
        parallel: int = 2,
        timeout: int = 3600,
    ) -> Dict:
        """Run experiments for specified tasks and evaluator approaches."""

        # Create experiment plan
        experiments = []
        for task in tasks:
            for evaluator in evaluators:
                experiments.append((task, evaluator))

        console.print(
            f"[bold blue]Running {len(experiments)} experiments...[/bold blue]"
        )

        # Run experiments with progress tracking
        results = {}
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:

            task_progress = progress.add_task(
                "Running experiments...", total=len(experiments)
            )

            # Run experiments in parallel batches
            with ThreadPoolExecutor(max_workers=parallel) as executor:
                future_to_exp = {
                    executor.submit(
                        self._run_single_experiment,
                        task,
                        evaluator,
                        iterations,
                        timeout,
                    ): (task, evaluator)
                    for task, evaluator in experiments
                }

                for future in as_completed(future_to_exp):
                    task, evaluator = future_to_exp[future]
                    try:
                        result = future.result()
                        results[f"{task}_{evaluator}"] = result
                        progress.advance(task_progress)

                        console.print(
                            f"[green]✓[/green] {task} ({evaluator}) completed"
                        )

                    except Exception as e:
                        console.print(f"[red]✗[/red] {task} ({evaluator}) failed: {e}")
                        results[f"{task}_{evaluator}"] = {"error": str(e)}
                        progress.advance(task_progress)

        return results

    def _run_single_experiment(
        self, task: str, evaluator: str, iterations: int, timeout: int
    ) -> Dict:
        """Run a single experiment (task + approach combination)."""

        task_path = self.base_path / "tasks" / task
        config_path = task_path / f"config_{evaluator}.yaml"

        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")

        # Create output directory for this experiment
        exp_output_dir = self.output_dir / f"{evaluator}"
        exp_output_dir.mkdir(parents=True, exist_ok=True)

        # Use a temporary directory for OpenEvolve to prevent nested directory issues
        import tempfile

        temp_output_dir = Path(tempfile.mkdtemp(prefix=f"openevolve_{evaluator}_"))

        if self.verbose:
            console.print(f"[dim]Output directory: {exp_output_dir}[/dim]")
            console.print(f"[dim]Temp directory: {temp_output_dir}[/dim]")

        # Prepare OpenEvolve command for multi-file evolution
        # Initial program: the source code directory (e.g., bayes_opt/)
        # Evaluator: separate evaluator file
        evaluator_file = task_path / "evaluator.py"

        # Load task metadata to get source directory/file
        task_yaml_path = task_path / "task.yaml"
        source_file = None

        if task_yaml_path.exists():
            try:
                import yaml

                with open(task_yaml_path, "r") as f:
                    task_metadata = yaml.safe_load(f)
                if "source_file" in task_metadata:
                    source_file = task_path / task_metadata["source_file"]
            except Exception as e:
                console.print(
                    "[yellow]Warning: Could not load task metadata for "
                    f"{task}: {e}[/yellow]"
                )
                # Use default fallback

        # For multi-file evolution, use openevolve-run.py with --directory/--block-id
        # First, find the available block IDs
        openevolve_run_script = (
            Path(__file__).parent.parent.parent / "openevolve" / "openevolve-run.py"
        )

        # List available block IDs
        list_cmd = [
            "python",
            str(openevolve_run_script),
            "--directory",
            str(task_path),
            "--list-blocks",
        ]

        try:
            t_list_start = time.time()
            list_result = subprocess.run(
                list_cmd, capture_output=True, text=True, timeout=30
            )
            t_list_end = time.time()
            if self.verbose:
                console.print(
                    "[dim]List-blocks finished in "
                    f"{t_list_end - t_list_start:.2f}s[/dim]"
                )
            if list_result.returncode != 0:
                console.print(
                    "[yellow]Warning: Could not list blocks for "
                    f"{task}: {list_result.stderr}[/yellow]"
                )
                # Fallback to single-file mode
                initial_program = (
                    source_file
                    if source_file and source_file.exists()
                    else evaluator_file
                )
                cmd = [
                    "python",
                    "-m",
                    "openevolve.cli",
                    str(initial_program),  # initial_program
                    str(evaluator_file),  # evaluation_file
                    "--config",
                    str(config_path),
                    "--iterations",
                    str(iterations),
                    "--output",
                    str(temp_output_dir),
                ]
            else:
                # Parse block IDs from output
                block_ids = []
                for line in list_result.stdout.split("\n"):
                    if "ID:" in line and "'" in line:
                        # Extract block ID from line like "  ID: 'block-id'"
                        block_id = line.split("'")[1]
                        block_ids.append(block_id)

                if block_ids:
                    # Use the first available block ID
                    block_id = block_ids[0]
                    console.print(f"[blue]Using block ID: {block_id}[/blue]")

                    # Build command based on evaluator mode
                    if evaluator == "handwritten":
                        # Use evaluator.py explicitly
                        cmd = [
                            "python",
                            str(openevolve_run_script),
                            "--directory",
                            str(task_path),
                            "--block-id",
                            block_id,
                            "",  # Empty initial_program (not used in multi-file mode)
                            str(evaluator_file),  # evaluation_file
                            "--config",
                            str(config_path),
                            "--iterations",
                            str(iterations),
                            "--output",
                            str(temp_output_dir),
                        ]
                    else:
                        # LLM-generated and LLM-judge: run in multi-file mode so the
                        # best program is applied to repository files (no
                        # evaluator.py; rely on config's llm_generate_evaluator or
                        # llm_criteria)
                        cmd = [
                            "python",
                            str(openevolve_run_script),
                            "--directory",
                            str(task_path),
                            "--block-id",
                            block_id,
                            "",  # Empty initial_program (not used in multi-file mode)
                            "--config",
                            str(config_path),
                            "--iterations",
                            str(iterations),
                            "--output",
                            str(temp_output_dir),
                        ]

                    # Optional: override log level for openevolve
                    if self.openevolve_log_level:
                        cmd.extend(["--log-level", self.openevolve_log_level])
                else:
                    console.print(
                        f"[yellow]No block IDs found for {task}, falling back "
                        "to single-file mode[/yellow]"
                    )
                    # Fallback to single-file mode
                    cmd = [
                        "python",
                        "-m",
                        "openevolve.cli",
                        str(evaluator_file),  # initial_program (fallback)
                        str(evaluator_file),  # evaluation_file
                        "--config",
                        str(config_path),
                        "--iterations",
                        str(iterations),
                        "--output",
                        str(temp_output_dir),
                    ]
        except Exception as e:
            console.print(
                f"[yellow]Warning: Error listing blocks for {task}: {e}[/yellow]"
            )
            # Fallback to single-file mode
            initial_program = (
                source_file if source_file and source_file.exists() else evaluator_file
            )
            cmd = [
                "python",
                "-m",
                "openevolve.cli",
                str(initial_program),  # initial_program
                str(evaluator_file),  # evaluation_file
                "--config",
                str(config_path),
                "--iterations",
                str(iterations),
                "--output",
                str(temp_output_dir),
            ]

        # Run OpenEvolve from the task directory for multi-file evolution
        start_time = time.time()
        try:
            if self.verbose:
                console.print(f"[dim]Running subprocess: {' '.join(cmd)}[/dim]")
                t_run_start = time.time()
            result = subprocess.run(
                cmd,
                capture_output=not self.verbose,
                text=True,
                timeout=timeout,
                cwd=task_path,  # Run from task directory for multi-file evolution
            )
            if self.verbose:
                t_run_end = time.time()
                console.print(
                    f"[dim]Subprocess finished in {t_run_end - t_run_start:.2f}s "
                    f"(rc={result.returncode})[/dim]"
                )

            elapsed_time = time.time() - start_time

            # Parse results
            experiment_result = {
                "task": task,
                "evaluator": evaluator,
                "iterations": iterations,
                "elapsed_time": elapsed_time,
                "return_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "success": result.returncode == 0,
            }

            # Try to load OpenEvolve results
            try:
                # Look for best_program_info.json in the best/ subdirectory
                best_program_info = exp_output_dir / "best" / "best_program_info.json"
                if best_program_info.exists():
                    with open(best_program_info, "r") as f:
                        best_info = json.load(f)
                    experiment_result["best_metrics"] = best_info.get("metrics", {})
                    experiment_result["best_score"] = best_info.get("metrics", {}).get(
                        "combined_score", 0.0
                    )

                # Load evolution history if available
                logs_dir = exp_output_dir / "logs"
                if logs_dir.exists():
                    log_files = list(logs_dir.glob("*.log"))
                    if log_files:
                        experiment_result["log_file"] = str(log_files[0])

                # Normalize best artifacts for multi-file runs: copy the evolved
                # target file into best/ so users see the actual evolved source
                # (not the temporary entry file)
                try:
                    evolved_root = exp_output_dir / "evolved_files"
                    best_dir = exp_output_dir / "best"
                    if evolved_root.exists():
                        best_dir.mkdir(parents=True, exist_ok=True)

                        # Fall back to the first python file under evolved_files
                        candidates = list(evolved_root.rglob("*.py"))
                        if candidates:
                            from shutil import copyfile

                            # Copy the primary evolved file into best/,
                            # preserving its original layout
                            primary = candidates[0]
                            dest_path = best_dir / primary.name
                            try:
                                copyfile(primary, dest_path)
                                experiment_result["best_source_file"] = str(dest_path)
                            except Exception:
                                pass

                        # Also write a manifest to map evolved files
                        manifest = {
                            "evolved_files": [
                                str(p.relative_to(exp_output_dir)) for p in candidates
                            ]
                        }
                        with open(best_dir / "manifest.json", "w") as mf:
                            json.dump(manifest, mf, indent=2)
                except Exception:
                    # Non-fatal; continue without normalization
                    pass

            except Exception as e:
                console.print(
                    "[yellow]Warning: Could not parse OpenEvolve results for "
                    f"{task}_{evaluator}: {e}[/yellow]"
                )

            # Copy results from temporary directory to final output directory
            try:
                import shutil

                if temp_output_dir.exists():
                    # Copy all contents from temp directory to final output directory
                    for item in temp_output_dir.iterdir():
                        if item.is_dir():
                            shutil.copytree(
                                item, exp_output_dir / item.name, dirs_exist_ok=True
                            )
                        else:
                            shutil.copy2(item, exp_output_dir / item.name)

                    # Clean up temporary directory
                    shutil.rmtree(temp_output_dir)
            except Exception as e:
                console.print(
                    "[yellow]Warning: Could not copy results from temp "
                    f"directory: {e}[/yellow]"
                )

            return experiment_result

        except subprocess.TimeoutExpired:
            # Clean up temporary directory on timeout
            try:
                import shutil

                if temp_output_dir.exists():
                    shutil.rmtree(temp_output_dir)
            except Exception:
                pass
            return {
                "task": task,
                "evaluator": evaluator,
                "iterations": iterations,
                "elapsed_time": timeout,
                "return_code": -1,
                "stdout": "",
                "stderr": f"Timeout after {timeout} seconds",
                "success": False,
                "error": "timeout",
            }
        except Exception as e:
            # Clean up temporary directory on exception
            try:
                import shutil

                if temp_output_dir.exists():
                    shutil.rmtree(temp_output_dir)
            except Exception:
                pass
            return {
                "task": task,
                "evaluator": evaluator,
                "iterations": iterations,
                "elapsed_time": time.time() - start_time,
                "return_code": -1,
                "stdout": "",
                "stderr": str(e),
                "success": False,
                "error": str(e),
            }
