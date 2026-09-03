#!/usr/bin/env python3
"""
Benchmark comparison: Initial vs Best Optimized Python Pathfinding Heap Operations

This script demonstrates the performance improvement discovered by OpenEvolve's
genetic programming optimization of python-pathfinding's heap operations.

The optimization improves heap operations in the A* pathfinding algorithm,
resulting in improved pathfinding performance across various grid sizes.
"""

import time
import sys
import os
import shutil
import tempfile
import importlib
import re
from pathlib import Path
import numpy as np


class Tee:
    """A class that writes to both stdout/stderr and a file."""
    def __init__(self, file_path, stream):
        self.file = open(file_path, 'w', encoding='utf-8')
        self.stream = stream
    
    def write(self, data):
        self.stream.write(data)
        self.file.write(data)
        self.file.flush()
    
    def flush(self):
        self.stream.flush()
        self.file.flush()
    
    def close(self):
        self.file.close()


def extract_files_from_evolve_block(filepath):
    """
    Extract multiple files from an EVOLVE-BLOCK formatted file.
    Returns a dict mapping file paths to their code content.
    """
    with open(filepath, 'r') as f:
        content = f.read()
    
    files = {}
    
    # Pattern to match ### FILE: path ### ... ### END FILE ###
    pattern = r'### FILE: (.*?) ###\s*\n(.*?)\n### END FILE ###'
    matches = re.finditer(pattern, content, re.DOTALL)
    
    for match in matches:
        file_path = match.group(1).strip()
        file_code = match.group(2).strip()
        files[file_path] = file_code
    
    return files


def setup_pathfinding_module(base_dir, files_dict=None):
    """
    Set up a pathfinding module by copying the base directory and optionally
    replacing files with evolved versions.
    
    Args:
        base_dir: Base directory containing the pathfinding package
        files_dict: Optional dict mapping file paths to code content to replace
        
    Returns:
        Temporary directory path and module name for importing
    """
    # Create temporary directory
    temp_dir = tempfile.mkdtemp(prefix='pathfinding_bench_')
    
    # Copy the entire pathfinding directory
    pathfinding_src = base_dir / 'pathfinding'
    pathfinding_dst = Path(temp_dir) / 'pathfinding'
    shutil.copytree(pathfinding_src, pathfinding_dst)
    
    # Replace files if provided
    if files_dict:
        for rel_path, code in files_dict.items():
            # Handle paths like 'pathfinding/core/heap.py'
            file_path = Path(temp_dir) / rel_path
            if file_path.exists():
                with open(file_path, 'w') as f:
                    f.write(code)
    
    # Add temp directory to Python path
    if temp_dir not in sys.path:
        sys.path.insert(0, temp_dir)
    
    return temp_dir


def cleanup_pathfinding_module(temp_dir):
    """Remove temporary directory and clean up sys.path"""
    if temp_dir in sys.path:
        sys.path.remove(temp_dir)
    
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    
    # Reload modules to clear cached imports
    modules_to_remove = [name for name in list(sys.modules.keys()) 
                        if name.startswith('pathfinding')]
    for name in modules_to_remove:
        if name in sys.modules:
            del sys.modules[name]


def create_random_grid(size=50, obstacle_ratio=0.3):
    """Create a random grid with obstacles for testing"""
    from pathfinding.core.grid import Grid
    matrix = np.random.random((size, size)) > obstacle_ratio
    return Grid(matrix=matrix)


def find_random_path(grid, finder_class):
    """Find a path between random start and end points"""
    size = grid.width
    
    # Find walkable nodes
    walkable_nodes = []
    for x in range(size):
        for y in range(size):
            if grid.walkable(x, y):
                walkable_nodes.append((x, y))
    
    if len(walkable_nodes) < 2:
        return None, 0.0
    
    # Pick random start and end
    start_idx = np.random.randint(0, len(walkable_nodes))
    end_idx = np.random.randint(0, len(walkable_nodes))
    while end_idx == start_idx:
        end_idx = np.random.randint(0, len(walkable_nodes))
    
    start = grid.node(*walkable_nodes[start_idx])
    end = grid.node(*walkable_nodes[end_idx])
    
    # Find path and measure time
    finder = finder_class()
    start_time = time.perf_counter()
    path, runs = finder.find_path(start, end, grid)
    elapsed = time.perf_counter() - start_time
    
    return path, elapsed


def benchmark_pathfinding_config(temp_dir, size, obstacle_ratio, num_trials, iterations=10):
    """
    Benchmark pathfinding performance for a specific configuration.
    
    Args:
        temp_dir: Temporary directory containing pathfinding package
        size: Grid size
        obstacle_ratio: Ratio of obstacles
        num_trials: Number of trials per iteration
        iterations: Number of iterations
        
    Returns:
        Dict with total_time, success_count, total_paths, avg_time
    """
    # Import the finder class from the temp directory
    if temp_dir not in sys.path:
        sys.path.insert(0, temp_dir)
    
    # Reload modules to ensure we're using the right version
    modules_to_reload = [name for name in list(sys.modules.keys()) 
                        if name.startswith('pathfinding')]
    for name in modules_to_reload:
        if name in sys.modules:
            try:
                importlib.reload(sys.modules[name])
            except Exception:
                pass
    
    from pathfinding.finder.a_star import AStarFinder
    
    total_time = 0.0
    success_count = 0
    total_paths = 0
    
    for _ in range(num_trials * iterations):
        grid = create_random_grid(size=size, obstacle_ratio=obstacle_ratio)
        path, elapsed = find_random_path(grid, AStarFinder)
        
        if path is not None and len(path) > 0:
            success_count += 1
            total_time += elapsed
        total_paths += 1
    
    return {
        'total_time': total_time,
        'success_count': success_count,
        'total_paths': total_paths,
        'avg_time': total_time / success_count if success_count > 0 else float('inf')
    }


def main():
    """Run the benchmark comparison."""
    print("=" * 100)
    print("OpenEvolve Python Pathfinding Heap Optimization - Performance Comparison")
    print("=" * 100)
    print()
    print("This benchmark compares the original python-pathfinding heap operations")
    print("implementation against optimized versions from handwritten and LLM-generated evaluations.")
    print()

    base_dir = Path(__file__).parent
    
    # Extract files from best programs
    handwritten_files = extract_files_from_evolve_block(
        base_dir / "best_program_handwritten.py"
    )
    llm_generated_files = extract_files_from_evolve_block(
        base_dir / "best_program_llm_generated.py"
    )

    # Test configurations: (size, obstacle_ratio, num_trials)
    test_configs = [
        (20, 0.2, 3),   # Small, sparse
        (20, 0.4, 3),   # Small, dense
        (40, 0.3, 5),   # Medium
        (60, 0.3, 3),   # Large
    ]
    iterations_per_config = 10

    print("Test Case: A* pathfinding on random grids")
    print("Comparing performance across different grid sizes and obstacle densities")
    print(f"Running {iterations_per_config} iterations per configuration")
    total_tests = sum(trials for _, _, trials in test_configs) * iterations_per_config
    print(f"Total pathfinding operations: {total_tests}")
    print()
    print(f"{'Config':>12} │ {'Initial (sec)':>14} │ {'Handwritten (sec)':>18} │ {'LLM-Gen (sec)':>15} │ "
          f"{'HW Speedup':>12} │ {'LLM Speedup':>13}")
    print("─" * 105)

    results = []
    total_time_initial = 0.0
    total_time_handwritten = 0.0
    total_time_llm_generated = 0.0
    total_success_initial = 0
    total_success_handwritten = 0
    total_success_llm_generated = 0

    # Setup initial module (use base directory directly)
    initial_temp_dir = setup_pathfinding_module(base_dir)
    
    # Setup handwritten module
    handwritten_temp_dir = setup_pathfinding_module(base_dir, handwritten_files)
    
    # Setup LLM-generated module
    llm_generated_temp_dir = setup_pathfinding_module(base_dir, llm_generated_files)
    
    try:
        # Benchmark each configuration separately
        for size, obstacle_ratio, num_trials in test_configs:
            config_label = f"{size}x{size} ({obstacle_ratio:.1f})"
            print(f"Testing {config_label}... ", end='', flush=True)
            
            # Benchmark initial program
            initial_result = benchmark_pathfinding_config(
                initial_temp_dir,
                size,
                obstacle_ratio,
                num_trials,
                iterations=iterations_per_config
            )
            
            # Benchmark handwritten program
            handwritten_result = benchmark_pathfinding_config(
                handwritten_temp_dir,
                size,
                obstacle_ratio,
                num_trials,
                iterations=iterations_per_config
            )
            
            # Benchmark LLM-generated program
            llm_generated_result = benchmark_pathfinding_config(
                llm_generated_temp_dir,
                size,
                obstacle_ratio,
                num_trials,
                iterations=iterations_per_config
            )
            
            initial_time = initial_result['total_time']
            handwritten_time = handwritten_result['total_time']
            llm_generated_time = llm_generated_result['total_time']
            
            total_time_initial += initial_time
            total_time_handwritten += handwritten_time
            total_time_llm_generated += llm_generated_time
            total_success_initial += initial_result['success_count']
            total_success_handwritten += handwritten_result['success_count']
            total_success_llm_generated += llm_generated_result['success_count']
            
            hw_speedup = initial_time / handwritten_time if handwritten_time > 0 else 0
            llm_speedup = initial_time / llm_generated_time if llm_generated_time > 0 else 0
            hw_speedup_pct = (1 - handwritten_time / initial_time) * 100 if initial_time > 0 else 0
            llm_speedup_pct = (1 - llm_generated_time / initial_time) * 100 if initial_time > 0 else 0
            
            # Status indicators
            hw_status = "🚀" if hw_speedup > 1.1 else ("✅" if hw_speedup > 1.05 else "➖")
            llm_status = "🚀" if llm_speedup > 1.1 else ("✅" if llm_speedup > 1.05 else "➖")
            
            print(f"\r{config_label:>12} │ {initial_time:>13.3f} │ {handwritten_time:>17.3f} │ "
                  f"{llm_generated_time:>14.3f} │ {hw_speedup:>6.2f}x {hw_status} ({hw_speedup_pct:+.1f}%) │ "
                  f"{llm_speedup:>6.2f}x {llm_status} ({llm_speedup_pct:+.1f}%)")
            
            results.append({
                'config': config_label,
                'initial': initial_time,
                'handwritten': handwritten_time,
                'llm_generated': llm_generated_time,
                'hw_speedup': hw_speedup,
                'llm_speedup': llm_speedup
            })
    
    finally:
        cleanup_pathfinding_module(initial_temp_dir)
        cleanup_pathfinding_module(handwritten_temp_dir)
        cleanup_pathfinding_module(llm_generated_temp_dir)

    print("─" * 105)

    total_hw_speedup = total_time_initial / total_time_handwritten if total_time_handwritten > 0 else 0
    total_llm_speedup = total_time_initial / total_time_llm_generated if total_time_llm_generated > 0 else 0
    total_hw_saved = total_time_initial - total_time_handwritten
    total_llm_saved = total_time_initial - total_time_llm_generated

    print(f"{'TOTAL':>12} │ {total_time_initial:>13.3f} │ {total_time_handwritten:>17.3f} │ "
          f"{total_time_llm_generated:>14.3f} │ {total_hw_speedup:>6.2f}x │ {total_llm_speedup:>6.2f}x")
    print("=" * 105)
    print()

    # Calculate statistics
    print("Statistical Summary:")
    print(f"  Total test configurations: {len(test_configs)}")
    print(f"  Total pathfinding operations: {total_tests}")
    print()
    print("  Correctness (successful paths):")
    print(f"    Initial:        {total_success_initial}/{total_tests} ({total_success_initial/total_tests*100:.1f}%)")
    print(f"    Handwritten:    {total_success_handwritten}/{total_tests} ({total_success_handwritten/total_tests*100:.1f}%)")
    print(f"    LLM-Generated:  {total_success_llm_generated}/{total_tests} ({total_success_llm_generated/total_tests*100:.1f}%)")
    print()
    # Calculate average times
    avg_time_initial = total_time_initial / total_success_initial if total_success_initial > 0 else float('inf')
    avg_time_handwritten = total_time_handwritten / total_success_handwritten if total_success_handwritten > 0 else float('inf')
    avg_time_llm_generated = total_time_llm_generated / total_success_llm_generated if total_success_llm_generated > 0 else float('inf')
    
    print("  Handwritten Evaluation Performance:")
    print(f"    Total time saved: {total_hw_saved:.3f} seconds")
    if total_hw_speedup > 0:
        hw_speedup_pct = (1 - total_time_handwritten / total_time_initial) * 100
        print(f"    Overall speedup:  {total_hw_speedup:.3f}x ({hw_speedup_pct:+.2f}% faster)")
        print(f"    Avg time/path:    {avg_time_handwritten*1000:.3f} ms")
    else:
        print(f"    No speedup detected")
    print()
    print("  LLM-Generated Evaluation Performance:")
    print(f"    Total time saved: {total_llm_saved:.3f} seconds")
    if total_llm_speedup > 0:
        llm_speedup_pct = (1 - total_time_llm_generated / total_time_initial) * 100
        print(f"    Overall speedup:  {total_llm_speedup:.3f}x ({llm_speedup_pct:+.2f}% faster)")
        print(f"    Avg time/path:    {avg_time_llm_generated*1000:.3f} ms")
    else:
        print(f"    No speedup detected")
    print()
    print(f"  💡 Real-world impact (best optimizer):")
    if total_hw_speedup > 0 or total_llm_speedup > 0:
        best_optimizer = "Handwritten" if total_hw_speedup > total_llm_speedup else "LLM-Generated"
        best_saved = max(total_hw_saved, total_llm_saved)
        time_saved_per_path = best_saved / total_tests if total_tests > 0 else 0
        print(f"     Best optimizer: {best_optimizer}")
        print(f"     Per path:       {time_saved_per_path * 1000:.3f} ms saved")
        print(f"     Per 1,000 paths: {time_saved_per_path * 1000:.1f} seconds saved")
        print(f"     Per 10,000 paths: {time_saved_per_path * 10000:.1f} seconds ({time_saved_per_path * 10000 / 60:.2f} minutes)")
    else:
        print(f"     No significant optimization detected")
    print()

if __name__ == "__main__":
    # Capture output to both terminal and output.txt
    base_dir = Path(__file__).parent
    output_file = base_dir / "output.txt"
    
    # Save original stdout and stderr
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    
    # Create Tee objects to write to both terminal and file
    tee_stdout = Tee(output_file, original_stdout)
    tee_stderr = Tee(output_file, original_stderr)
    
    try:
        sys.stdout = tee_stdout
        sys.stderr = tee_stderr
        main()
    finally:
        # Restore original stdout and stderr
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        tee_stdout.close()
        tee_stderr.close()

