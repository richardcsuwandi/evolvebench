#!/usr/bin/env python3
"""
Benchmark comparison: Initial vs Best Optimized difflib Differ._fancy_replace

This script demonstrates the performance improvement discovered by OpenEvolve's
genetic programming optimization of Python's difflib module.

The optimization eliminates redundant calls to the expensive ratio() function,
resulting in a mean 5.24% speedup on pathological cases (tested across 1400
comparisons with 7 different file sizes).
"""

import time
import sys
import importlib.util
import re
from pathlib import Path


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


def load_module_from_file(filepath, module_name):
    """Load a Python module from a file path."""
    spec = importlib.util.spec_from_file_location(module_name, filepath)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_module_from_evolve_block(filepath, module_name):
    """
    Load a Python module from an EVOLVE-BLOCK formatted file.
    Extracts the code between ### FILE: filename ### and ### END FILE ### markers.
    """
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Extract content between ### FILE: difflib.py ### and ### END FILE ###
    pattern = r'### FILE: difflib\.py ###\s*\n(.*?)\n### END FILE ###'
    match = re.search(pattern, content, re.DOTALL)
    
    if match:
        code = match.group(1)
        # Create a module using spec_from_file_location but compile the extracted code
        spec = importlib.util.spec_from_file_location(module_name, filepath)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        # Execute the extracted code in the module's namespace
        exec(compile(code, str(filepath), 'exec'), module.__dict__)
        return module
    else:
        # Fallback: try loading as regular module
        return load_module_from_file(filepath, module_name)


def benchmark_differ(differ_class, test_name, size, iterations=100):
    """
    Benchmark a Differ implementation on pathological cases.

    Args:
        differ_class: The Differ class to test
        test_name: Name for display
        size: Number of similar lines to compare
        iterations: Number of times to run the test

    Returns:
        Total time in seconds (not average, to show cumulative savings)
    """
    # Pathological case from GitHub issue #119105:
    # Many similar lines that differ by only one character in the middle
    a = ["0123456789\n"] * size
    b = ["01234a56789\n"] * size

    total_time = 0.0
    for _ in range(iterations):
        d = differ_class()
        start = time.perf_counter()
        result = list(d.compare(a, b))
        elapsed = time.perf_counter() - start
        total_time += elapsed

    return total_time  # Return total time in seconds


def main():
    """Run the benchmark comparison."""
    print("=" * 100)
    print("OpenEvolve Difflib Optimization - Performance Comparison")
    print("=" * 100)
    print()
    print("This benchmark compares the original difflib Differ._fancy_replace")
    print("implementation against optimized versions from handwritten and LLM-generated evaluations.")
    print()

    # Load all three modules
    base_dir = Path(__file__).parent
    initial_module = load_module_from_file(
        base_dir / "difflib.py",
        "initial_difflib"
    )
    handwritten_module = load_module_from_evolve_block(
        base_dir / "best_program_handwritten.py",
        "handwritten_difflib"
    )
    llm_generated_module = load_module_from_evolve_block(
        base_dir / "best_program_llm_generated.py",
        "llm_generated_difflib"
    )

    # Comprehensive test across multiple sizes with high iteration count
    test_sizes = [100, 200, 500, 1000, 1500, 2000, 3000]
    iterations_per_test = 200  # High iteration count for statistical accuracy

    print("Pathological Case (GitHub issue #119105):")
    print("Comparing files with many similar lines differing by one character")
    print(f"Running {iterations_per_test} iterations per test size for statistical accuracy")
    print(f"Total comparisons: {len(test_sizes) * iterations_per_test}")
    print()
    print(f"{'Size':>6} │ {'Initial (sec)':>14} │ {'Handwritten (sec)':>18} │ {'LLM-Gen (sec)':>15} │ "
          f"{'HW Speedup':>12} │ {'LLM Speedup':>13}")
    print("─" * 105)

    results = []
    total_time_initial = 0.0
    total_time_handwritten = 0.0
    total_time_llm_generated = 0.0

    for size in test_sizes:
        print(f"Testing size {size}... ", end='', flush=True)

        # Benchmark initial program
        initial_time = benchmark_differ(
            initial_module.Differ,
            "Initial",
            size,
            iterations=iterations_per_test
        )

        # Benchmark handwritten program
        handwritten_time = benchmark_differ(
            handwritten_module.Differ,
            "Handwritten",
            size,
            iterations=iterations_per_test
        )

        # Benchmark LLM-generated program
        llm_generated_time = benchmark_differ(
            llm_generated_module.Differ,
            "LLM-Generated",
            size,
            iterations=iterations_per_test
        )

        hw_speedup = initial_time / handwritten_time
        llm_speedup = initial_time / llm_generated_time
        hw_speedup_pct = (1 - handwritten_time / initial_time) * 100
        llm_speedup_pct = (1 - llm_generated_time / initial_time) * 100

        # Status indicators
        hw_status = "🚀" if hw_speedup > 1.1 else ("✅" if hw_speedup > 1.05 else "➖")
        llm_status = "🚀" if llm_speedup > 1.1 else ("✅" if llm_speedup > 1.05 else "➖")

        print(f"\r{size:>6} │ {initial_time:>13.3f} │ {handwritten_time:>17.3f} │ "
              f"{llm_generated_time:>14.3f} │ {hw_speedup:>6.2f}x {hw_status} ({hw_speedup_pct:+.1f}%) │ "
              f"{llm_speedup:>6.2f}x {llm_status} ({llm_speedup_pct:+.1f}%)")

        results.append({
            'size': size,
            'initial': initial_time,
            'handwritten': handwritten_time,
            'llm_generated': llm_generated_time,
            'hw_speedup': hw_speedup,
            'llm_speedup': llm_speedup
        })

        total_time_initial += initial_time
        total_time_handwritten += handwritten_time
        total_time_llm_generated += llm_generated_time

    print("─" * 105)

    total_hw_speedup = total_time_initial / total_time_handwritten
    total_llm_speedup = total_time_initial / total_time_llm_generated
    total_hw_saved = total_time_initial - total_time_handwritten
    total_llm_saved = total_time_initial - total_time_llm_generated

    print(f"{'TOTAL':>6} │ {total_time_initial:>13.3f} │ {total_time_handwritten:>17.3f} │ "
          f"{total_time_llm_generated:>14.3f} │ {total_hw_speedup:>6.2f}x │ {total_llm_speedup:>6.2f}x")
    print("=" * 105)
    print()

    # Calculate statistics for handwritten
    hw_speedups = [r['hw_speedup'] for r in results]
    hw_mean_speedup = sum(hw_speedups) / len(hw_speedups)
    hw_min_speedup = min(hw_speedups)
    hw_max_speedup = max(hw_speedups)
    hw_variance = sum((s - hw_mean_speedup) ** 2 for s in hw_speedups) / len(hw_speedups)
    hw_std_dev = hw_variance ** 0.5
    hw_mean_pct = (1 - 1/hw_mean_speedup) * 100
    hw_min_pct = (1 - 1/hw_min_speedup) * 100
    hw_max_pct = (1 - 1/hw_max_speedup) * 100

    # Calculate statistics for LLM-generated
    llm_speedups = [r['llm_speedup'] for r in results]
    llm_mean_speedup = sum(llm_speedups) / len(llm_speedups)
    llm_min_speedup = min(llm_speedups)
    llm_max_speedup = max(llm_speedups)
    llm_variance = sum((s - llm_mean_speedup) ** 2 for s in llm_speedups) / len(llm_speedups)
    llm_std_dev = llm_variance ** 0.5
    llm_mean_pct = (1 - 1/llm_mean_speedup) * 100
    llm_min_pct = (1 - 1/llm_min_speedup) * 100
    llm_max_pct = (1 - 1/llm_max_speedup) * 100

    # Summary with statistics
    print("Statistical Summary:")
    print(f"  Total test configurations: {len(test_sizes)} sizes")
    print(f"  Total iterations: {iterations_per_test * len(test_sizes)} comparisons")
    print()
    print("  Handwritten Evaluation Performance:")
    print(f"    Total time saved: {total_hw_saved:.3f} seconds")
    print(f"    Mean speedup:     {hw_mean_speedup:.3f}x ({hw_mean_pct:+.2f}% faster)")
    print(f"    Min speedup:      {hw_min_speedup:.3f}x ({hw_min_pct:+.2f}% faster)")
    print(f"    Max speedup:      {hw_max_speedup:.3f}x ({hw_max_pct:+.2f}% faster)")
    print(f"    Std deviation:    {hw_std_dev:.3f}x")
    print()
    print("  LLM-Generated Evaluation Performance:")
    print(f"    Total time saved: {total_llm_saved:.3f} seconds")
    print(f"    Mean speedup:     {llm_mean_speedup:.3f}x ({llm_mean_pct:+.2f}% faster)")
    print(f"    Min speedup:      {llm_min_speedup:.3f}x ({llm_min_pct:+.2f}% faster)")
    print(f"    Max speedup:      {llm_max_speedup:.3f}x ({llm_max_pct:+.2f}% faster)")
    print(f"    Std deviation:    {llm_std_dev:.3f}x")
    print()
    print(f"  💡 Real-world impact (best optimizer):")
    best_optimizer = "Handwritten" if total_hw_speedup > total_llm_speedup else "LLM-Generated"
    best_saved = max(total_hw_saved, total_llm_saved)
    time_saved_per_op = best_saved / (iterations_per_test * len(test_sizes))
    print(f"     Best optimizer: {best_optimizer}")
    print(f"     Per operation:  {time_saved_per_op * 1000:.3f} ms saved")
    print(f"     Per 1,000 ops:  {time_saved_per_op * 1000:.1f} seconds saved")
    print(f"     Per 10,000 ops: {time_saved_per_op * 10000:.1f} seconds ({time_saved_per_op * 10000 / 60:.2f} minutes)")
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