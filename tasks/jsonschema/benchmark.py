#!/usr/bin/env python3
"""
Benchmark comparison: Initial vs Best Optimized jsonschema equality functions

This script demonstrates the performance improvement discovered by OpenEvolve's
genetic programming optimization of Python's jsonschema equality checking.

The optimization improves type-aware dispatch and reduces isinstance() overhead,
addressing Python 3.12's performance regression where isinstance() calls became
51% slower, causing 33% overall slowdown in equality checking.
"""

import time
import sys
import importlib.util
import re
import copy
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
    
    # Extract content between ### FILE: jsonschema.py ### and ### END FILE ###
    pattern = r'### FILE: jsonschema\.py ###\s*\n(.*?)\n### END FILE ###'
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


def create_nested_structure(depth, width):
    """Create a nested dictionary structure for testing."""
    if depth == 0:
        return {f"key_{i}": i for i in range(width)}
    
    result = {}
    for i in range(width):
        result[f"level_{i}"] = create_nested_structure(depth - 1, width)
    return result


def create_mixed_structure(size):
    """Create a mixed structure with dicts, lists, and primitives."""
    structure = {
        "metadata": {
            "version": "1.0",
            "author": "test",
            "tags": ["tag1", "tag2", "tag3"]
        },
        "data": list(range(size)),
        "nested": {}
    }
    
    for i in range(size // 5):
        structure["nested"][f"key_{i}"] = {
            "value": i,
            "list": [i, i+1, i+2],
            "dict": {f"inner_{j}": j for j in range(3)}
        }
    
    return structure


def benchmark_equality(equal_func, test_name, test_data, iterations=100):
    """
    Benchmark an equality function on test data.

    Args:
        equal_func: The equal function to test
        test_name: Name for display
        test_data: The data structure to compare
        iterations: Number of times to run the test

    Returns:
        Total time in seconds (not average, to show cumulative savings)
    """
    # Create a deep copy for comparison
    data_copy = copy.deepcopy(test_data)

    # Warmup
    for _ in range(10):
        equal_func(test_data, data_copy)

    total_time = 0.0
    for _ in range(iterations):
        start = time.perf_counter()
        result = equal_func(test_data, data_copy)
        elapsed = time.perf_counter() - start
        total_time += elapsed

    return total_time  # Return total time in seconds


def main():
    """Run the benchmark comparison."""
    print("=" * 100)
    print("OpenEvolve JSONSchema Optimization - Performance Comparison")
    print("=" * 100)
    print()
    print("This benchmark compares the original jsonschema equality checking")
    print("implementation against optimized versions from handwritten and LLM-generated evaluations.")
    print()
    print("Issue #1304: Python 3.12 shows 33% performance degradation compared to 3.10")
    print("due to slower isinstance() calls (51% slower) affecting equality checking.")
    print()

    # Load all three modules
    base_dir = Path(__file__).parent
    initial_module = load_module_from_file(
        base_dir / "jsonschema.py",
        "initial_jsonschema"
    )
    handwritten_module = load_module_from_evolve_block(
        base_dir / "best_program_handwritten.py",
        "handwritten_jsonschema"
    )
    llm_generated_module = load_module_from_evolve_block(
        base_dir / "best_program_llm_generated.py",
        "llm_generated_jsonschema"
    )

    # Test configurations matching the evaluator
    test_configs = [
        {
            'name': 'Small nested (depth=2, width=3)',
            'data': create_nested_structure(2, 3),
            'iterations': 1000
        },
        {
            'name': 'Medium nested (depth=3, width=5)',
            'data': create_nested_structure(3, 5),
            'iterations': 500
        },
        {
            'name': 'Large nested (depth=4, width=4)',
            'data': create_nested_structure(4, 4),
            'iterations': 200
        },
        {
            'name': 'Small mixed (size=10)',
            'data': create_mixed_structure(10),
            'iterations': 1000
        },
        {
            'name': 'Medium mixed (size=20)',
            'data': create_mixed_structure(20),
            'iterations': 500
        },
        {
            'name': 'Large mixed (size=50)',
            'data': create_mixed_structure(50),
            'iterations': 200
        },
        {
            'name': 'Very large mixed (size=100)',
            'data': create_mixed_structure(100),
            'iterations': 100
        },
        {
            'name': 'Flat dict (100 keys)',
            'data': {f'key_{i}': i for i in range(100)},
            'iterations': 1000
        },
        {
            'name': 'Flat dict (500 keys)',
            'data': {f'key_{i}': i for i in range(500)},
            'iterations': 200
        },
    ]

    print("Test Configurations:")
    print("Testing equality checking on various nested and mixed structures")
    print(f"Total test configurations: {len(test_configs)}")
    print()
    print(f"{'Test Case':<35} │ {'Initial (sec)':>14} │ {'Handwritten (sec)':>18} │ {'LLM-Gen (sec)':>15} │ "
          f"{'HW Speedup':>12} │ {'LLM Speedup':>13}")
    print("─" * 110)

    results = []
    total_time_initial = 0.0
    total_time_handwritten = 0.0
    total_time_llm_generated = 0.0

    for config in test_configs:
        print(f"Testing {config['name']}... ", end='', flush=True)

        # Benchmark initial program
        initial_time = benchmark_equality(
            initial_module.equal,
            "Initial",
            config['data'],
            iterations=config['iterations']
        )

        # Benchmark handwritten program
        handwritten_time = benchmark_equality(
            handwritten_module.equal,
            "Handwritten",
            config['data'],
            iterations=config['iterations']
        )

        # Benchmark LLM-generated program
        llm_generated_time = benchmark_equality(
            llm_generated_module.equal,
            "LLM-Generated",
            config['data'],
            iterations=config['iterations']
        )

        hw_speedup = initial_time / handwritten_time if handwritten_time > 0 else 0
        llm_speedup = initial_time / llm_generated_time if llm_generated_time > 0 else 0
        hw_speedup_pct = (1 - handwritten_time / initial_time) * 100 if initial_time > 0 else 0
        llm_speedup_pct = (1 - llm_generated_time / initial_time) * 100 if initial_time > 0 else 0

        # Status indicators
        hw_status = "🚀" if hw_speedup > 1.1 else ("✅" if hw_speedup > 1.05 else "➖")
        llm_status = "🚀" if llm_speedup > 1.1 else ("✅" if llm_speedup > 1.05 else "➖")

        print(f"\r{config['name']:<35} │ {initial_time:>13.3f} │ {handwritten_time:>17.3f} │ "
              f"{llm_generated_time:>14.3f} │ {hw_speedup:>6.2f}x {hw_status} ({hw_speedup_pct:+.1f}%) │ "
              f"{llm_speedup:>6.2f}x {llm_status} ({llm_speedup_pct:+.1f}%)")

        results.append({
            'name': config['name'],
            'initial': initial_time,
            'handwritten': handwritten_time,
            'llm_generated': llm_generated_time,
            'hw_speedup': hw_speedup,
            'llm_speedup': llm_speedup,
            'iterations': config['iterations']
        })

        total_time_initial += initial_time
        total_time_handwritten += handwritten_time
        total_time_llm_generated += llm_generated_time

    print("─" * 110)

    total_hw_speedup = total_time_initial / total_time_handwritten if total_time_handwritten > 0 else 0
    total_llm_speedup = total_time_initial / total_time_llm_generated if total_time_llm_generated > 0 else 0
    total_hw_saved = total_time_initial - total_time_handwritten
    total_llm_saved = total_time_initial - total_time_llm_generated

    print(f"{'TOTAL':<35} │ {total_time_initial:>13.3f} │ {total_time_handwritten:>17.3f} │ "
          f"{total_time_llm_generated:>14.3f} │ {total_hw_speedup:>6.2f}x │ {total_llm_speedup:>6.2f}x")
    print("=" * 110)
    print()

    # Calculate statistics for handwritten
    hw_speedups = [r['hw_speedup'] for r in results]
    hw_mean_speedup = sum(hw_speedups) / len(hw_speedups) if hw_speedups else 0
    hw_min_speedup = min(hw_speedups) if hw_speedups else 0
    hw_max_speedup = max(hw_speedups) if hw_speedups else 0
    hw_variance = sum((s - hw_mean_speedup) ** 2 for s in hw_speedups) / len(hw_speedups) if hw_speedups else 0
    hw_std_dev = hw_variance ** 0.5
    hw_mean_pct = (1 - 1/hw_mean_speedup) * 100 if hw_mean_speedup > 0 else 0
    hw_min_pct = (1 - 1/hw_min_speedup) * 100 if hw_min_speedup > 0 else 0
    hw_max_pct = (1 - 1/hw_max_speedup) * 100 if hw_max_speedup > 0 else 0

    # Calculate statistics for LLM-generated
    llm_speedups = [r['llm_speedup'] for r in results]
    llm_mean_speedup = sum(llm_speedups) / len(llm_speedups) if llm_speedups else 0
    llm_min_speedup = min(llm_speedups) if llm_speedups else 0
    llm_max_speedup = max(llm_speedups) if llm_speedups else 0
    llm_variance = sum((s - llm_mean_speedup) ** 2 for s in llm_speedups) / len(llm_speedups) if llm_speedups else 0
    llm_std_dev = llm_variance ** 0.5
    llm_mean_pct = (1 - 1/llm_mean_speedup) * 100 if llm_mean_speedup > 0 else 0
    llm_min_pct = (1 - 1/llm_min_speedup) * 100 if llm_min_speedup > 0 else 0
    llm_max_pct = (1 - 1/llm_max_speedup) * 100 if llm_max_speedup > 0 else 0

    # Calculate total iterations
    total_iterations = sum(r['iterations'] for r in results)

    # Summary with statistics
    print("Statistical Summary:")
    print(f"  Total test configurations: {len(test_configs)}")
    print(f"  Total iterations: {total_iterations} comparisons")
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
    time_saved_per_op = best_saved / total_iterations if total_iterations > 0 else 0
    print(f"     Best optimizer: {best_optimizer}")
    print(f"     Per operation:  {time_saved_per_op * 1000:.3f} ms saved")
    print(f"     Per 1,000 ops:  {time_saved_per_op * 1000:.1f} seconds saved")
    print(f"     Per 10,000 ops: {time_saved_per_op * 10000:.1f} seconds ({time_saved_per_op * 10000 / 60:.2f} minutes)")
    print()
    print("  📊 Context:")
    print("     Original issue: Python 3.12 showed 33% slowdown vs 3.10")
    print("     Root cause: isinstance() calls became 51% slower")
    print("     Impact: Equality checking called 56.4M times during 16MB SBOM validation")
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

