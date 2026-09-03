#!/usr/bin/env python3
"""
Detailed benchmark comparing pandas rolling rank implementations
"""

import sys
import os
import time
import statistics
import types
import numpy as np
import pandas as pd
from pathlib import Path

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))


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

def extract_code_from_evolve_block(file_path):
    """
    Extract code from EVOLVE-BLOCK format files.
    Returns the code content from the FILE section or EVOLVE-BLOCK section.
    """
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Check if this is a multi-file EVOLVE-BLOCK format file (best_program files)
    if '### FILE:' in content:
        # Extract code between ### FILE: ... ### and ### END FILE ###
        start_marker = '### FILE:'
        end_marker = '### END FILE ###'
        
        start_idx = content.find(start_marker)
        if start_idx == -1:
            return content
        
        # Find the start of actual code (after filename line)
        code_start = content.find('\n', start_idx) + 1
        
        # Find the end marker
        end_idx = content.find(end_marker, code_start)
        if end_idx == -1:
            # If no end marker, take everything after the FILE marker
            return content[code_start:]
        
        return content[code_start:end_idx].strip()
    
    # Check if this file has EVOLVE-BLOCK markers (like pandas_rolling_rank.py)
    if '# EVOLVE-BLOCK-START' in content:
        import re
        # Extract code between EVOLVE-BLOCK-START and EVOLVE-BLOCK-END
        # But we need the full file context, so let's extract just the compute method
        pattern = r'# EVOLVE-BLOCK-START[^\n]*\n(.*?)# EVOLVE-BLOCK-END'
        match = re.search(pattern, content, re.DOTALL)
        if match:
            # For pandas_rolling_rank, we need the full file but can replace the compute method
            # Actually, let's just return the full content since the class needs context
            return content
    
    return content

def load_program(program_path):
    """Load program from file, handling both regular files and EVOLVE-BLOCK format"""
    # Extract code (handles EVOLVE-BLOCK format if present)
    code = extract_code_from_evolve_block(program_path)
    
    # Create namespace with necessary imports
    namespace = {
        '__builtins__': __builtins__,
        'np': np,
        'pd': pd,
        'numpy': np,
        'pandas': pd,
    }
    
    # Add numba if available (for llm_generated version)
    try:
        from numba import njit
        namespace['njit'] = njit
    except ImportError:
        pass
    
    # Execute the code
    exec(code, namespace)
    
    # Return the namespace as a SimpleNamespace
    return types.SimpleNamespace(**namespace)

def test_correctness(program_module):
    """Test correctness of the RollingRank implementation"""
    RollingRank = program_module.RollingRank
    
    correctness_tests = []
    
    # Test 1: Simple ascending sequence
    test_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    window_size = 3
    roller = RollingRank(window_size=window_size, method='average', ascending=True)
    result = roller.compute(test_data)
    expected = pd.Series(test_data).rolling(window_size).rank(method='average').values
    correctness_tests.append({
        'name': 'simple_sequence',
        'passed': np.allclose(result, expected, equal_nan=True),
        'result': result.tolist(),
        'expected': expected.tolist()
    })
    
    # Test 2: With duplicates
    test_data = np.array([1.0, 2.0, 2.0, 3.0, 2.0])
    roller = RollingRank(window_size=3, method='average', ascending=True)
    result = roller.compute(test_data)
    expected = pd.Series(test_data).rolling(3).rank(method='average').values
    correctness_tests.append({
        'name': 'duplicates',
        'passed': np.allclose(result, expected, equal_nan=True),
        'result': result.tolist(),
        'expected': expected.tolist()
    })
    
    # Test 3: With NaN values
    test_data = np.array([1.0, 2.0, 3.0, np.nan, 5.0])
    roller = RollingRank(window_size=3, method='average', ascending=True)
    result = roller.compute(test_data)
    expected = pd.Series(test_data).rolling(3).rank(method='average').values
    correctness_tests.append({
        'name': 'with_nan',
        'passed': np.allclose(result, expected, equal_nan=True),
        'result': result.tolist(),
        'expected': expected.tolist()
    })
    
    # Test 4: Method='min'
    test_data = np.array([3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0])
    roller = RollingRank(window_size=4, method='min', ascending=True)
    result = roller.compute(test_data)
    expected = pd.Series(test_data).rolling(4).rank(method='min').values
    correctness_tests.append({
        'name': 'method_min',
        'passed': np.allclose(result, expected, equal_nan=True),
        'result': result.tolist(),
        'expected': expected.tolist()
    })
    
    # Test 5: Method='max'
    test_data = np.array([3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0])
    roller = RollingRank(window_size=4, method='max', ascending=True)
    result = roller.compute(test_data)
    expected = pd.Series(test_data).rolling(4).rank(method='max').values
    correctness_tests.append({
        'name': 'method_max',
        'passed': np.allclose(result, expected, equal_nan=True),
        'result': result.tolist(),
        'expected': expected.tolist()
    })
    
    # Test 6: Descending order
    test_data = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
    roller = RollingRank(window_size=3, method='average', ascending=False)
    result = roller.compute(test_data)
    expected = pd.Series(test_data).rolling(3).rank(method='average', ascending=False).values
    correctness_tests.append({
        'name': 'descending',
        'passed': np.allclose(result, expected, equal_nan=True),
        'result': result.tolist(),
        'expected': expected.tolist()
    })
    
    # Test 7: Percentile mode
    test_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    roller = RollingRank(window_size=4, method='average', ascending=True, pct=True)
    result = roller.compute(test_data)
    expected = pd.Series(test_data).rolling(4).rank(method='average', pct=True).values
    correctness_tests.append({
        'name': 'percentile',
        'passed': np.allclose(result, expected, equal_nan=True),
        'result': result.tolist(),
        'expected': expected.tolist()
    })
    
    # Test 8: Larger dataset
    np.random.seed(42)
    test_data = np.random.randn(100)
    roller = RollingRank(window_size=10, method='average', ascending=True)
    result = roller.compute(test_data)
    expected = pd.Series(test_data).rolling(10).rank(method='average').values
    correctness_tests.append({
        'name': 'large_random',
        'passed': np.allclose(result, expected, equal_nan=True, rtol=1e-10),
        'max_diff': np.max(np.abs(result - expected)) if not np.all(np.isnan(result)) else 0.0
    })
    
    passed_count = sum(1 for test in correctness_tests if test['passed'])
    num_tests = len(correctness_tests)
    
    return {
        'num_passed': passed_count,
        'num_tests': num_tests,
        'passed': passed_count == num_tests,
        'tests': correctness_tests
    }

def run_performance_benchmark(program_module, n_values, window_size, num_runs=10):
    """Run performance benchmark for specific configuration"""
    RollingRank = program_module.RollingRank
    
    # Warm up (especially important for Numba JIT)
    np.random.seed(42)
    warmup_data = np.random.randn(min(100, n_values))
    roller = RollingRank(window_size=window_size, method='average', ascending=True)
    roller.compute(warmup_data)
    
    times = []
    
    for _ in range(num_runs):
        np.random.seed(42)
        test_data = np.random.randn(n_values)
        
        roller = RollingRank(window_size=window_size, method='average', ascending=True)
        start = time.perf_counter()
        result = roller.compute(test_data)
        elapsed = time.perf_counter() - start
        
        # Verify correctness
        expected = pd.Series(test_data).rolling(window_size).rank(method='average').values
        if np.allclose(result, expected, equal_nan=True, rtol=1e-10):
            times.append(elapsed)
    
    if not times:
        return None
    
    return {
        'mean': statistics.mean(times),
        'stdev': statistics.stdev(times) if len(times) > 1 else 0,
        'min': min(times),
        'max': max(times),
    }

if __name__ == "__main__":
    # Capture output to both terminal and output.txt
    script_dir = Path(__file__).parent
    output_file = script_dir / "output.txt"
    
    # Save original stdout and stderr
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    
    # Create Tee objects to write to both terminal and file
    tee_stdout = Tee(output_file, original_stdout)
    tee_stderr = Tee(output_file, original_stderr)
    
    try:
        sys.stdout = tee_stdout
        sys.stderr = tee_stderr
        
        print("=" * 80)
        print("DETAILED PANDAS ROLLING RANK BENCHMARK - IMPLEMENTATION COMPARISON")
        print("=" * 80)

        # Get the directory where this script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Load programs
        baseline_path = os.path.join(script_dir, 'pandas_rolling_rank.py')
        handwritten_path = os.path.join(script_dir, 'best_program_handwritten.py')
        llm_generated_path = os.path.join(script_dir, 'best_program_llm_generated.py')

        print(f"\n📂 Loading programs:")
        print(f"   Baseline (initial): {baseline_path}")
        print(f"   Handwritten (best): {handwritten_path}")
        print(f"   LLM Generated (best): {llm_generated_path}")

        baseline = load_program(baseline_path)
        handwritten = load_program(handwritten_path)
        llm_generated = load_program(llm_generated_path)

        # Verify correctness first
        print("\n✅ Correctness Tests:")
        baseline_correct = test_correctness(baseline)
        handwritten_correct = test_correctness(handwritten)
        llm_generated_correct = test_correctness(llm_generated)
        
        print(f"   Baseline (initial): {baseline_correct['num_passed']}/{baseline_correct['num_tests']} passed")
        print(f"   Handwritten (best): {handwritten_correct['num_passed']}/{handwritten_correct['num_tests']} passed")
        print(f"   LLM Generated (best): {llm_generated_correct['num_passed']}/{llm_generated_correct['num_tests']} passed")

        test_configs = [
            (1000, 50, "Small"),
            (5000, 50, "Medium"),
            (10000, 50, "Large"),
        ]

        print("\n" + "=" * 80)
        print("PERFORMANCE RESULTS (10 runs per configuration)")
        print("=" * 80)

        for n_values, window_size, label in test_configs:
            print(f"\n📊 {label} Dataset: {n_values} values, window size {window_size}")
            print("-" * 80)

            print(f"   Running baseline...", end="", flush=True)
            baseline_stats = run_performance_benchmark(baseline, n_values, window_size, num_runs=10)
            print(f" done")

            print(f"   Running handwritten...", end="", flush=True)
            handwritten_stats = run_performance_benchmark(handwritten, n_values, window_size, num_runs=10)
            print(f" done")

            print(f"   Running llm_generated...", end="", flush=True)
            llm_generated_stats = run_performance_benchmark(llm_generated, n_values, window_size, num_runs=10)
            print(f" done")

            print(f"\n   Baseline:")
            print(f"      Time: {baseline_stats['mean']:.6f}s ± {baseline_stats['stdev']:.6f}s")
            print(f"      Range: [{baseline_stats['min']:.6f}s - {baseline_stats['max']:.6f}s]")

            print(f"\n   Handwritten:")
            print(f"      Time: {handwritten_stats['mean']:.6f}s ± {handwritten_stats['stdev']:.6f}s")
            print(f"      Range: [{handwritten_stats['min']:.6f}s - {handwritten_stats['max']:.6f}s]")

            print(f"\n   LLM Generated:")
            print(f"      Time: {llm_generated_stats['mean']:.6f}s ± {llm_generated_stats['stdev']:.6f}s")
            print(f"      Range: [{llm_generated_stats['min']:.6f}s - {llm_generated_stats['max']:.6f}s]")

            # Calculate speedups vs baseline
            handwritten_speedup = (baseline_stats['mean'] - handwritten_stats['mean']) / baseline_stats['mean'] * 100
            handwritten_factor = baseline_stats['mean'] / handwritten_stats['mean'] if handwritten_stats['mean'] > 0 else 0
            
            llm_generated_speedup = (baseline_stats['mean'] - llm_generated_stats['mean']) / baseline_stats['mean'] * 100
            llm_generated_factor = baseline_stats['mean'] / llm_generated_stats['mean'] if llm_generated_stats['mean'] > 0 else 0

            print(f"\n   🚀 Speedup vs Baseline:")
            print(f"      Handwritten: {handwritten_speedup:+.2f}% ({handwritten_factor:.2f}x)")
            print(f"      LLM Generated: {llm_generated_speedup:+.2f}% ({llm_generated_factor:.2f}x)")

        print("\n" + "=" * 80)
    finally:
        # Restore original stdout and stderr
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        tee_stdout.close()
        tee_stderr.close()

