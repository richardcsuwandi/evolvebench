#!/usr/bin/env python3
"""
Benchmark comparison: Initial vs Best Optimized PyMOO Non-dominated Sorting

This script compares the initial fast_non_dominated_sort function from non_dominated_sorting.py
with the optimized versions from handwritten and LLM-generated evaluations to demonstrate
the performance improvements discovered by OpenEvolve.
"""

import time
import sys
import os
import shutil
import tempfile
import importlib
import re
import statistics
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from pymoo.util.dominator import Dominator


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


def extract_code_from_evolve_block(filepath):
    """
    Extract code from EVOLVE-BLOCK format files.
    For initial file, removes EVOLVE-BLOCK markers from the function.
    For best program files, extracts code between ### FILE: ... ### and ### END FILE ###.
    """
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Check if this is a best program file with FILE markers
    if '### FILE:' in content:
        # Extract content between ### FILE: pymoo/functions/standard/non_dominated_sorting.py ### and ### END FILE ###
        pattern = r'### FILE: pymoo/functions/standard/non_dominated_sorting\.py ###\s*\n(.*?)\n### END FILE ###'
        match = re.search(pattern, content, re.DOTALL)
        
        if match:
            return match.group(1).strip()
        else:
            # Fallback: return full content
            return content
    
    # Check if this is the initial file with EVOLVE-BLOCK markers
    # Remove the EVOLVE-BLOCK markers from the function
    if '# EVOLVE-BLOCK-START' in content:
        # Remove EVOLVE-BLOCK-START comment
        content = re.sub(r'\s*# EVOLVE-BLOCK-START[^\n]*\n', '\n', content)
        # Remove EVOLVE-BLOCK-END comment
        content = re.sub(r'\s*# EVOLVE-BLOCK-END\n', '\n', content)
        return content
    
    # Fallback: return full content
    return content


def setup_pymoo_module(base_dir, code=None):
    """
    Set up a pymoo module by copying the base directory and optionally
    replacing non_dominated_sorting.py with evolved code.
    
    Args:
        base_dir: Base directory containing the pymoo package
        code: Optional code content to replace pymoo/functions/standard/non_dominated_sorting.py
        
    Returns:
        Temporary directory path for importing
    """
    # Create temporary directory
    temp_dir = tempfile.mkdtemp(prefix='pymoo_bench_')
    
    # Copy the entire pymoo directory
    pymoo_src = base_dir / 'pymoo'
    pymoo_dst = Path(temp_dir) / 'pymoo'
    shutil.copytree(pymoo_src, pymoo_dst)
    
    # Replace non_dominated_sorting.py if code provided
    if code:
        sorting_file = pymoo_dst / 'functions' / 'standard' / 'non_dominated_sorting.py'
        with open(sorting_file, 'w') as f:
            f.write(code)
    
    return temp_dir


def cleanup_pymoo_module(temp_dir):
    """Remove temporary directory and clean up sys.path and modules"""
    if temp_dir in sys.path:
        sys.path.remove(temp_dir)
    
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    
    # Clear pymoo modules from cache
    modules_to_remove = [name for name in list(sys.modules.keys()) 
                        if name.startswith('pymoo')]
    for name in modules_to_remove:
        if name in sys.modules:
            del sys.modules[name]


def test_correctness(func, test_cases):
    """Verify that the function produces correct results for test cases"""
    for F in test_cases:
        try:
            result = func(F)
            # Basic correctness checks
            if not isinstance(result, list):
                return False
            # Check that all points are assigned to exactly one front
            total_points = sum(len(front) for front in result)
            if total_points != F.shape[0]:
                return False
        except Exception as e:
            print(f"Error in correctness test: {e}")
            return False
    return True


def benchmark_function(func, F, iterations: int = 20) -> Dict[str, float]:
    """Benchmark a function and return timing statistics"""
    times = []

    # Warmup
    for _ in range(3):
        func(F)

    # Actual timing
    for _ in range(iterations):
        start = time.perf_counter()
        result = func(F)
        end = time.perf_counter()
        times.append(end - start)

    return {
        'mean': statistics.mean(times),
        'median': statistics.median(times),
        'stdev': statistics.stdev(times) if len(times) > 1 else 0,
        'min': min(times),
        'max': max(times)
    }


def generate_test_data(n_points, n_objectives, seed=42):
    """Generate random test data for performance comparison"""
    np.random.seed(seed)
    return np.random.rand(n_points, n_objectives) * 100


def test_correctness_comparison(func1, func2, F):
    """Test if two functions produce the same result"""
    result1 = func1(F)
    result2 = func2(F)
    
    # Sort each front for comparison
    sorted_result1 = [sorted(front) for front in result1]
    sorted_result2 = [sorted(front) for front in result2]
    
    # Compare results
    if len(sorted_result1) != len(sorted_result2):
        return False
    
    for f1, f2 in zip(sorted_result1, sorted_result2):
        if f1 != f2:
            return False
    
    return True


def create_test_graphs() -> List[Tuple[str, np.ndarray]]:
    """Create various test cases"""
    test_cases = []
    
    # Bi-objective cases (where optimizations should shine)
    test_cases.append(("Bi-objective (50 points)", generate_test_data(50, 2)))
    test_cases.append(("Bi-objective (100 points)", generate_test_data(100, 2)))
    test_cases.append(("Bi-objective (500 points)", generate_test_data(500, 2)))
    test_cases.append(("Bi-objective (1000 points)", generate_test_data(1000, 2)))
    
    # Multi-objective cases
    test_cases.append(("Multi-objective (50 points, 3 obj)", generate_test_data(50, 3)))
    test_cases.append(("Multi-objective (100 points, 3 obj)", generate_test_data(100, 3)))
    
    return test_cases


def main():
    """Main benchmark function"""
    print("🚀 PyMOO Non-dominated Sorting Optimization Performance Test")
    print("=" * 70)
    
    # Get the task directory
    task_dir = Path(__file__).parent
    
    # Paths to files
    initial_file = task_dir / 'pymoo' / 'functions' / 'standard' / 'non_dominated_sorting.py'
    handwritten_file = task_dir / 'best_program_handwritten.py'
    llm_generated_file = task_dir / 'best_program_llm_generated.py'
    
    # Extract code from files
    print("\n📂 Loading programs...")
    initial_code = extract_code_from_evolve_block(initial_file)
    handwritten_code = extract_code_from_evolve_block(handwritten_file)
    llm_generated_code = extract_code_from_evolve_block(llm_generated_file)
    
    # Set up temporary modules
    print("🔧 Setting up temporary modules...")
    initial_temp_dir = setup_pymoo_module(task_dir, initial_code)
    handwritten_temp_dir = setup_pymoo_module(task_dir, handwritten_code)
    llm_generated_temp_dir = setup_pymoo_module(task_dir, llm_generated_code)
    
    try:
        # Load modules
        print("📦 Loading modules...")
        
        # Load initial version
        if initial_temp_dir not in sys.path:
            sys.path.insert(0, initial_temp_dir)
        modules_to_clear = [name for name in list(sys.modules.keys()) if name.startswith('pymoo')]
        for name in modules_to_clear:
            if name in sys.modules:
                del sys.modules[name]
        initial_module = importlib.import_module('pymoo.functions.standard.non_dominated_sorting')
        initial_func = initial_module.fast_non_dominated_sort
        
        # Load handwritten version
        if initial_temp_dir in sys.path:
            sys.path.remove(initial_temp_dir)
        if handwritten_temp_dir not in sys.path:
            sys.path.insert(0, handwritten_temp_dir)
        modules_to_clear = [name for name in list(sys.modules.keys()) if name.startswith('pymoo')]
        for name in modules_to_clear:
            if name in sys.modules:
                del sys.modules[name]
        handwritten_module = importlib.import_module('pymoo.functions.standard.non_dominated_sorting')
        handwritten_func = handwritten_module.fast_non_dominated_sort
        
        # Load LLM-generated version
        if handwritten_temp_dir in sys.path:
            sys.path.remove(handwritten_temp_dir)
        if llm_generated_temp_dir not in sys.path:
            sys.path.insert(0, llm_generated_temp_dir)
        modules_to_clear = [name for name in list(sys.modules.keys()) if name.startswith('pymoo')]
        for name in modules_to_clear:
            if name in sys.modules:
                del sys.modules[name]
        llm_generated_module = importlib.import_module('pymoo.functions.standard.non_dominated_sorting')
        llm_generated_func = llm_generated_module.fast_non_dominated_sort
        
        # Test correctness
        print("\n🔍 Verifying correctness...")
        test_cases = [
            generate_test_data(50, 2),
            generate_test_data(100, 2),
            generate_test_data(50, 3),
        ]
        initial_correct = test_correctness(initial_func, test_cases)
        handwritten_correct = test_correctness(handwritten_func, test_cases)
        llm_generated_correct = test_correctness(llm_generated_func, test_cases)
        
        # Compare results between versions
        all_match = True
        for F in test_cases:
            if not test_correctness_comparison(initial_func, handwritten_func, F):
                all_match = False
                break
            if not test_correctness_comparison(initial_func, llm_generated_func, F):
                all_match = False
                break
        
        print(f"   Initial:         {'✅' if initial_correct else '❌'}")
        print(f"   Handwritten:    {'✅' if handwritten_correct else '❌'}")
        print(f"   LLM-generated:  {'✅' if llm_generated_correct else '❌'}")
        print(f"   Results match:  {'✅' if all_match else '❌'}")
        
        if not (initial_correct and handwritten_correct and llm_generated_correct and all_match):
            print("\n❌ Correctness verification failed!")
            return
        
        # Create test cases
        test_graphs = create_test_graphs()
        
        # Benchmark results storage
        results = []
        
        print("\n📊 Performance Comparison Results:")
        print("=" * 70)
        print(f"{'Test Case':<30} {'Initial (ms)':<15} {'Handwritten (ms)':<18} {'LLM-gen (ms)':<15} {'HW Speedup':<12} {'LLM Speedup':<12}")
        print("-" * 70)
        
        for test_name, F in test_graphs:
            # Benchmark initial
            if initial_temp_dir not in sys.path:
                sys.path.insert(0, initial_temp_dir)
            if handwritten_temp_dir in sys.path:
                sys.path.remove(handwritten_temp_dir)
            if llm_generated_temp_dir in sys.path:
                sys.path.remove(llm_generated_temp_dir)
            modules_to_clear = [name for name in list(sys.modules.keys()) if name.startswith('pymoo')]
            for name in modules_to_clear:
                if name in sys.modules:
                    del sys.modules[name]
            initial_module = importlib.import_module('pymoo.functions.standard.non_dominated_sorting')
            initial_func = initial_module.fast_non_dominated_sort
            initial_stats = benchmark_function(initial_func, F)
            
            # Benchmark handwritten
            if initial_temp_dir in sys.path:
                sys.path.remove(initial_temp_dir)
            if handwritten_temp_dir not in sys.path:
                sys.path.insert(0, handwritten_temp_dir)
            modules_to_clear = [name for name in list(sys.modules.keys()) if name.startswith('pymoo')]
            for name in modules_to_clear:
                if name in sys.modules:
                    del sys.modules[name]
            handwritten_module = importlib.import_module('pymoo.functions.standard.non_dominated_sorting')
            handwritten_func = handwritten_module.fast_non_dominated_sort
            handwritten_stats = benchmark_function(handwritten_func, F)
            
            # Benchmark LLM-generated
            if handwritten_temp_dir in sys.path:
                sys.path.remove(handwritten_temp_dir)
            if llm_generated_temp_dir not in sys.path:
                sys.path.insert(0, llm_generated_temp_dir)
            modules_to_clear = [name for name in list(sys.modules.keys()) if name.startswith('pymoo')]
            for name in modules_to_clear:
                if name in sys.modules:
                    del sys.modules[name]
            llm_generated_module = importlib.import_module('pymoo.functions.standard.non_dominated_sorting')
            llm_generated_func = llm_generated_module.fast_non_dominated_sort
            llm_generated_stats = benchmark_function(llm_generated_func, F)
            
            # Calculate speedups
            handwritten_speedup = initial_stats['mean'] / handwritten_stats['mean'] if handwritten_stats['mean'] > 0 else float('inf')
            llm_generated_speedup = initial_stats['mean'] / llm_generated_stats['mean'] if llm_generated_stats['mean'] > 0 else float('inf')
            
            # Store results
            results.append({
                'test_name': test_name,
                'initial': initial_stats,
                'handwritten': handwritten_stats,
                'llm_generated': llm_generated_stats,
                'handwritten_speedup': handwritten_speedup,
                'llm_generated_speedup': llm_generated_speedup
            })
            
            # Display results
            print(f"{test_name:<30} {initial_stats['mean']*1000:<15.3f} {handwritten_stats['mean']*1000:<18.3f} {llm_generated_stats['mean']*1000:<15.3f} {handwritten_speedup:<12.2f} {llm_generated_speedup:<12.2f}")
        
        # Summary
        avg_handwritten_speedup = statistics.mean([r['handwritten_speedup'] for r in results])
        avg_llm_generated_speedup = statistics.mean([r['llm_generated_speedup'] for r in results])
        avg_handwritten_improvement = (avg_handwritten_speedup - 1) * 100
        avg_llm_generated_improvement = (avg_llm_generated_speedup - 1) * 100
        
        print("\n" + "=" * 70)
        print("📋 SUMMARY")
        print("=" * 70)
        print(f"Average speedup (Handwritten):    {avg_handwritten_speedup:.2f}x ({avg_handwritten_improvement:+.1f}%)")
        print(f"Average speedup (LLM-generated): {avg_llm_generated_speedup:.2f}x ({avg_llm_generated_improvement:+.1f}%)")
        
    finally:
        # Cleanup
        print("\n🧹 Cleaning up temporary files...")
        cleanup_pymoo_module(initial_temp_dir)
        cleanup_pymoo_module(handwritten_temp_dir)
        cleanup_pymoo_module(llm_generated_temp_dir)


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
