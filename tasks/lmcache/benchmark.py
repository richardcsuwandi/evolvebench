#!/usr/bin/env python3
"""
Detailed benchmark breaking down performance by cache size
"""

import sys
import os
import time
import statistics
import types
from pathlib import Path

# Add current directory to path to import evaluator
sys.path.insert(0, os.path.dirname(__file__))
from evaluator import test_correctness_basic, test_performance_cache_operations


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
    Returns the code content from the FILE section.
    """
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Check if this is an EVOLVE-BLOCK format file
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
    
    return content

def load_program(program_path):
    """Load program from file, handling both regular files and EVOLVE-BLOCK format"""
    # Extract code (handles EVOLVE-BLOCK format if present)
    code = extract_code_from_evolve_block(program_path)
    
    # Create namespace with necessary imports
    future_annotations_code = "from __future__ import annotations\n"
    
    namespace = {
        '__builtins__': __builtins__,
    }
    
    # Add common imports that might be needed
    try:
        from typing import Any, Optional, Dict, List, Set, Tuple
        namespace.update({
            'Any': Any,
            'Optional': Optional,
            'Dict': Dict,
            'List': List,
            'Set': Set,
            'Tuple': Tuple,
        })
    except ImportError:
        pass
    
    try:
        from sortedcontainers import SortedDict
        namespace['SortedDict'] = SortedDict
    except ImportError:
        pass
    
    try:
        from collections import defaultdict, OrderedDict
        namespace['defaultdict'] = defaultdict
        namespace['OrderedDict'] = OrderedDict
    except ImportError:
        pass
    
    # Prepend future annotations if not already present
    if 'from __future__ import annotations' not in code:
        code_to_exec = future_annotations_code + code
    else:
        code_to_exec = code
    
    exec(code_to_exec, namespace)
    return types.SimpleNamespace(**namespace)

def run_size_benchmark(program_module, cache_size, num_ops, num_runs=10):
    """Run benchmark for specific cache size"""
    times = []
    for _ in range(num_runs):
        result = test_performance_cache_operations(program_module, cache_size, num_ops)
        if result['success']:
            times.append(result['elapsed_time'])

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
        print("DETAILED LMCACHE LFU BENCHMARK - BY CACHE SIZE")
        print("=" * 80)

        # Get the directory where this script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Load programs
        baseline_path = os.path.join(script_dir, 'lmcache.py')
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
        baseline_correct = test_correctness_basic(baseline)
        handwritten_correct = test_correctness_basic(handwritten)
        llm_generated_correct = test_correctness_basic(llm_generated)
        
        print(f"   Baseline (initial): {baseline_correct['num_passed']}/{baseline_correct['num_tests']} passed")
        print(f"   Handwritten (best): {handwritten_correct['num_passed']}/{handwritten_correct['num_tests']} passed")
        print(f"   LLM Generated (best): {llm_generated_correct['num_passed']}/{llm_generated_correct['num_tests']} passed")

        test_configs = [
            (100, 1000, "Small"),
            (1000, 10000, "Medium"),
            (10000, 100000, "Large"),
        ]

        print("\n" + "=" * 80)
        print("PERFORMANCE RESULTS (10 runs per configuration)")
        print("=" * 80)

        for cache_size, num_ops, label in test_configs:
            print(f"\n📊 {label} Cache: {cache_size} entries, {num_ops} operations")
            print("-" * 80)

            print(f"   Running baseline...", end="", flush=True)
            baseline_stats = run_size_benchmark(baseline, cache_size, num_ops, num_runs=10)
            print(f" done")

            print(f"   Running handwritten...", end="", flush=True)
            handwritten_stats = run_size_benchmark(handwritten, cache_size, num_ops, num_runs=10)
            print(f" done")

            print(f"   Running llm_generated...", end="", flush=True)
            llm_generated_stats = run_size_benchmark(llm_generated, cache_size, num_ops, num_runs=10)
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

            # Calculate speedups
            handwritten_speedup = (baseline_stats['mean'] - handwritten_stats['mean']) / baseline_stats['mean'] * 100
            handwritten_abs_diff = baseline_stats['mean'] - handwritten_stats['mean']
            
            llm_generated_speedup = (baseline_stats['mean'] - llm_generated_stats['mean']) / baseline_stats['mean'] * 100
            llm_generated_abs_diff = baseline_stats['mean'] - llm_generated_stats['mean']

            print(f"\n   🚀 Speedup vs Baseline:")
            print(f"      Handwritten: {handwritten_speedup:+.2f}% (absolute: {handwritten_abs_diff:+.6f}s)")
            print(f"      LLM Generated: {llm_generated_speedup:+.2f}% (absolute: {llm_generated_abs_diff:+.6f}s)")

        print("\n" + "=" * 80)
    finally:
        # Restore original stdout and stderr
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        tee_stdout.close()
        tee_stderr.close()
