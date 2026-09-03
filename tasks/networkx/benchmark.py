#!/usr/bin/env python3
"""
Benchmark comparison: Initial vs Best Optimized NetworkX BFS

This script compares the initial _single_source_shortest_path_basic function
from betweenness.py with the optimized versions from handwritten and LLM-generated
evaluations to demonstrate the performance improvements discovered by OpenEvolve.
"""

import time
import sys
import os
import shutil
import tempfile
import importlib
import re
import statistics
from pathlib import Path
from typing import Dict, List, Tuple
import networkx as nx


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
        # Extract content between ### FILE: networkx/algorithms/centrality/betweenness.py ### and ### END FILE ###
        pattern = r'### FILE: networkx/algorithms/centrality/betweenness\.py ###\s*\n(.*?)\n### END FILE ###'
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


def setup_networkx_module(base_dir, code=None):
    """
    Set up a networkx module by copying the base directory and optionally
    replacing betweenness.py with evolved code.
    
    Args:
        base_dir: Base directory containing the networkx package
        code: Optional code content to replace networkx/algorithms/centrality/betweenness.py
        
    Returns:
        Temporary directory path for importing
    """
    # Create temporary directory
    temp_dir = tempfile.mkdtemp(prefix='networkx_bench_')
    
    # Copy the entire networkx directory
    networkx_src = base_dir / 'networkx'
    networkx_dst = Path(temp_dir) / 'networkx'
    shutil.copytree(networkx_src, networkx_dst)
    
    # Replace betweenness.py if code provided
    if code:
        betweenness_file = networkx_dst / 'algorithms' / 'centrality' / 'betweenness.py'
        with open(betweenness_file, 'w') as f:
            f.write(code)
    
    return temp_dir


def cleanup_networkx_module(temp_dir):
    """Remove temporary directory and clean up sys.path and modules"""
    if temp_dir in sys.path:
        sys.path.remove(temp_dir)
    
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    
    # Clear networkx modules from cache
    modules_to_remove = [name for name in list(sys.modules.keys()) 
                        if name.startswith('networkx')]
    for name in modules_to_remove:
        if name in sys.modules:
            del sys.modules[name]


def test_correctness(func, graph_sizes: List[int] = [50, 100]) -> bool:
    """Verify that the function produces correct results"""
    for n in graph_sizes:
        G = nx.erdos_renyi_graph(n, 0.1, seed=42)
        source = 0
        
        try:
            result = func(G, source)
            S, P, sigma, D = result
            
            # Basic correctness checks
            if len(S) != len(D):
                return False
            if source not in D or D[source] != 0:
                return False
            if sigma[source] != 1.0:
                return False
        except Exception as e:
            print(f"Error in correctness test: {e}")
            return False
    
    return True


def benchmark_function(func, G, source: int, iterations: int = 20) -> Dict[str, float]:
    """Benchmark a function and return timing statistics"""
    times = []

    # Warmup
    for _ in range(3):
        func(G, source)

    # Actual timing
    for _ in range(iterations):
        start = time.perf_counter()
        result = func(G, source)
        end = time.perf_counter()
        times.append(end - start)

    return {
        'mean': statistics.mean(times),
        'median': statistics.median(times),
        'stdev': statistics.stdev(times) if len(times) > 1 else 0,
        'min': min(times),
        'max': max(times)
    }


def create_test_graphs() -> List[Tuple[str, nx.Graph, int]]:
    """Create various test graphs"""
    graphs = []

    # Small dense graph (stress test method calls)
    G1 = nx.complete_graph(100)
    graphs.append(("Dense Complete (K100)", G1, 0))

    # Medium dense graph
    G2 = nx.erdos_renyi_graph(300, 0.1, seed=42)
    graphs.append(("Medium Dense (300 nodes, p=0.1)", G2, 0))

    # Large sparse graph
    G3 = nx.erdos_renyi_graph(1000, 0.01, seed=42)
    graphs.append(("Large Sparse (1000 nodes, p=0.01)", G3, 0))

    # Very large sparse graph
    G4 = nx.erdos_renyi_graph(2000, 0.005, seed=42)
    graphs.append(("Very Large Sparse (2000 nodes, p=0.005)", G4, 0))

    # Grid graph (high diameter, many BFS iterations)
    G5 = nx.grid_2d_graph(40, 40)
    G5 = nx.convert_node_labels_to_integers(G5)
    graphs.append(("Grid 40x40 (1600 nodes)", G5, 0))

    return graphs


def main():
    """Main benchmark function"""
    print("🚀 NetworkX BFS Optimization Performance Test")
    print("=" * 70)
    
    # Get the task directory
    task_dir = Path(__file__).parent
    
    # Paths to files
    initial_file = task_dir / 'networkx' / 'algorithms' / 'centrality' / 'betweenness.py'
    handwritten_file = task_dir / 'best_program_handwritten.py'
    llm_generated_file = task_dir / 'best_program_llm_generated.py'
    
    # Extract code from files
    print("\n📂 Loading programs...")
    initial_code = extract_code_from_evolve_block(initial_file)
    handwritten_code = extract_code_from_evolve_block(handwritten_file)
    llm_generated_code = extract_code_from_evolve_block(llm_generated_file)
    
    # Set up temporary modules
    print("🔧 Setting up temporary modules...")
    initial_temp_dir = setup_networkx_module(task_dir, initial_code)
    handwritten_temp_dir = setup_networkx_module(task_dir, handwritten_code)
    llm_generated_temp_dir = setup_networkx_module(task_dir, llm_generated_code)
    
    try:
        # Load modules
        print("📦 Loading modules...")
        
        # Load initial version
        if initial_temp_dir not in sys.path:
            sys.path.insert(0, initial_temp_dir)
        modules_to_clear = [name for name in list(sys.modules.keys()) if name.startswith('networkx')]
        for name in modules_to_clear:
            if name in sys.modules:
                del sys.modules[name]
        initial_module = importlib.import_module('networkx.algorithms.centrality.betweenness')
        initial_func = initial_module._single_source_shortest_path_basic
        
        # Load handwritten version
        if initial_temp_dir in sys.path:
            sys.path.remove(initial_temp_dir)
        if handwritten_temp_dir not in sys.path:
            sys.path.insert(0, handwritten_temp_dir)
        modules_to_clear = [name for name in list(sys.modules.keys()) if name.startswith('networkx')]
        for name in modules_to_clear:
            if name in sys.modules:
                del sys.modules[name]
        handwritten_module = importlib.import_module('networkx.algorithms.centrality.betweenness')
        handwritten_func = handwritten_module._single_source_shortest_path_basic
        
        # Load LLM-generated version
        if handwritten_temp_dir in sys.path:
            sys.path.remove(handwritten_temp_dir)
        if llm_generated_temp_dir not in sys.path:
            sys.path.insert(0, llm_generated_temp_dir)
        modules_to_clear = [name for name in list(sys.modules.keys()) if name.startswith('networkx')]
        for name in modules_to_clear:
            if name in sys.modules:
                del sys.modules[name]
        llm_generated_module = importlib.import_module('networkx.algorithms.centrality.betweenness')
        llm_generated_func = llm_generated_module._single_source_shortest_path_basic
        
        # Test correctness
        print("\n🔍 Verifying correctness...")
        initial_correct = test_correctness(initial_func)
        handwritten_correct = test_correctness(handwritten_func)
        llm_generated_correct = test_correctness(llm_generated_func)
        
        print(f"   Initial:         {'✅' if initial_correct else '❌'}")
        print(f"   Handwritten:    {'✅' if handwritten_correct else '❌'}")
        print(f"   LLM-generated:  {'✅' if llm_generated_correct else '❌'}")
        
        if not (initial_correct and handwritten_correct and llm_generated_correct):
            print("\n❌ Correctness verification failed!")
            return
        
        # Create test graphs
        test_graphs = create_test_graphs()
        
        # Benchmark results storage
        results = []
        
        print("\n📊 Performance Comparison Results:")
        print("=" * 70)
        
        for graph_name, G, source in test_graphs:
            print(f"\n🧪 Testing: {graph_name}")
            print(f"   Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
            
            # Benchmark initial
            if initial_temp_dir not in sys.path:
                sys.path.insert(0, initial_temp_dir)
            if handwritten_temp_dir in sys.path:
                sys.path.remove(handwritten_temp_dir)
            if llm_generated_temp_dir in sys.path:
                sys.path.remove(llm_generated_temp_dir)
            modules_to_clear = [name for name in list(sys.modules.keys()) if name.startswith('networkx')]
            for name in modules_to_clear:
                if name in sys.modules:
                    del sys.modules[name]
            initial_module = importlib.import_module('networkx.algorithms.centrality.betweenness')
            initial_func = initial_module._single_source_shortest_path_basic
            initial_stats = benchmark_function(initial_func, G, source)
            
            # Benchmark handwritten
            if initial_temp_dir in sys.path:
                sys.path.remove(initial_temp_dir)
            if handwritten_temp_dir not in sys.path:
                sys.path.insert(0, handwritten_temp_dir)
            modules_to_clear = [name for name in list(sys.modules.keys()) if name.startswith('networkx')]
            for name in modules_to_clear:
                if name in sys.modules:
                    del sys.modules[name]
            handwritten_module = importlib.import_module('networkx.algorithms.centrality.betweenness')
            handwritten_func = handwritten_module._single_source_shortest_path_basic
            handwritten_stats = benchmark_function(handwritten_func, G, source)
            
            # Benchmark LLM-generated
            if handwritten_temp_dir in sys.path:
                sys.path.remove(handwritten_temp_dir)
            if llm_generated_temp_dir not in sys.path:
                sys.path.insert(0, llm_generated_temp_dir)
            modules_to_clear = [name for name in list(sys.modules.keys()) if name.startswith('networkx')]
            for name in modules_to_clear:
                if name in sys.modules:
                    del sys.modules[name]
            llm_generated_module = importlib.import_module('networkx.algorithms.centrality.betweenness')
            llm_generated_func = llm_generated_module._single_source_shortest_path_basic
            llm_generated_stats = benchmark_function(llm_generated_func, G, source)
            
            # Calculate speedups
            handwritten_speedup = initial_stats['mean'] / handwritten_stats['mean']
            llm_generated_speedup = initial_stats['mean'] / llm_generated_stats['mean']
            handwritten_improvement = (handwritten_speedup - 1) * 100
            llm_generated_improvement = (llm_generated_speedup - 1) * 100
            
            # Store results
            results.append({
                'graph_name': graph_name,
                'initial': initial_stats,
                'handwritten': handwritten_stats,
                'llm_generated': llm_generated_stats,
                'handwritten_speedup': handwritten_speedup,
                'llm_generated_speedup': llm_generated_speedup
            })
            
            # Display results
            print(f"   📈 Results:")
            print(f"      Initial:         {initial_stats['mean']*1000:.3f}ms ± {initial_stats['stdev']*1000:.3f}ms")
            print(f"      Handwritten:    {handwritten_stats['mean']*1000:.3f}ms ± {handwritten_stats['stdev']*1000:.3f}ms (speedup: {handwritten_speedup:.2f}x, {handwritten_improvement:+.1f}%)")
            print(f"      LLM-generated:  {llm_generated_stats['mean']*1000:.3f}ms ± {llm_generated_stats['stdev']*1000:.3f}ms (speedup: {llm_generated_speedup:.2f}x, {llm_generated_improvement:+.1f}%)")
        
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
        cleanup_networkx_module(initial_temp_dir)
        cleanup_networkx_module(handwritten_temp_dir)
        cleanup_networkx_module(llm_generated_temp_dir)


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
