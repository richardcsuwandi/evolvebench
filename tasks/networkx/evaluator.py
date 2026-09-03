#!/usr/bin/env python3
"""
Evaluator for NetworkX BFS Shortest Path Algorithm

This evaluator tests the _single_source_shortest_path_basic function optimization.
It uses aggressive performance targets and sigmoid scoring to create proper
gradients for OpenEvolve to optimize against.

Based on empirical profiling data:
- Current performance: 0.2527s average for betweenness centrality
- Function accounts for 24.6% of total execution time
- Target: 2-10x speedup through algorithmic improvements
"""

import sys
import time
import math
import importlib
import tempfile
import os
from pathlib import Path
from collections import deque
from typing import Dict, Any

# Add the optimization tests directory to path
test_dir = Path(__file__).parent.parent / "optimization_tests"
if test_dir.exists():
    sys.path.insert(0, str(test_dir))

# Add the main directory for NetworkX imports
main_dir = Path(__file__).parent.parent
sys.path.insert(0, str(main_dir))


def sigmoid_performance_score(actual_time: float, target_time: float) -> float:
    """
    Create performance gradient using sigmoid scoring.

    This avoids the problem of linear scoring where anything faster than
    target gets perfect score (1.0), leaving no gradient for optimization.

    Sigmoid creates smooth gradient:
    - At target_time: score = 0.5
    - 2x faster: score ≈ 0.88
    - 2x slower: score ≈ 0.12
    """
    if actual_time <= 0:
        return 0.95  # Nearly perfect for instantaneous

    # Calculate relative performance vs target
    relative_slowdown = (actual_time - target_time) / target_time

    # Apply sigmoid curve (steepness=3 creates good gradient)
    return 1.0 / (1.0 + math.exp(3 * relative_slowdown))


def get_aggressive_targets(graph_size: str) -> float:
    """
    Set aggressive performance targets based on graph size.

    These targets are ~50x more aggressive than current performance
    to ensure initial code scores around 0.3-0.5, not 1.0.

    Based on test case measurements:
    - Small (100 nodes): current ~0.001s, target 0.00002s
    - Medium (300 nodes): current ~0.005s, target 0.0001s
    - Large (500 nodes): current ~0.015s, target 0.0003s
    """
    targets = {
        "small": 0.00002,   # 0.02ms - extremely aggressive
        "medium": 0.0001,   # 0.1ms - extremely aggressive
        "large": 0.0003,    # 0.3ms - extremely aggressive
        "xlarge": 0.001     # 1ms - for very large graphs
    }
    return targets.get(graph_size, 0.0001)


class GraphGenerator:
    """Generate test graphs of different sizes and structures"""

    @staticmethod
    def create_test_graph(size_category: str):
        """Create a test graph based on size category"""
        if size_category == "small":
            return GraphGenerator._create_erdos_renyi(100, 0.1)
        elif size_category == "medium":
            return GraphGenerator._create_erdos_renyi(300, 0.05)
        elif size_category == "large":
            return GraphGenerator._create_erdos_renyi(500, 0.03)
        elif size_category == "xlarge":
            return GraphGenerator._create_erdos_renyi(800, 0.02)
        else:
            return GraphGenerator._create_erdos_renyi(100, 0.1)

    @staticmethod
    def _create_erdos_renyi(n, p, seed=42):
        """Create Erdős-Rényi random graph as adjacency dict"""
        import random
        random.seed(seed)

        # Create graph as adjacency dictionary (NetworkX-like interface)
        class TestGraph:
            def __init__(self):
                self.adj = {i: {} for i in range(n)}
                self.nodes_list = list(range(n))

                # Add random edges
                for i in range(n):
                    for j in range(i + 1, n):
                        if random.random() < p:
                            self.adj[i][j] = {}
                            self.adj[j][i] = {}

            def __iter__(self):
                return iter(self.nodes_list)

            def __getitem__(self, node):
                return self.adj[node]

            def number_of_nodes(self):
                return len(self.nodes_list)

            def number_of_edges(self):
                return sum(len(neighbors) for neighbors in self.adj.values()) // 2

        return TestGraph()


def test_correctness(func, graph, source=0) -> bool:
    """Test if the function produces correct results"""
    try:
        S, P, sigma, D = func(graph, source)

        # Basic correctness checks
        if not isinstance(S, list) or not isinstance(P, dict):
            return False
        if not isinstance(sigma, dict) or not isinstance(D, dict):
            return False

        # Source node checks
        if source not in D or D[source] != 0:
            return False
        if source not in sigma or sigma[source] != 1.0:
            return False

        # All reachable nodes should be in results
        for node in graph:
            if node not in D or node not in sigma or node not in P:
                return False

        # Distance consistency
        for node in D:
            if D[node] < 0:
                return False

        # Path count consistency
        for node in sigma:
            if sigma[node] < 0:
                return False

        return True

    except Exception as e:
        return False


def measure_performance(func, graph, source=0, iterations=3) -> Dict[str, Any]:
    """Measure performance of the BFS function"""
    times = []

    for _ in range(iterations):
        start = time.perf_counter()
        try:
            S, P, sigma, D = func(graph, source)
            elapsed = time.perf_counter() - start
            times.append(elapsed)
        except Exception as e:
            return {
                "correct": False,
                "elapsed_time": float('inf'),
                "error": str(e)
            }

    # Check correctness
    correct = test_correctness(func, graph, source)

    avg_time = sum(times) / len(times) if times else float('inf')

    return {
        "correct": correct,
        "elapsed_time": avg_time,
        "min_time": min(times) if times else float('inf'),
        "max_time": max(times) if times else float('inf'),
        "graph_nodes": graph.number_of_nodes(),
        "graph_edges": graph.number_of_edges()
    }


def execute_and_measure(code: str, test_size: str = "medium") -> Dict[str, Any]:
    """Execute evolved code and measure its performance"""

    # Create a temporary module to execute the code
    try:
        # if provided code is a full upstream file (with heavy imports),
        # extract only the target function to avoid import errors (e.g., networkx)
        def _extract_target_function(src: str) -> str:
            """return a minimal module containing only _single_source_shortest_path_basic.

            this avoids importing external deps present at top-level in full files.
            """
            target_def = "def _single_source_shortest_path_basic"
            lines = src.splitlines()
            start_idx = None
            for i, line in enumerate(lines):
                if line.lstrip().startswith(target_def) and (len(line) - len(line.lstrip()) == 0):
                    start_idx = i
                    break
            # also handle cases where the def is indented in a larger file
            if start_idx is None:
                for i, line in enumerate(lines):
                    if line.lstrip().startswith(target_def):
                        start_idx = i
                        break
            if start_idx is None:
                return src  # fallback: return as-is

            # capture until next top-level def/class or end
            def is_toplevel_def(l: str) -> bool:
                s = l.lstrip()
                return (len(l) - len(l.lstrip()) == 0) and (s.startswith("def ") or s.startswith("class "))

            end_idx = len(lines)
            for j in range(start_idx + 1, len(lines)):
                if is_toplevel_def(lines[j]):
                    end_idx = j
                    break

            func_block = "\n".join(lines[start_idx:end_idx])
            header = "from collections import deque\n"
            return header + "\n\n" + func_block + "\n"

        # prefer minimal module if code seems to include external imports or evolve markers
        code_to_write = code
        if ("import networkx" in code) or ("EVOLVE-BLOCK-START" in code) or ("### FILE:" in code):
            code_to_write = _extract_target_function(code)

        # Handle NetworkX algorithm registry to prevent duplicate registration errors
        # The issue occurs when NetworkX code is imported multiple times and tries to
        # register the same algorithm in the dispatch registry. We temporarily patch
        # the _dispatchable class to allow duplicate registrations during evaluation.
        original_dispatchable_new = None
        registry_patched = False
        
        try:
            from networkx.utils.backends import _dispatchable, _registered_algorithms
            
            # Store original __new__ method
            original_dispatchable_new = _dispatchable.__new__
            
            # Create a patched version that allows duplicates
            def patched_new(cls, func=None, *, name=None, **kwargs):
                # Determine name first (before the check happens in original __new__)
                if name is None and func is not None and hasattr(func, '__name__'):
                    name = func.__name__
                
                # Check if this algorithm is already registered
                if name and name in _registered_algorithms:
                    # Algorithm already registered - return the existing one instead of raising error
                    return _registered_algorithms[name]
                
                # Call original __new__ - it might still raise if name wasn't determined above
                try:
                    return original_dispatchable_new(cls, func=func, name=name, **kwargs)
                except KeyError as e:
                    error_msg = str(e)
                    if "Algorithm already exists in dispatch namespace" in error_msg or "already exists in dispatch registry" in error_msg:
                        # Duplicate registration - return existing if we can find it
                        if name and name in _registered_algorithms:
                            return _registered_algorithms[name]
                        # If we still don't have name, try to get it from func
                        if func and hasattr(func, '__name__'):
                            name = func.__name__
                            if name in _registered_algorithms:
                                return _registered_algorithms[name]
                    # Re-raise if it's a different KeyError or we can't resolve
                    raise
            
            # Temporarily patch the __new__ method
            _dispatchable.__new__ = staticmethod(patched_new)
            registry_patched = True
        except (ImportError, AttributeError):
            # NetworkX not imported yet or structure different, no patching needed
            pass

        # Write code to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code_to_write)
            temp_file = f.name

        try:
            # Import the module - duplicate registrations will now be handled gracefully
            import importlib.util
            spec = importlib.util.spec_from_file_location("evolved_module", temp_file)
            module = importlib.util.module_from_spec(spec)
            sys.modules["evolved_module"] = module
            spec.loader.exec_module(module)
        finally:
            # Restore original __new__ method if we patched it
            if registry_patched and original_dispatchable_new is not None:
                try:
                    _dispatchable.__new__ = original_dispatchable_new
                except (ImportError, AttributeError):
                    pass

        # Get the function
        if hasattr(module, '_single_source_shortest_path_basic'):
            func = module._single_source_shortest_path_basic
        else:
            return {"correct": False, "elapsed_time": float('inf'), "error": "Function not found"}

        # Create test graph
        graph = GraphGenerator.create_test_graph(test_size)

        # Measure performance
        result = measure_performance(func, graph, source=0)

        # Cleanup
        os.unlink(temp_file)
        if "evolved_module" in sys.modules:
            del sys.modules["evolved_module"]

        return result

    except Exception as e:
        return {"correct": False, "elapsed_time": float('inf'), "error": str(e)}


def evaluate_stage1(code_or_path: str) -> Dict[str, float]:
    """
    Stage 1: Quick validation with small graph and aggressive target

    Args:
        code_or_path: Either Python code as string or path to Python file
    """
    # Check if input is a file path or code string
    if os.path.exists(code_or_path):
        # It's a file path - read the code
        with open(code_or_path, 'r') as f:
            code = f.read()
    else:
        # It's code directly
        code = code_or_path

    result = execute_and_measure(code, "small")

    correctness = 1.0 if result.get("correct", False) else 0.0
    elapsed_time = result.get("elapsed_time", float('inf'))

    # Use aggressive target and sigmoid scoring
    target_time = get_aggressive_targets("small")
    performance = sigmoid_performance_score(elapsed_time, target_time)

    # Penalize heavily for incorrectness
    if correctness < 1.0:
        performance *= 0.1

    combined_score = correctness * performance

    return {
        "correctness": correctness,
        "performance": performance,
        "combined_score": combined_score,
        "elapsed_time": elapsed_time,
        "target_time": target_time
    }


def evaluate_stage2(code_or_path: str) -> Dict[str, float]:
    """
    Stage 2: Comprehensive testing with multiple graph sizes

    Args:
        code_or_path: Either Python code as string or path to Python file
    """
    # Check if input is a file path or code string
    if os.path.exists(code_or_path):
        # It's a file path - read the code
        with open(code_or_path, 'r') as f:
            code = f.read()
    else:
        # It's code directly
        code = code_or_path
    test_sizes = ["small", "medium", "large"]
    weights = [0.2, 0.4, 0.4]  # Medium and large graphs matter most

    total_correctness = 0.0
    total_performance = 0.0

    for size, weight in zip(test_sizes, weights):
        result = execute_and_measure(code, size)

        correctness = 1.0 if result.get("correct", False) else 0.0
        elapsed_time = result.get("elapsed_time", float('inf'))

        # Use aggressive targets for each size
        target_time = get_aggressive_targets(size)
        performance = sigmoid_performance_score(elapsed_time, target_time)

        total_correctness += correctness * weight
        total_performance += performance * weight

    # Apply strong correctness gate
    if total_correctness < 0.9:
        total_performance *= (total_correctness / 0.9) ** 2

    combined_score = total_correctness * total_performanceta

    return {
        "correctness": total_correctness,
        "performance": total_performance,
        "combined_score": combined_score
    }


def evaluate(code_or_path: str) -> Dict[str, float]:
    """
    Main OpenEvolve entry point
    Returns dict with required keys: correctness, performance, combined_score

    Args:
        code_or_path: Either Python code as string or path to Python file
    """
    try:
        # Check if input is a file path or code string
        if os.path.exists(code_or_path):
            # It's a file path - read the code
            with open(code_or_path, 'r') as f:
                code = f.read()
        else:
            # It's code directly
            code = code_or_path

        # Stage 1 filter - quick validation
        stage1_result = evaluate_stage1(code)

        # If correctness is too low, don't proceed to expensive stage 2
        if stage1_result["correctness"] < 0.5:
            return {
                "correctness": stage1_result["correctness"],
                "performance": stage1_result["performance"],
                "combined_score": stage1_result["combined_score"]
            }

        # Stage 2 - comprehensive evaluation
        stage2_result = evaluate_stage2(code)

        return {
            "correctness": stage2_result["correctness"],
            "performance": stage2_result["performance"],
            "combined_score": stage2_result["combined_score"]
        }

    except Exception as e:
        # Return error metrics on any exception
        return {
            "correctness": 0.0,
            "performance": 0.0,
            "combined_score": 0.0
        }


if __name__ == "__main__":
    # Test the evaluator with the original code
    print("Testing NetworkX BFS Evaluator")
    print("=" * 50)

    # Read the initial program
    initial_file = Path(__file__).parent.parent / "initial_program.py"
    if initial_file.exists():
        with open(initial_file, 'r') as f:
            initial_code = f.read()

        print("Testing with initial BFS implementation...")
        result = evaluate(initial_code)

        print(f"Correctness: {result['correctness']:.3f}")
        print(f"Performance: {result['performance']:.3f}")
        print(f"Combined Score: {result['combined_score']:.3f}")

        if result['combined_score'] > 0.5:
            print("⚠️  Initial score too high - consider more aggressive targets")
        else:
            print("✓ Good performance gradient for optimization")
    else:
        print("❌ Initial program not found")

    # Test sigmoid scoring
    print(f"\nSigmoid Scoring Test (target: 0.001s):")
    test_times = [0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01]
    for t in test_times:
        score = sigmoid_performance_score(t, 0.001)
        print(f"  {t:.4f}s → {score:.3f}")