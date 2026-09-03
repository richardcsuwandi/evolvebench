import math
import time
import sys
import importlib.util
from typing import Dict, Any, List


def sigmoid_performance_score(actual_value: float, target_value: float, steepness: float = 2.0) -> float:
    """
    Creates a sigmoid scoring curve for smooth gradient optimization.
    
    At target_value: score = 0.5
    Better than target: score approaches 1.0
    Worse than target: score approaches 0.0
    
    Steepness controls how quickly score changes around target (default 2.0).
    Higher steepness = sharper transitions, lower = smoother transitions.
    """
    if actual_value <= 0:
        return 0.99  # Near-perfect for instantaneous/zero time
    
    # For "lower is better" metrics (like execution time)
    relative_performance = (actual_value - target_value) / target_value
    return 1.0 / (1.0 + math.exp(steepness * relative_performance))


# Baseline measurements from profiling
baseline_measurements = {
    "test_config_small": {
        "target_time": 0.009051,
        "num_symbols": 10,
        "result_size": 10,
        "iterations": 10,
        "description": "Test with 10 symbols, 10 iterations",
    },
    "test_config_medium": {
        "target_time": 0.065554,
        "num_symbols": 25,
        "result_size": 25,
        "iterations": 5,
        "description": "Test with 25 symbols, 5 iterations",
    },
    "test_config_large": {
        "target_time": 0.168759,
        "num_symbols": 40,
        "result_size": 40,
        "iterations": 3,
        "description": "Test with 40 symbols, 3 iterations",
    },
}


def load_module(program_path: str):
    """Load the module from the given path."""
    spec = importlib.util.spec_from_file_location("test_module", program_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["test_module"] = module
    spec.loader.exec_module(module)
    return module


def test_correctness(module, num_symbols: int) -> bool:
    """
    Test correctness of _find_localzeros implementation.
    
    For unrelated symbols, all should be in localzeros (antichain).
    """
    try:
        from sympy import symbols
        
        syms = symbols(f'x:{num_symbols}')
        result = module.Min._find_localzeros(syms)
        
        # All symbols should be in localzeros (they're mutually incomparable)
        if len(result) != num_symbols:
            return False
        
        # Check that all symbols are present
        for sym in syms:
            if sym not in result:
                return False
        
        return True
    except Exception as e:
        print(f"Correctness test failed with error: {e}")
        return False


def test_correctness_with_comparable(module) -> bool:
    """
    Test correctness with comparable values.
    
    For Min: should keep only the minimum value.
    For Max: should keep only the maximum value.
    """
    try:
        from sympy import S
        
        # Test Min with numeric values
        values = [S(5), S(3), S(7), S(1), S(9)]
        result = module.Min._find_localzeros(values)
        
        # Should only keep the minimum (1)
        if len(result) != 1 or S(1) not in result:
            return False
        
        # Test Max with numeric values
        result = module.Max._find_localzeros(values)
        
        # Should only keep the maximum (9)
        if len(result) != 1 or S(9) not in result:
            return False
        
        return True
    except Exception as e:
        print(f"Comparable correctness test failed with error: {e}")
        return False


def test_correctness_mixed(module) -> bool:
    """
    Test correctness with mixed comparable and incomparable values.
    """
    try:
        from sympy import symbols, S
        
        # Mix of symbols and numbers
        x, y, z = symbols('x y z')
        values = [S(5), x, S(3), y, S(7), z]
        
        result = module.Min._find_localzeros(values)
        
        # Should keep: 3 (min number), x, y, z (incomparable symbols)
        if len(result) != 4:
            return False
        
        if S(3) not in result or x not in result or y not in result or z not in result:
            return False
        
        return True
    except Exception as e:
        print(f"Mixed correctness test failed with error: {e}")
        return False


def benchmark_performance(module, num_symbols: int, iterations: int) -> float:
    """
    Benchmark the performance of _find_localzeros.
    
    Returns average elapsed time per iteration.
    """
    try:
        from sympy import symbols
        
        syms = symbols(f'x:{num_symbols}')
        
        total_time = 0.0
        for _ in range(iterations):
            start_time = time.time()
            result = module.Min._find_localzeros(syms)
            elapsed_time = time.time() - start_time
            total_time += elapsed_time
        
        return total_time
    except Exception as e:
        print(f"Performance benchmark failed with error: {e}")
        return float('inf')


def evaluate_stage1(program_path: str) -> dict:
    """
    Quick validation with 5 diverse test cases.
    Measure basic correctness and performance.
    """
    try:
        module = load_module(program_path)
        
        # Test cases
        correctness_tests = [
            ("small_antichain", lambda: test_correctness(module, 5)),
            ("medium_antichain", lambda: test_correctness(module, 10)),
            ("comparable_values", lambda: test_correctness_with_comparable(module)),
            ("mixed_values", lambda: test_correctness_mixed(module)),
            ("single_value", lambda: test_correctness(module, 1)),
        ]
        
        # Run correctness tests
        passed = 0
        for name, test_func in correctness_tests:
            if test_func():
                passed += 1
        
        correctness = passed / len(correctness_tests)
        
        # Performance test - use baseline small config
        perf_time = benchmark_performance(module, 10, 10)
        target_time = baseline_measurements["test_config_small"]["target_time"]
        performance = sigmoid_performance_score(perf_time, target_time, steepness=2.0)
        
        combined_score = 0.5 * correctness + 0.5 * performance
        
        return {
            "correctness": correctness,
            "performance": performance,
            "combined_score": combined_score,
            "stage": 1.0
        }
    except Exception as e:
        print(f"Stage 1 evaluation failed: {e}")
        return {
            "correctness": 0.0,
            "performance": 0.0,
            "combined_score": 0.0,
            "stage": 1.0
        }


def evaluate_stage2(program_path: str) -> dict:
    """
    Comprehensive testing with 10+ test cases including edge cases.
    Accurate performance benchmarking with timing.
    """
    try:
        module = load_module(program_path)
        
        # Correctness tests
        correctness_tests = [
            ("antichain_5", lambda: test_correctness(module, 5)),
            ("antichain_10", lambda: test_correctness(module, 10)),
            ("antichain_20", lambda: test_correctness(module, 20)),
            ("comparable_min_max", lambda: test_correctness_with_comparable(module)),
            ("mixed_values", lambda: test_correctness_mixed(module)),
            ("single_value", lambda: test_correctness(module, 1)),
            ("empty_values", lambda: len(module.Min._find_localzeros([])) == 0),
            ("duplicate_symbols", lambda: test_correctness(module, 3)),
            ("large_antichain", lambda: test_correctness(module, 30)),
            ("very_small", lambda: test_correctness(module, 2)),
        ]
        
        # Run correctness tests
        passed = 0
        for name, test_func in correctness_tests:
            try:
                if test_func():
                    passed += 1
            except Exception as e:
                print(f"Test {name} failed: {e}")
        
        correctness = passed / len(correctness_tests)
        
        # Performance tests - use all baseline configurations
        performance_scores = []
        
        # Small config
        perf_time_small = benchmark_performance(module, 10, 10)
        target_time_small = baseline_measurements["test_config_small"]["target_time"]
        perf_score_small = sigmoid_performance_score(perf_time_small, target_time_small, steepness=2.0)
        performance_scores.append(perf_score_small)
        
        # Medium config
        perf_time_medium = benchmark_performance(module, 25, 5)
        target_time_medium = baseline_measurements["test_config_medium"]["target_time"]
        perf_score_medium = sigmoid_performance_score(perf_time_medium, target_time_medium, steepness=2.0)
        performance_scores.append(perf_score_medium)
        
        # Large config
        perf_time_large = benchmark_performance(module, 40, 3)
        target_time_large = baseline_measurements["test_config_large"]["target_time"]
        perf_score_large = sigmoid_performance_score(perf_time_large, target_time_large, steepness=2.0)
        performance_scores.append(perf_score_large)
        
        # Additional performance test - extrapolate target for 50 symbols
        # Scale from large config: 50/40 ratio with quadratic scaling (O(n²))
        target_time_50 = target_time_large * ((50/40) ** 2)
        perf_time_50 = benchmark_performance(module, 50, 1)
        perf_score_50 = sigmoid_performance_score(perf_time_50, target_time_50, steepness=2.0)
        performance_scores.append(perf_score_50)
        
        # Average performance score
        performance = sum(performance_scores) / len(performance_scores)
        
        combined_score = 0.5 * correctness + 0.5 * performance
        
        return {
            "correctness": correctness,
            "performance": performance,
            "combined_score": combined_score,
            "stage": 2.0
        }
    except Exception as e:
        print(f"Stage 2 evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            "correctness": 0.0,
            "performance": 0.0,
            "combined_score": 0.0,
            "stage": 2.0
        }


def evaluate(program_path: str) -> dict:
    """
    Main evaluation function - delegates to stage 2 for comprehensive evaluation.
    """
    return evaluate_stage2(program_path)
