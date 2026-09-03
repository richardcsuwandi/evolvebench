import importlib.util
import sys
import time
import math
import copy
from collections.abc import Mapping, Sequence


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


baseline_measurements = {
    "test_config_small": {
        "target_time": 0.003082,
        "num_comparisons": 5000,
        "data_complexity": "Small nested structure",
        "avg_time_per_comparison": 0.000001,
        "comparisons_per_second": 1622327.689980,
    },
    "test_config_medium": {
        "target_time": 0.011033,
        "num_comparisons": 2000,
        "data_complexity": "Medium nested structure with lists and dicts",
        "avg_time_per_comparison": 0.000006,
        "comparisons_per_second": 181269.978519,
    },
    "test_config_large": {
        "target_time": 0.032264,
        "num_comparisons": 1000,
        "data_complexity": "Large complex structure with deep nesting",
        "avg_time_per_comparison": 0.000032,
        "comparisons_per_second": 30994.240951,
    },
}


def load_program(program_path: str):
    """Load the program module from the given path."""
    spec = importlib.util.spec_from_file_location("program_module", program_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["program_module"] = module
    spec.loader.exec_module(module)
    return module


def evaluate_stage1(program_path: str) -> dict:
    """
    Quick validation with 5 diverse test cases.
    """
    try:
        module = load_program(program_path)
        equal = module.equal
        
        # Test cases for correctness
        test_cases = [
            # Test 1: Basic equality
            {"input": ({"a": 1, "b": 2}, {"a": 1, "b": 2}), "output": True},
            # Test 2: Bool vs int distinction (JSON Schema semantics)
            {"input": (True, 1), "output": False},
            {"input": (False, 0), "output": False},
            # Test 3: Nested structures
            {"input": ({"a": [1, 2, 3]}, {"a": [1, 2, 3]}), "output": True},
            # Test 4: Different values
            {"input": ({"a": 1}, {"a": 2}), "output": False},
            # Test 5: Lists
            {"input": ([1, 2, 3], [1, 2, 3]), "output": True},
        ]
        
        # Correctness evaluation
        passed = 0
        for test in test_cases:
            try:
                result = equal(test["input"][0], test["input"][1])
                if result == test["output"]:
                    passed += 1
            except Exception:
                pass
        
        correctness = passed / len(test_cases)
        
        # Performance evaluation - small benchmark
        small_data = {'a': 1, 'b': 2, 'c': {'d': 3, 'e': 4}}
        small_copy = copy.deepcopy(small_data)
        
        # Warmup
        for _ in range(10):
            equal(small_data, small_copy)
        
        # Benchmark
        num_comparisons = 5000
        start = time.time()
        for _ in range(num_comparisons):
            equal(small_data, small_copy)
        elapsed = time.time() - start
        
        # Use sigmoid scoring
        target_time = baseline_measurements["test_config_small"]["target_time"]
        performance = sigmoid_performance_score(elapsed, target_time, steepness=2.0)
        
        combined_score = 0.7 * correctness + 0.3 * performance
        
        return {
            "correctness": correctness,
            "performance": performance,
            "combined_score": combined_score,
            "stage": 1.0
        }
        
    except Exception as e:
        return {
            "correctness": 0.0,
            "performance": 0.0,
            "combined_score": 0.0,
            "stage": 1.0,
            "error": str(e)
        }


def evaluate_stage2(program_path: str) -> dict:
    """
    Comprehensive testing with 10+ test cases including edge cases.
    """
    try:
        module = load_program(program_path)
        equal = module.equal
        
        # Comprehensive test cases
        test_cases = [
            # Basic equality tests
            {"input": (1, 1), "output": True},
            {"input": (1, 2), "output": False},
            {"input": ("hello", "hello"), "output": True},
            {"input": ("hello", "world"), "output": False},
            
            # Bool vs int distinction (critical for JSON Schema)
            {"input": (True, 1), "output": False},
            {"input": (False, 0), "output": False},
            {"input": (True, True), "output": True},
            {"input": (False, False), "output": True},
            
            # Dict equality
            {"input": ({}, {}), "output": True},
            {"input": ({"a": 1}, {"a": 1}), "output": True},
            {"input": ({"a": 1}, {"a": 2}), "output": False},
            {"input": ({"a": 1}, {"b": 1}), "output": False},
            {"input": ({"a": 1, "b": 2}, {"b": 2, "a": 1}), "output": True},
            
            # List equality
            {"input": ([], []), "output": True},
            {"input": ([1, 2, 3], [1, 2, 3]), "output": True},
            {"input": ([1, 2, 3], [1, 2, 4]), "output": False},
            {"input": ([1, 2], [1, 2, 3]), "output": False},
            
            # Nested structures
            {"input": ({"a": [1, 2, 3]}, {"a": [1, 2, 3]}), "output": True},
            {"input": ({"a": [1, 2, 3]}, {"a": [1, 2, 4]}), "output": False},
            {"input": ({"a": {"b": {"c": 1}}}, {"a": {"b": {"c": 1}}}), "output": True},
            {"input": ([{"a": 1}, {"b": 2}], [{"a": 1}, {"b": 2}]), "output": True},
            
            # Mixed types
            {"input": (1, "1"), "output": False},
            {"input": ([1], {"0": 1}), "output": False},
            {"input": (None, None), "output": True},
            {"input": (None, 0), "output": False},
            
            # Edge cases with bools in structures
            {"input": ({"a": True}, {"a": 1}), "output": False},
            {"input": ([True, False], [1, 0]), "output": False},
            
            # Identity check
            {"input": (None, None), "output": True},
        ]
        
        # Correctness evaluation
        passed = 0
        for test in test_cases:
            try:
                result = equal(test["input"][0], test["input"][1])
                if result == test["output"]:
                    passed += 1
            except Exception:
                pass
        
        correctness = passed / len(test_cases)
        
        # Performance evaluation - multiple benchmarks
        performance_scores = []
        
        # Test 1: Small nested structure
        small_data = {'a': 1, 'b': 2, 'c': {'d': 3, 'e': 4}}
        small_copy = copy.deepcopy(small_data)
        num_comparisons_small = 5000
        
        # Warmup
        for _ in range(10):
            equal(small_data, small_copy)
        
        start = time.time()
        for _ in range(num_comparisons_small):
            equal(small_data, small_copy)
        elapsed_small = time.time() - start
        
        perf_small = sigmoid_performance_score(
            elapsed_small, 
            baseline_measurements["test_config_small"]["target_time"],
            steepness=2.0
        )
        performance_scores.append(perf_small)
        
        # Test 2: Medium nested structure
        medium_data = {
            'metadata': {'version': '1.0', 'author': 'test'},
            'data': [1, 2, 3, 4, 5],
            'nested': {'a': {'b': {'c': {'d': 1}}}}
        }
        medium_copy = copy.deepcopy(medium_data)
        num_comparisons_medium = 2000
        
        # Warmup
        for _ in range(10):
            equal(medium_data, medium_copy)
        
        start = time.time()
        for _ in range(num_comparisons_medium):
            equal(medium_data, medium_copy)
        elapsed_medium = time.time() - start
        
        perf_medium = sigmoid_performance_score(
            elapsed_medium,
            baseline_measurements["test_config_medium"]["target_time"],
            steepness=2.0
        )
        performance_scores.append(perf_medium)
        
        # Test 3: Large flat dict
        large_data = {f'key_{i}': i for i in range(100)}
        large_copy = copy.deepcopy(large_data)
        num_comparisons_large = 1000
        
        # Warmup
        for _ in range(10):
            equal(large_data, large_copy)
        
        start = time.time()
        for _ in range(num_comparisons_large):
            equal(large_data, large_copy)
        elapsed_large = time.time() - start
        
        perf_large = sigmoid_performance_score(
            elapsed_large,
            baseline_measurements["test_config_large"]["target_time"],
            steepness=2.0
        )
        performance_scores.append(perf_large)
        
        # Average performance score
        performance = sum(performance_scores) / len(performance_scores)
        
        combined_score = 0.7 * correctness + 0.3 * performance
        
        return {
            "correctness": correctness,
            "performance": performance,
            "combined_score": combined_score,
            "stage": 2.0,
            "details": {
                "tests_passed": passed,
                "total_tests": len(test_cases),
                "perf_small": perf_small,
                "perf_medium": perf_medium,
                "perf_large": perf_large,
                "elapsed_small": elapsed_small,
                "elapsed_medium": elapsed_medium,
                "elapsed_large": elapsed_large
            }
        }
        
    except Exception as e:
        return {
            "correctness": 0.0,
            "performance": 0.0,
            "combined_score": 0.0,
            "stage": 2.0,
            "error": str(e)
        }


def evaluate(program_path: str) -> dict:
    """
    Main evaluation function (REQUIRED).
    Delegates to evaluate_stage2 for comprehensive evaluation.
    """
    return evaluate_stage2(program_path)
