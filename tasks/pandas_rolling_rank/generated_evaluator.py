import sys
import importlib.util
import time
import math
import numpy as np
import pandas as pd
from typing import Dict, Any


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
        "target_time": 0.000122,
        "n_values": 1000,
        "window_size": 50,
        "method": "average",
        "ascending": True,
        "pct": False,
        "description": "Small test with n_values=1000, window_size=50, method=average, ascending=True, pct=False",
    },
    "test_config_medium": {
        "target_time": 0.001943,
        "n_values": 10000,
        "window_size": 100,
        "method": "average",
        "ascending": True,
        "pct": False,
        "description": "Medium test with n_values=10000, window_size=100, method=average, ascending=True, pct=False",
    },
    "test_config_large": {
        "target_time": 0.014675,
        "n_values": 50000,
        "window_size": 200,
        "method": "average",
        "ascending": True,
        "pct": False,
        "description": "Large test with n_values=50000, window_size=200, method=average, ascending=True, pct=False",
    },
}


def load_program(program_path: str):
    """Load the program module from the given path."""
    spec = importlib.util.spec_from_file_location("program_module", program_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["program_module"] = module
    spec.loader.exec_module(module)
    return module


def compare_results(result, expected, rtol=1e-5, atol=1e-8):
    """Compare two arrays allowing for NaN values."""
    if len(result) != len(expected):
        return False
    
    result = np.asarray(result, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    
    # Check NaN positions match
    result_nan = np.isnan(result)
    expected_nan = np.isnan(expected)
    if not np.array_equal(result_nan, expected_nan):
        return False
    
    # Check non-NaN values match
    non_nan_mask = ~result_nan
    if not np.allclose(result[non_nan_mask], expected[non_nan_mask], rtol=rtol, atol=atol):
        return False
    
    return True


def generate_test_cases_stage1():
    """Generate 5 diverse test cases for stage 1."""
    test_cases = []
    
    # Test 1: Basic small window
    np.random.seed(42)
    values1 = np.random.randn(100)
    test_cases.append({
        "input": {
            "values": values1,
            "window_size": 10,
            "method": "average",
            "ascending": True,
            "pct": False
        },
        "config": "basic_small"
    })
    
    # Test 2: With NaN values
    values2 = np.random.randn(100)
    values2[10:15] = np.nan
    test_cases.append({
        "input": {
            "values": values2,
            "window_size": 20,
            "method": "average",
            "ascending": True,
            "pct": False
        },
        "config": "with_nan"
    })
    
    # Test 3: Descending order
    values3 = np.random.randn(100)
    test_cases.append({
        "input": {
            "values": values3,
            "window_size": 15,
            "method": "average",
            "ascending": False,
            "pct": False
        },
        "config": "descending"
    })
    
    # Test 4: Min method
    values4 = np.random.randn(100)
    test_cases.append({
        "input": {
            "values": values4,
            "window_size": 10,
            "method": "min",
            "ascending": True,
            "pct": False
        },
        "config": "min_method"
    })
    
    # Test 5: Percentile rank
    values5 = np.random.randn(100)
    test_cases.append({
        "input": {
            "values": values5,
            "window_size": 10,
            "method": "average",
            "ascending": True,
            "pct": True
        },
        "config": "percentile"
    })
    
    return test_cases


def generate_test_cases_stage2():
    """Generate 10+ comprehensive test cases for stage 2."""
    test_cases = generate_test_cases_stage1()
    
    # Test 6: Max method
    np.random.seed(100)
    values6 = np.random.randn(200)
    test_cases.append({
        "input": {
            "values": values6,
            "window_size": 25,
            "method": "max",
            "ascending": True,
            "pct": False
        },
        "config": "max_method"
    })
    
    # Test 7: Baseline small config
    np.random.seed(200)
    values7 = np.random.randn(1000)
    test_cases.append({
        "input": {
            "values": values7,
            "window_size": 50,
            "method": "average",
            "ascending": True,
            "pct": False
        },
        "config": "test_config_small"
    })
    
    # Test 8: Baseline medium config
    np.random.seed(300)
    values8 = np.random.randn(10000)
    test_cases.append({
        "input": {
            "values": values8,
            "window_size": 100,
            "method": "average",
            "ascending": True,
            "pct": False
        },
        "config": "test_config_medium"
    })
    
    # Test 9: Baseline large config
    np.random.seed(400)
    values9 = np.random.randn(50000)
    test_cases.append({
        "input": {
            "values": values9,
            "window_size": 200,
            "method": "average",
            "ascending": True,
            "pct": False
        },
        "config": "test_config_large"
    })
    
    # Test 10: With inf values
    values10 = np.random.randn(100)
    values10[20:25] = np.inf
    test_cases.append({
        "input": {
            "values": values10,
            "window_size": 15,
            "method": "average",
            "ascending": True,
            "pct": False
        },
        "config": "with_inf"
    })
    
    # Test 11: Descending with percentile
    values11 = np.random.randn(150)
    test_cases.append({
        "input": {
            "values": values11,
            "window_size": 20,
            "method": "average",
            "ascending": False,
            "pct": True
        },
        "config": "descending_pct"
    })
    
    # Test 12: Large window
    values12 = np.random.randn(500)
    test_cases.append({
        "input": {
            "values": values12,
            "window_size": 100,
            "method": "average",
            "ascending": True,
            "pct": False
        },
        "config": "large_window"
    })
    
    return test_cases


def get_expected_result(test_input):
    """Get expected result using pandas."""
    values = test_input["values"]
    window_size = test_input["window_size"]
    method = test_input["method"]
    ascending = test_input["ascending"]
    pct = test_input["pct"]
    
    series = pd.Series(values)
    result = series.rolling(window=window_size, min_periods=window_size).rank(
        method=method, ascending=ascending, pct=pct
    ).values
    
    return result


def evaluate_stage1(program_path: str) -> Dict[str, Any]:
    """Quick validation with 5 diverse test cases."""
    try:
        module = load_program(program_path)
        RollingRank = module.RollingRank
        
        test_cases = generate_test_cases_stage1()
        
        correct_count = 0
        total_tests = len(test_cases)
        performance_scores = []
        
        for test_case in test_cases:
            test_input = test_case["input"]
            config = test_case["config"]
            
            try:
                # Get expected result
                expected = get_expected_result(test_input)
                
                # Create RollingRank instance
                roller = RollingRank(
                    window_size=test_input["window_size"],
                    method=test_input["method"],
                    ascending=test_input["ascending"],
                    pct=test_input["pct"]
                )
                
                # Warmup for Numba
                roller.compute(np.random.randn(min(100, len(test_input["values"]))))
                
                # Time the computation
                start_time = time.time()
                result = roller.compute(test_input["values"])
                elapsed_time = time.time() - start_time
                
                # Check correctness
                if compare_results(result, expected):
                    correct_count += 1
                
                # Estimate target time for performance scoring
                n_values = len(test_input["values"])
                window_size = test_input["window_size"]
                
                # Scale from baseline small config
                base_n = baseline_measurements["test_config_small"]["n_values"]
                base_w = baseline_measurements["test_config_small"]["window_size"]
                base_time = baseline_measurements["test_config_small"]["target_time"]
                
                # Rough scaling: O(n*w)
                estimated_target = base_time * (n_values / base_n) * (window_size / base_w)
                
                perf_score = sigmoid_performance_score(elapsed_time, estimated_target, steepness=2.0)
                performance_scores.append(perf_score)
                
            except Exception as e:
                # Test failed
                performance_scores.append(0.0)
        
        correctness = correct_count / total_tests if total_tests > 0 else 0.0
        performance = sum(performance_scores) / len(performance_scores) if performance_scores else 0.0
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


def evaluate_stage2(program_path: str) -> Dict[str, Any]:
    """Comprehensive testing with 10+ test cases including edge cases."""
    try:
        module = load_program(program_path)
        RollingRank = module.RollingRank
        
        test_cases = generate_test_cases_stage2()
        
        correct_count = 0
        total_tests = len(test_cases)
        performance_scores = []
        
        for test_case in test_cases:
            test_input = test_case["input"]
            config = test_case["config"]
            
            try:
                # Get expected result
                expected = get_expected_result(test_input)
                
                # Create RollingRank instance
                roller = RollingRank(
                    window_size=test_input["window_size"],
                    method=test_input["method"],
                    ascending=test_input["ascending"],
                    pct=test_input["pct"]
                )
                
                # Warmup for Numba
                roller.compute(np.random.randn(min(100, len(test_input["values"]))))
                
                # Time the computation
                start_time = time.time()
                result = roller.compute(test_input["values"])
                elapsed_time = time.time() - start_time
                
                # Check correctness
                if compare_results(result, expected):
                    correct_count += 1
                
                # Get target time for performance scoring
                if config in baseline_measurements:
                    target_time = baseline_measurements[config]["target_time"]
                else:
                    # Estimate target time based on problem size
                    n_values = len(test_input["values"])
                    window_size = test_input["window_size"]
                    
                    # Scale from baseline small config
                    base_n = baseline_measurements["test_config_small"]["n_values"]
                    base_w = baseline_measurements["test_config_small"]["window_size"]
                    base_time = baseline_measurements["test_config_small"]["target_time"]
                    
                    # Rough scaling: O(n*w)
                    target_time = base_time * (n_values / base_n) * (window_size / base_w)
                
                perf_score = sigmoid_performance_score(elapsed_time, target_time, steepness=2.0)
                performance_scores.append(perf_score)
                
            except Exception as e:
                # Test failed
                performance_scores.append(0.0)
        
        correctness = correct_count / total_tests if total_tests > 0 else 0.0
        performance = sum(performance_scores) / len(performance_scores) if performance_scores else 0.0
        combined_score = 0.7 * correctness + 0.3 * performance
        
        return {
            "correctness": correctness,
            "performance": performance,
            "combined_score": combined_score,
            "stage": 2.0
        }
        
    except Exception as e:
        return {
            "correctness": 0.0,
            "performance": 0.0,
            "combined_score": 0.0,
            "stage": 2.0,
            "error": str(e)
        }


def evaluate(program_path: str) -> Dict[str, Any]:
    """Main evaluation function."""
    return evaluate_stage2(program_path)
