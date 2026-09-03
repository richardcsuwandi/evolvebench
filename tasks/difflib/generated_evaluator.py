import sys
import importlib.util
import time
import math
import random
import string
from typing import Dict, List, Tuple


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
        "target_time": 0.001309,
        "num_lines": 20,
        "line_length": 20,
        "similarity": 0.900000,
        "description": "Small test with 20 lines, length 20, 90% similarity",
    },
    "test_config_medium": {
        "target_time": 0.022646,
        "num_lines": 100,
        "line_length": 50,
        "similarity": 0.850000,
        "description": "Medium test with 100 lines, length 50, 85% similarity",
    },
    "test_config_large": {
        "target_time": 0.126782,
        "num_lines": 300,
        "line_length": 80,
        "similarity": 0.800000,
        "description": "Large test with 300 lines, length 80, 80% similarity",
    },
}


def load_program(program_path: str):
    """Load the program module from the given path."""
    spec = importlib.util.spec_from_file_location("program_module", program_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["program_module"] = module
    spec.loader.exec_module(module)
    return module


def generate_similar_lines(num_lines: int, line_length: int, similarity: float) -> Tuple[List[str], List[str]]:
    """Generate two lists of similar lines for testing."""
    def random_string(length):
        return ''.join(random.choices(string.ascii_letters + string.digits, k=length))
    
    a = []
    b = []
    
    for _ in range(num_lines):
        base_line = random_string(line_length) + '\n'
        a.append(base_line)
        
        # Create similar line based on similarity ratio
        if random.random() < similarity:
            # Make it similar by changing only a few characters
            chars = list(base_line[:-1])  # Exclude newline
            num_changes = max(1, int(line_length * (1 - similarity)))
            for _ in range(num_changes):
                pos = random.randint(0, len(chars) - 1)
                chars[pos] = random.choice(string.ascii_letters + string.digits)
            b.append(''.join(chars) + '\n')
        else:
            # Make it different
            b.append(random_string(line_length) + '\n')
    
    return a, b


def run_differ_test(module, a: List[str], b: List[str]) -> Tuple[bool, float, List[str]]:
    """Run a differ test and return success, elapsed time, and result."""
    try:
        differ = module.Differ()
        
        start_time = time.perf_counter()
        result = list(differ.compare(a, b))
        elapsed_time = time.perf_counter() - start_time
        
        # Basic validation: result should not be empty for different inputs
        success = len(result) > 0
        
        return success, elapsed_time, result
    except Exception as e:
        return False, float('inf'), []


def verify_correctness(module) -> Tuple[bool, str]:
    """Verify basic correctness of the Differ implementation."""
    try:
        differ = module.Differ()
        
        # Test 1: Identical sequences
        a = ["line1\n", "line2\n", "line3\n"]
        b = ["line1\n", "line2\n", "line3\n"]
        result = list(differ.compare(a, b))
        if not all(line.startswith('  ') for line in result):
            return False, "Identical sequences should produce all equal lines"
        
        # Test 2: Completely different sequences
        a = ["aaa\n", "bbb\n"]
        b = ["xxx\n", "yyy\n"]
        result = list(differ.compare(a, b))
        if len(result) == 0:
            return False, "Different sequences should produce output"
        
        # Test 3: Similar sequences (fancy_replace should be triggered)
        a = ["0123456789\n"] * 5
        b = ["01234a6789\n"] * 5
        result = list(differ.compare(a, b))
        if len(result) == 0:
            return False, "Similar sequences should produce output"
        
        # Test 4: Empty sequences
        a = []
        b = []
        result = list(differ.compare(a, b))
        if len(result) != 0:
            return False, "Empty sequences should produce no output"
        
        # Test 5: One empty sequence
        a = ["line\n"]
        b = []
        result = list(differ.compare(a, b))
        if len(result) == 0:
            return False, "One empty sequence should produce output"
        
        return True, "All correctness tests passed"
    except Exception as e:
        return False, f"Exception during correctness check: {str(e)}"


def evaluate_stage1(program_path: str) -> dict:
    """Quick validation with 5 diverse test cases."""
    try:
        module = load_program(program_path)
        
        # Verify basic correctness
        correct, message = verify_correctness(module)
        if not correct:
            return {
                "correctness": 0.0,
                "performance": 0.0,
                "combined_score": 0.0,
                "stage": 1.0,
                "error": message
            }
        
        # Run 5 diverse test cases
        test_configs = [
            {"num_lines": 10, "line_length": 15, "similarity": 0.95, "target_time": 0.0005},
            {"num_lines": 20, "line_length": 20, "similarity": 0.90, "target_time": 0.001309},
            {"num_lines": 50, "line_length": 30, "similarity": 0.85, "target_time": 0.008},
            {"num_lines": 100, "line_length": 50, "similarity": 0.85, "target_time": 0.022646},
            {"num_lines": 150, "line_length": 60, "similarity": 0.80, "target_time": 0.055},
        ]
        
        correctness_scores = []
        performance_scores = []
        
        for config in test_configs:
            a, b = generate_similar_lines(config["num_lines"], config["line_length"], config["similarity"])
            success, elapsed_time, result = run_differ_test(module, a, b)
            
            correctness_scores.append(1.0 if success else 0.0)
            performance_scores.append(sigmoid_performance_score(elapsed_time, config["target_time"], steepness=2.0))
        
        correctness = sum(correctness_scores) / len(correctness_scores)
        performance = sum(performance_scores) / len(performance_scores)
        combined_score = 0.5 * correctness + 0.5 * performance
        
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
    """Comprehensive testing with 10+ test cases including edge cases."""
    try:
        module = load_program(program_path)
        
        # Verify basic correctness
        correct, message = verify_correctness(module)
        if not correct:
            return {
                "correctness": 0.0,
                "performance": 0.0,
                "combined_score": 0.0,
                "stage": 2.0,
                "error": message
            }
        
        # Comprehensive test cases
        test_configs = [
            # Baseline configurations
            {"num_lines": 20, "line_length": 20, "similarity": 0.90, "target_time": 0.001309, "name": "small"},
            {"num_lines": 100, "line_length": 50, "similarity": 0.85, "target_time": 0.022646, "name": "medium"},
            {"num_lines": 300, "line_length": 80, "similarity": 0.80, "target_time": 0.126782, "name": "large"},
            
            # Additional test cases
            {"num_lines": 5, "line_length": 10, "similarity": 0.95, "target_time": 0.0002, "name": "tiny"},
            {"num_lines": 50, "line_length": 30, "similarity": 0.90, "target_time": 0.008, "name": "small-medium"},
            {"num_lines": 150, "line_length": 60, "similarity": 0.85, "target_time": 0.055, "name": "medium-large"},
            {"num_lines": 200, "line_length": 70, "similarity": 0.82, "target_time": 0.085, "name": "large-1"},
            {"num_lines": 250, "line_length": 75, "similarity": 0.81, "target_time": 0.105, "name": "large-2"},
            
            # High similarity (pathological case)
            {"num_lines": 100, "line_length": 50, "similarity": 0.95, "target_time": 0.030, "name": "high-similarity"},
            
            # Low similarity (should be faster)
            {"num_lines": 100, "line_length": 50, "similarity": 0.50, "target_time": 0.015, "name": "low-similarity"},
            
            # Long lines
            {"num_lines": 50, "line_length": 100, "similarity": 0.85, "target_time": 0.020, "name": "long-lines"},
            
            # Many short lines
            {"num_lines": 200, "line_length": 20, "similarity": 0.85, "target_time": 0.040, "name": "many-short"},
        ]
        
        correctness_scores = []
        performance_scores = []
        
        for config in test_configs:
            a, b = generate_similar_lines(config["num_lines"], config["line_length"], config["similarity"])
            success, elapsed_time, result = run_differ_test(module, a, b)
            
            correctness_scores.append(1.0 if success else 0.0)
            performance_scores.append(sigmoid_performance_score(elapsed_time, config["target_time"], steepness=2.0))
        
        # Edge cases
        edge_cases = [
            {"a": [], "b": [], "target_time": 0.00001, "name": "empty"},
            {"a": ["x\n"], "b": [], "target_time": 0.00001, "name": "one-empty"},
            {"a": ["x\n"], "b": ["x\n"], "target_time": 0.00001, "name": "single-identical"},
            {"a": ["x\n"], "b": ["y\n"], "target_time": 0.00001, "name": "single-different"},
        ]
        
        for case in edge_cases:
            success, elapsed_time, result = run_differ_test(module, case["a"], case["b"])
            correctness_scores.append(1.0 if success else 0.0)
            performance_scores.append(sigmoid_performance_score(elapsed_time, case["target_time"], steepness=2.0))
        
        correctness = sum(correctness_scores) / len(correctness_scores)
        performance = sum(performance_scores) / len(performance_scores)
        combined_score = 0.5 * correctness + 0.5 * performance
        
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


def evaluate(program_path: str) -> dict:
    """Main evaluation function."""
    return evaluate_stage2(program_path)
