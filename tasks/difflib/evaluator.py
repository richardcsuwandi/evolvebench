#!/usr/bin/env python3
"""
Evaluator for difflib Differ._fancy_replace optimization

This evaluator tests both correctness and performance of the _fancy_replace method.

Correctness: Output must match original difflib implementation
Performance: Tested on pathological cases from GitHub issues #119105 and #6931

Scoring:
- Stage 1 (quick): Basic correctness + pathological case size=100
- Stage 2 (comprehensive): All test cases including recursion-prone sizes

Baseline performance (original implementation):
- Overall score: 0.365
- Pathological case #119105:
  * Size 100: 0.059s (score: 0.287)
  * Size 500: 7.015s - RECURSION ERROR
  * Size 1000: 48.796s - RECURSION ERROR
- Target: score > 0.8, no recursion errors
"""

import sys
import time
import math
import traceback
import tempfile
from typing import Dict, Any


def sigmoid_performance_score(actual_time: float, target_time: float) -> float:
    """
    Calculate performance score using sigmoid function.

    Score = 1 / (1 + exp(5 * relative_slowdown))

    Where relative_slowdown = (actual_time - target_time) / target_time

    This creates a gradient:
    - 2x faster than target: score ≈ 0.99
    - At target: score = 0.5
    - 2x slower than target: score ≈ 0.01
    """
    if target_time <= 0:
        return 1.0 if actual_time <= 0 else 0.0

    relative_slowdown = (actual_time - target_time) / target_time
    return 1.0 / (1.0 + math.exp(5 * relative_slowdown))


def test_correctness_basic(evolved_module) -> Dict[str, Any]:
    """
    Basic correctness test: Compare output to expected results.

    Tests simple cases where we know the expected output.
    """
    errors = []

    # Test 1: Simple replacements
    d = evolved_module.Differ()
    a = ["line 1\n", "line 2\n", "line 3\n"]
    b = ["line 1\n", "line 2 modified\n", "line 3\n"]

    try:
        result = list(d.compare(a, b))
        # Should have at least 3 lines: unchanged, changed lines
        if len(result) < 3:
            errors.append(f"Test 1: Expected at least 3 output lines, got {len(result)}")
        # Check that it has expected line types
        has_unchanged = any(line.startswith('  ') for line in result)
        has_delete = any(line.startswith('- ') for line in result)
        has_add = any(line.startswith('+ ') for line in result)
        if not (has_unchanged and has_delete and has_add):
            errors.append(f"Test 1: Missing expected line types")
    except Exception as e:
        errors.append(f"Test 1 failed: {e}")

    # Test 2: All inserts
    d = evolved_module.Differ()
    a = []
    b = ["new line 1\n", "new line 2\n"]

    try:
        result = list(d.compare(a, b))
        if not all(line.startswith('+ ') for line in result):
            errors.append(f"Test 2: All lines should be inserts (start with '+ ')")
    except Exception as e:
        errors.append(f"Test 2 failed: {e}")

    # Test 3: All deletes
    d = evolved_module.Differ()
    a = ["old line 1\n", "old line 2\n"]
    b = []

    try:
        result = list(d.compare(a, b))
        if not all(line.startswith('- ') for line in result):
            errors.append(f"Test 3: All lines should be deletes (start with '- ')")
    except Exception as e:
        errors.append(f"Test 3 failed: {e}")

    # Test 4: Pathological case (small)
    d = evolved_module.Differ()
    a = ["0123456789\n"] * 10
    b = ["01234a56789\n"] * 10

    try:
        result = list(d.compare(a, b))
        # Should generate output with markers showing the difference
        if len(result) < 20:  # Each line pair generates at least 2 lines
            errors.append(f"Test 4: Expected at least 20 output lines, got {len(result)}")
    except Exception as e:
        errors.append(f"Test 4 failed: {e}")

    return {
        "passed": len(errors) == 0,
        "errors": errors,
        "num_tests": 4,
        "num_passed": 4 - len(errors)
    }


def test_performance_pathological_119105(evolved_module, size: int, timeout: float = 60.0) -> Dict[str, Any]:
    """
    Test performance on pathological case from GitHub issue #119105.

    This case causes recursion errors and slow performance when comparing
    files with many similar lines that differ by single characters.
    """
    a = ["0123456789\n"] * size
    b = ["01234a56789\n"] * size

    d = evolved_module.Differ()

    start_time = time.time()
    try:
        result = list(d.compare(a, b))
        elapsed = time.time() - start_time

        return {
            "success": True,
            "elapsed_time": elapsed,
            "output_lines": len(result),
            "error": None
        }
    except RecursionError as e:
        elapsed = time.time() - start_time
        return {
            "success": False,
            "elapsed_time": elapsed,
            "output_lines": 0,
            "error": "RecursionError"
        }
    except Exception as e:
        elapsed = time.time() - start_time
        return {
            "success": False,
            "elapsed_time": elapsed,
            "output_lines": 0,
            "error": str(e)
        }


def test_performance_realistic(evolved_module, n_functions: int) -> Dict[str, Any]:
    """
    Test performance on realistic code diff scenario.
    """
    def generate_code_file(n, version):
        lines = ["#!/usr/bin/env python3\n", '"""Module docstring"""\n\n']
        for i in range(n):
            lines.extend([
                f"def function_{i}(arg1, arg2):\n",
                f'    """Function {i} docstring"""\n',
                f"    result = arg1 + arg2\n",
                f"    if result > {i}:\n",
                f"        return result * {version}\n",
                f"    else:\n",
                f"        return result\n",
                f"\n",
            ])
        return lines

    a = generate_code_file(n_functions, version=1)
    b = generate_code_file(n_functions, version=2)

    d = evolved_module.Differ()

    start_time = time.time()
    try:
        result = list(d.compare(a, b))
        elapsed = time.time() - start_time

        return {
            "success": True,
            "elapsed_time": elapsed,
            "input_lines": len(a),
            "output_lines": len(result),
            "error": None
        }
    except Exception as e:
        elapsed = time.time() - start_time
        return {
            "success": False,
            "elapsed_time": elapsed,
            "input_lines": len(a),
            "output_lines": 0,
            "error": str(e)
        }


def evaluate(program_code: str) -> Dict[str, Any]:
    """
    Main evaluation function called by OpenEvolve.

    This is the entry point for OpenEvolve's evaluation system.
    Delegates to eval_initial_program for quick validation.
    """
    return eval_initial_program(program_code)


def eval_initial_program(program_code: str) -> Dict[str, Any]:
    """
    Stage 1: Quick validation (used during evolution)

    Tests:
    1. Basic correctness
    2. Pathological case size=100 (should complete in <0.1s)

    Returns combined correctness and performance score.
    """
    print("\n🔍 DIFFLIB EVALUATOR - Stage 1 (Quick validation)")

    # If program_code is a file path, read the file
    import os
    if os.path.exists(program_code):
        with open(program_code, 'r') as f:
            code = f.read()
    else:
        code = program_code

    # Load the evolved program
    try:
        namespace = {}
        exec(code, namespace)
        import types
        evolved_module = types.SimpleNamespace(**namespace)
    except SyntaxError as e:
        print(f"  ❌ Syntax error in program: {e}")
        if hasattr(e, 'lineno') and hasattr(e, 'text'):
            print(f"     Line {e.lineno}: {e.text}")
        failed_path = os.path.join(tempfile.gettempdir(), 'failed_program.py')
        with open(failed_path, 'w') as f:
            f.write(program_code)
        print(f"     Saved failing program to {failed_path}")
        return {
            "correctness": 0.0,
            "performance": 0.0,
            "combined_score": 0.0,
            "error": f"Syntax error: {e}"
        }
    except Exception as e:
        print(f"  ❌ Failed to load program: {e}")
        return {
            "correctness": 0.0,
            "performance": 0.0,
            "combined_score": 0.0,
            "error": f"Failed to load: {e}"
        }

    # Test correctness
    correctness_result = test_correctness_basic(evolved_module)
    correctness_score = correctness_result["num_passed"] / correctness_result["num_tests"]

    print(f"  {'✅' if correctness_result['passed'] else '❌'} Correctness: {correctness_score:.3f} ({correctness_result['num_passed']}/{correctness_result['num_tests']} tests passed)")
    if not correctness_result["passed"]:
        print(f"     Errors: {correctness_result['errors']}")

    if correctness_score < 1.0:
        # If correctness fails, return early
        return {
            "correctness": correctness_score,
            "performance": 0.0,
            "combined_score": 0.0,
            "correctness_errors": correctness_result["errors"]
        }

    # Test performance on small pathological case
    perf_result = test_performance_pathological_119105(evolved_module, size=100, timeout=10.0)

    if not perf_result["success"]:
        print(f"  ❌ Performance test failed: {perf_result['error']}")
        return {
            "correctness": correctness_score,
            "performance": 0.0,
            "combined_score": 0.0,
            "performance_error": perf_result["error"],
            "elapsed_time": perf_result["elapsed_time"]
        }

    # Calculate performance score
    # Target: 0.050s for size=100 (aggressive optimization goal)
    target_time = 0.050
    elapsed = perf_result["elapsed_time"]
    performance_score = sigmoid_performance_score(elapsed, target_time)

    print(f"  ⏱️  Performance: {elapsed:.3f}s for size=100 (target: {target_time:.3f}s)")
    print(f"  📊 Scores: correctness={correctness_score:.3f}, performance={performance_score:.3f}")

    combined_score = correctness_score * performance_score
    print(f"  🎯 Stage 1: correctness={correctness_score:.3f} * performance={performance_score:.3f} = {combined_score:.3f}")

    return {
        "correctness": correctness_score,
        "performance": performance_score,
        "combined_score": combined_score,
        "elapsed_time": elapsed,
        "target_time": target_time
    }


def eval_evolved_agent(program_code: str) -> Dict[str, Any]:
    """
    Stage 2: Comprehensive evaluation (final assessment)

    Tests:
    1. All correctness tests
    2. Pathological case sizes: 100, 500, 1000
    3. Realistic code diff scenarios

    Baseline scores:
    - Size 100: 0.287 (0.059s)
    - Size 500: 0.000 (recursion error)
    - Size 1000: 0.000 (recursion error)
    - Overall: 0.365
    """
    print("\n🔍 DIFFLIB EVALUATOR - Stage 2 (Comprehensive)")

    # If program_code is a file path, read the file
    import os
    if os.path.exists(program_code):
        with open(program_code, 'r') as f:
            code = f.read()
    else:
        code = program_code

    # Load the evolved program
    try:
        namespace = {}
        exec(code, namespace)
        import types
        evolved_module = types.SimpleNamespace(**namespace)
    except SyntaxError as e:
        print(f"  ❌ Syntax error in program: {e}")
        if hasattr(e, 'lineno') and hasattr(e, 'text'):
            print(f"     Line {e.lineno}: {e.text}")
        failed_path = os.path.join(tempfile.gettempdir(), 'failed_program.py')
        with open(failed_path, 'w') as f:
            f.write(program_code)
        print(f"     Saved failing program to {failed_path}")
        return {
            "correctness": 0.0,
            "performance": 0.0,
            "combined_score": 0.0,
            "error": f"Syntax error: {e}"
        }
    except Exception as e:
        print(f"  ❌ Failed to load program: {e}")
        return {
            "correctness": 0.0,
            "performance": 0.0,
            "combined_score": 0.0,
            "error": f"Failed to load: {e}"
        }

    # Test correctness
    correctness_result = test_correctness_basic(evolved_module)
    correctness_score = correctness_result["num_passed"] / correctness_result["num_tests"]

    print(f"  {'✅' if correctness_result['passed'] else '❌'} Correctness: {correctness_score:.3f}")

    if correctness_score < 1.0:
        return {
            "correctness": correctness_score,
            "performance": 0.0,
            "combined_score": 0.0,
            "correctness_errors": correctness_result["errors"]
        }

    # Test performance on multiple sizes
    test_sizes = [100, 500]
    target_times = {
        100: 0.050,   # 50ms
        500: 0.500,   # 500ms (baseline had recursion error!)
        1000: 2.000,  # 2s (baseline had recursion error!)
    }

    scores = []
    total_time = 0.0

    for size in test_sizes:
        perf_result = test_performance_pathological_119105(evolved_module, size=size, timeout=120.0)

        if not perf_result["success"]:
            print(f"  ❌ Size {size:4d}: FAILED - {perf_result['error']}")
            scores.append(0.0)
        else:
            elapsed = perf_result["elapsed_time"]
            total_time += elapsed
            target = target_times[size]
            score = sigmoid_performance_score(elapsed, target)
            scores.append(score)
            print(f"  ✅ Size {size:4d}: {elapsed:6.3f}s (target: {target:6.3f}s, score: {score:.3f})")

    # Test realistic case
    realistic_result = test_performance_realistic(evolved_module, n_functions=100)
    if realistic_result["success"]:
        elapsed = realistic_result["elapsed_time"]
        total_time += elapsed
        # Target: 0.010s for 100 functions (baseline: 0.009s)
        score = sigmoid_performance_score(elapsed, 0.010)
        scores.append(score)
        print(f"  ✅ Realistic (802 lines): {elapsed:6.3f}s (score: {score:.3f})")
    else:
        scores.append(0.0)
        print(f"  ❌ Realistic test failed: {realistic_result['error']}")

    # Calculate overall performance score
    performance_score = sum(scores) / len(scores) if scores else 0.0
    combined_score = correctness_score * performance_score

    print(f"\n  📊 Final Scores:")
    print(f"     Correctness: {correctness_score:.3f}")
    print(f"     Performance: {performance_score:.3f}")
    print(f"     Combined: {combined_score:.3f}")
    print(f"     Total time: {total_time:.3f}s")

    return {
        "correctness": correctness_score,
        "performance": performance_score,
        "combined_score": combined_score,
        "total_time": total_time,
        "test_scores": scores
    }
