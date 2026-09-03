"""
Evaluator for SymPy Min/Max _find_localzeros Optimization
==========================================================

This evaluator tests both correctness and performance of the _find_localzeros algorithm.

Correctness Tests:
1. Basic ordering: correct minimal/maximal elements identified
2. Duplicate handling: identical elements handled correctly
3. Set size: result has expected number of elements
4. Identity preservation: unrelated symbols remain in result

Performance Tests:
- Construction time for Min(*symbols('x:N')) for various N
- Target: <0.2s for 50 symbols (significantly better than current)
"""

import time
import math
from typing import Dict, Any
from sympy import symbols, Symbol


def sigmoid_performance_score(actual_time: float, target_time: float, sensitivity: float = 5.0) -> float:
    """
    Convert performance timing to 0-1 score using sigmoid.

    Args:
        actual_time: Measured execution time
        target_time: Ideal target time
        sensitivity: How quickly score drops (higher = more sensitive)

    Returns:
        Score from 0 to 1 (1 is best)
    """
    if actual_time <= 0:
        return 0.0
    relative_slowdown = (actual_time - target_time) / target_time
    return 1 / (1 + math.exp(sensitivity * relative_slowdown))


def test_correctness_basic(program_module) -> Dict[str, Any]:
    """
    Test basic correctness of _find_localzeros.

    Includes tests for:
    - Plain symbols (unrelated, all should be in localzeros)
    - Symbols with assumptions (negative, positive, etc.)
    - Duplicates and edge cases
    """
    tests_passed = 0
    total_tests = 0

    try:
        # Test 1: Small set of plain symbols
        total_tests += 1
        syms = symbols('a b c')
        result = program_module.Min._find_localzeros(syms)
        if len(result) == 3 and all(s in result for s in syms):
            tests_passed += 1

        # Test 2: Larger set (10 plain symbols)
        total_tests += 1
        syms = symbols('x:10')
        result = program_module.Min._find_localzeros(syms)
        if len(result) == 10 and all(s in result for s in syms):
            tests_passed += 1

        # Test 3: With duplicates
        total_tests += 1
        x, y = symbols('x y')
        syms = [x, y, x, y]  # duplicates
        result = program_module.Min._find_localzeros(syms)
        # Should have 2 elements (x and y), duplicates removed
        if len(result) == 2 and x in result and y in result:
            tests_passed += 1

        # Test 4: Single symbol
        total_tests += 1
        x = symbols('x')
        result = program_module.Min._find_localzeros([x])
        if len(result) == 1 and x in result:
            tests_passed += 1

        # Test 5: Empty set
        total_tests += 1
        result = program_module.Min._find_localzeros([])
        if len(result) == 0:
            tests_passed += 1

        # Test 6: Negative and nonnegative symbols (CRITICAL - caught regression)
        total_tests += 1
        n = Symbol('n', negative=True)
        nn = Symbol('nn', nonnegative=True)
        result = program_module.Min._find_localzeros([n, nn])
        # n is negative (< 0), nn is nonnegative (>= 0), so n < nn
        # Min should return only n
        if len(result) == 1 and n in result and nn not in result:
            tests_passed += 1

        # Test 7: Positive and negative symbols
        total_tests += 1
        n = Symbol('n', negative=True)
        p = Symbol('p', positive=True)
        result = program_module.Min._find_localzeros([n, p])
        # n < 0 < p, so Min should return only n
        if len(result) == 1 and n in result and p not in result:
            tests_passed += 1

        # Test 8: Nonpositive and positive symbols
        total_tests += 1
        np = Symbol('np', nonpositive=True)
        p = Symbol('p', positive=True)
        result = program_module.Min._find_localzeros([np, p])
        # np <= 0 < p, so Min should return only np
        if len(result) == 1 and np in result and p not in result:
            tests_passed += 1

        # Test 9: Multiple symbols with mixed assumptions
        total_tests += 1
        n = Symbol('n', negative=True)
        nn = Symbol('nn', nonnegative=True)
        p = Symbol('p', positive=True)
        result = program_module.Min._find_localzeros([p, n, nn])
        # n < 0 <= nn, p > 0, so Min should return only n
        if len(result) == 1 and n in result:
            tests_passed += 1

        # Test 10: Plain symbols mixed with assumption symbols
        total_tests += 1
        x = Symbol('x')  # plain symbol
        n = Symbol('n', negative=True)
        result = program_module.Min._find_localzeros([x, n])
        # x is unknown, n < 0
        # Both could be minimal (x could be < n), so both should be returned
        if len(result) == 2 and x in result and n in result:
            tests_passed += 1

        return {
            'success': True,
            'num_passed': tests_passed,
            'num_tests': total_tests,
            'score': tests_passed / total_tests if total_tests > 0 else 0.0
        }

    except Exception as e:
        return {
            'success': False,
            'num_passed': tests_passed,
            'num_tests': total_tests,
            'score': 0.0,
            'error': str(e)
        }


def test_performance_construction(program_module) -> Dict[str, Any]:
    """
    Test performance of _find_localzeros for increasing symbol counts.

    Measures construction time and compares against targets.
    """
    try:
        # Test different sizes
        test_configs = [
            {'size': 10, 'target_time': 0.05},   # 50ms
            {'size': 20, 'target_time': 0.08},   # 80ms
            {'size': 30, 'target_time': 0.12},   # 120ms
            {'size': 40, 'target_time': 0.16},   # 160ms
            {'size': 50, 'target_time': 0.20},   # 200ms (target)
        ]

        total_score = 0.0
        results = []

        for config in test_configs:
            size = config['size']
            target = config['target_time']

            syms = symbols(f'x:{size}')

            # Warmup
            _ = program_module.Min._find_localzeros(syms)

            # Actual timing (3 runs, take median)
            times = []
            for _ in range(3):
                start = time.time()
                result = program_module.Min._find_localzeros(syms)
                elapsed = time.time() - start
                times.append(elapsed)

            median_time = sorted(times)[1]
            score = sigmoid_performance_score(median_time, target, sensitivity=5.0)
            total_score += score

            results.append({
                'size': size,
                'time': median_time,
                'target': target,
                'score': score
            })

        avg_score = total_score / len(test_configs)

        return {
            'success': True,
            'score': avg_score,
            'results': results,
            'time_50_symbols': results[-1]['time']  # Most important metric
        }

    except Exception as e:
        return {
            'success': False,
            'score': 0.0,
            'error': str(e)
        }


def evaluate(program_code: str) -> Dict[str, Any]:
    """
    Main evaluation function for OpenEvolve.

    Tests both correctness (0.5 weight) and performance (0.5 weight).

    Args:
        program_code: Program code as string (can be full file or block content)

    Returns:
        Dictionary with keys:
        - correctness: score 0-1
        - performance: score 0-1
        - combined_score: weighted average
        - details: additional information
    """
    import types

    # Handle both file paths and program code strings
    if isinstance(program_code, str) and '\n' in program_code and not program_code.strip().startswith('"""'):
        # This looks like program code, not a file path
        code = program_code
    else:
        # This might be a file path, try to read it
        try:
            with open(program_code, 'r') as f:
                code = f.read()
        except:
            # If it fails, treat as program code
            code = program_code

    # Execute the program code
    namespace = {}
    try:
        exec(code, namespace)
    except IndentationError as e:
        # Handle indentation errors that occur when OpenEvolve passes block content
        # Try to fix by removing leading indentation
        try:
            lines = code.split('\n')
            # Find the minimum indentation (excluding empty lines)
            min_indent = float('inf')
            for line in lines:
                if line.strip():  # Skip empty lines
                    indent = len(line) - len(line.lstrip())
                    min_indent = min(min_indent, indent)
            
            if min_indent > 0 and min_indent < float('inf'):
                # Remove the minimum indentation from all lines
                fixed_lines = []
                for line in lines:
                    if line.strip():  # Skip empty lines
                        fixed_lines.append(line[min_indent:])
                    else:
                        fixed_lines.append(line)
                
                fixed_code = '\n'.join(fixed_lines)
                exec(fixed_code, namespace)
            else:
                raise e
        except Exception as e2:
            return {
                'correctness': 0.0,
                'performance': 0.0,
                'combined_score': 0.0,
                'error': f'Execution error: {str(e)} (indentation fix failed: {str(e2)})'
            }
    except Exception as e:
        return {
            'correctness': 0.0,
            'performance': 0.0,
            'combined_score': 0.0,
            'error': f'Execution error: {str(e)}'
        }

    # Create module-like object
    program_module = types.SimpleNamespace(**namespace)

    # Stage 1: Correctness Tests
    correctness_result = test_correctness_basic(program_module)
    if not correctness_result['success']:
        return {
            'correctness': 0.0,
            'performance': 0.0,
            'combined_score': 0.0,
            'error': correctness_result.get('error', 'Correctness test failed')
        }

    correctness_score = correctness_result['score']

    # If correctness is too low, don't bother with performance
    if correctness_score < 0.6:
        return {
            'correctness': correctness_score,
            'performance': 0.0,
            'combined_score': correctness_score * 0.5,
            'details': 'Correctness too low, skipped performance tests'
        }

    # Stage 2: Performance Tests
    performance_result = test_performance_construction(program_module)
    if not performance_result['success']:
        return {
            'correctness': correctness_score,
            'performance': 0.0,
            'combined_score': correctness_score * 0.5,
            'error': performance_result.get('error', 'Performance test failed')
        }

    performance_score = performance_result['score']

    # Compute combined score
    combined_score = (correctness_score * 0.5) + (performance_score * 0.5)

    return {
        'correctness': correctness_score,
        'performance': performance_score,
        'combined_score': combined_score,
        'time_50_symbols': performance_result.get('time_50_symbols', None),
        'correctness_details': correctness_result,
        'performance_details': performance_result
    }


if __name__ == "__main__":
    # Test the evaluator with the initial program
    print("Testing evaluator with sympy_implementation.py...")
    print("=" * 60)

    # Read the full file content
    with open('sympy_implementation.py', 'r') as f:
        program_code = f.read()
    
    result = evaluate(program_code)

    print(f"Correctness Score: {result['correctness']:.3f}")
    print(f"Performance Score: {result['performance']:.3f}")
    print(f"Combined Score:    {result['combined_score']:.3f}")
    print()
    print(f"Time for 50 symbols: {result.get('time_50_symbols', 'N/A'):.4f}s")
    print()
    print("Correctness Details:")
    details = result.get('correctness_details', {})
    print(f"  Tests Passed: {details.get('num_passed', 0)}/{details.get('num_tests', 0)}")
    print()
    print("Performance Details:")
    perf_results = result.get('performance_details', {}).get('results', [])
    for r in perf_results:
        print(f"  {r['size']:2d} symbols: {r['time']:.4f}s (target: {r['target']:.4f}s, score: {r['score']:.3f})")
