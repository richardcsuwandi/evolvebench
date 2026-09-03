"""
Evaluator for Pandas Rolling Rank Rediscovery Experiment
=========================================================
This evaluator validates evolved rolling rank implementations against
pandas' official implementation and benchmarks performance.
Evaluation Strategy:
1. Correctness: Compare results against pandas.Series.rolling().rank()
2. Performance: Measure execution time on various input sizes
3. Robustness: Test edge cases (NaN, duplicates, small windows)
"""

import sys
import time
import numpy as np
import traceback


def evaluate(program_path: str) -> dict:
    """
    Evaluate a rolling rank implementation.
    Args:
        program_path: Path to the program file to evaluate
    Returns:
        dict with keys:
            - correctness: float in [0, 1]
            - performance: float in [0, 1] (higher is better)
            - artifacts: dict with test details
    """
    try:
        # Import pandas for ground truth
        import pandas as pd

        # Read the program from the file
        with open(program_path, 'r') as f:
            program_str = f.read()

        # Execute the program to get the RollingRank class
        namespace = {}
        exec(program_str, namespace)
        RollingRank = namespace.get('RollingRank')

        if RollingRank is None:
            return {
                'correctness': 0.0,
                'performance': 0.0,
                'artifacts': {'error': 'RollingRank class not found'}
            }

        # ==================================================================
        # STAGE 1: Quick Correctness Tests
        # ==================================================================

        correctness_tests = []

        # Test 1: Simple ascending sequence
        test_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        window_size = 3
        roller = RollingRank(window_size=window_size, method='average', ascending=True)
        result = roller.compute(test_data)
        expected = pd.Series(test_data).rolling(window_size).rank(method='average').values
        correctness_tests.append({
            'name': 'simple_sequence',
            'passed': np.allclose(result, expected, equal_nan=True),
            'result': result.tolist(),
            'expected': expected.tolist()
        })

        # Test 2: With duplicates
        test_data = np.array([1.0, 2.0, 2.0, 3.0, 2.0])
        roller = RollingRank(window_size=3, method='average', ascending=True)
        result = roller.compute(test_data)
        expected = pd.Series(test_data).rolling(3).rank(method='average').values
        correctness_tests.append({
            'name': 'duplicates',
            'passed': np.allclose(result, expected, equal_nan=True),
            'result': result.tolist(),
            'expected': expected.tolist()
        })

        # Test 3: With NaN values - simpler case
        test_data = np.array([1.0, 2.0, 3.0, np.nan, 5.0])
        roller = RollingRank(window_size=3, method='average', ascending=True)
        result = roller.compute(test_data)
        expected = pd.Series(test_data).rolling(3).rank(method='average').values
        correctness_tests.append({
            'name': 'with_nan',
            'passed': np.allclose(result, expected, equal_nan=True),
            'result': result.tolist(),
            'expected': expected.tolist()
        })

        # Test 4: Method='min'
        test_data = np.array([3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0])
        roller = RollingRank(window_size=4, method='min', ascending=True)
        result = roller.compute(test_data)
        expected = pd.Series(test_data).rolling(4).rank(method='min').values
        correctness_tests.append({
            'name': 'method_min',
            'passed': np.allclose(result, expected, equal_nan=True),
            'result': result.tolist(),
            'expected': expected.tolist()
        })

        # Test 5: Method='max'
        test_data = np.array([3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0])
        roller = RollingRank(window_size=4, method='max', ascending=True)
        result = roller.compute(test_data)
        expected = pd.Series(test_data).rolling(4).rank(method='max').values
        correctness_tests.append({
            'name': 'method_max',
            'passed': np.allclose(result, expected, equal_nan=True),
            'result': result.tolist(),
            'expected': expected.tolist()
        })

        # Test 6: Descending order
        test_data = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        roller = RollingRank(window_size=3, method='average', ascending=False)
        result = roller.compute(test_data)
        expected = pd.Series(test_data).rolling(3).rank(method='average', ascending=False).values
        correctness_tests.append({
            'name': 'descending',
            'passed': np.allclose(result, expected, equal_nan=True),
            'result': result.tolist(),
            'expected': expected.tolist()
        })

        # Test 7: Percentile mode
        test_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        roller = RollingRank(window_size=4, method='average', ascending=True, pct=True)
        result = roller.compute(test_data)
        expected = pd.Series(test_data).rolling(4).rank(method='average', pct=True).values
        correctness_tests.append({
            'name': 'percentile',
            'passed': np.allclose(result, expected, equal_nan=True),
            'result': result.tolist(),
            'expected': expected.tolist()
        })

        # Test 8: Larger dataset
        np.random.seed(42)
        test_data = np.random.randn(100)
        roller = RollingRank(window_size=10, method='average', ascending=True)
        result = roller.compute(test_data)
        expected = pd.Series(test_data).rolling(10).rank(method='average').values
        correctness_tests.append({
            'name': 'large_random',
            'passed': np.allclose(result, expected, equal_nan=True, rtol=1e-10),
            'max_diff': np.max(np.abs(result - expected)) if not np.all(np.isnan(result)) else 0.0
        })

        # Calculate correctness score
        passed_count = sum(1 for test in correctness_tests if test['passed'])
        correctness_score = passed_count / len(correctness_tests)

        if correctness_score < 1.0:
            # If correctness fails, return immediately
            # Still calculate combined_score for evolution guidance
            combined_score = 0.7 * correctness_score + 0.3 * 0.0
            return {
                'correctness': correctness_score,
                'performance': 0.0,
                'combined_score': combined_score,
                'artifacts': {
                    'correctness_tests': correctness_tests,
                    'note': 'Correctness tests failed - performance not measured'
                }
            }

        # ==================================================================
        # STAGE 2: Performance Benchmarking
        # ==================================================================

        # Benchmark configurations
        benchmark_configs = [
            {'n': 1000, 'window': 50},
            {'n': 5000, 'window': 50},
            {'n': 10000, 'window': 50},
        ]

        performance_results = []

        for config in benchmark_configs:
            n = config['n']
            window = config['window']

            # Generate test data
            np.random.seed(42)
            test_data = np.random.randn(n)

            # Time the evolved implementation
            roller = RollingRank(window_size=window, method='average', ascending=True)
            start = time.time()
            result = roller.compute(test_data)
            evolved_time = time.time() - start

            # Time pandas implementation (ground truth)
            start = time.time()
            expected = pd.Series(test_data).rolling(window).rank(method='average').values
            pandas_time = time.time() - start

            # Verify correctness on this dataset
            is_correct = np.allclose(result, expected, equal_nan=True, rtol=1e-10)

            performance_results.append({
                'n': n,
                'window': window,
                'evolved_time': evolved_time,
                'pandas_time': pandas_time,
                'speedup': pandas_time / evolved_time if evolved_time > 0 else 0.0,
                'correct': is_correct
            })

        # Calculate performance score
        # We want to be at least as fast as pandas (speedup >= 1.0)
        # Score based on average speedup, capped at 1.0 (we're not trying to beat pandas)
        avg_speedup = np.mean([r['speedup'] for r in performance_results])

        # Performance score: reward getting close to or matching pandas performance
        # 0.0 = 10x slower than pandas
        # 0.5 = 2x slower than pandas
        # 1.0 = as fast as or faster than pandas
        if avg_speedup >= 1.0:
            performance_score = 1.0
        elif avg_speedup >= 0.1:
            # Linear scale from 0.1x to 1.0x maps to 0.0 to 1.0
            performance_score = (avg_speedup - 0.1) / 0.9
        else:
            performance_score = 0.0

        # Calculate combined score (weighted average)
        # Correctness is critical, performance is secondary
        combined_score = 0.7 * correctness_score + 0.3 * performance_score

        return {
            'correctness': correctness_score,
            'performance': performance_score,
            'combined_score': combined_score,
            'artifacts': {
                'correctness_tests': correctness_tests,
                'performance_results': performance_results,
                'avg_speedup': avg_speedup,
                'target': 'Match or exceed pandas performance (speedup >= 1.0)'
            }
        }

    except Exception as e:
        return {
            'correctness': 0.0,
            'performance': 0.0,
            'artifacts': {
                'error': str(e),
                'traceback': traceback.format_exc()
            }
        }


if __name__ == "__main__":
    # Test the evaluator with the initial program
    with open('pandas_rolling_rank.py', 'r') as f:
        program_str = f.read()

    print("Testing Evaluator with Pandas Default Implementation")
    print("=" * 70)
    print()

    result = evaluate(program_str)

    print(f"Correctness: {result['correctness']:.2%}")
    print(f"Performance: {result['performance']:.2%}")
    print()

    if 'correctness_tests' in result['artifacts']:
        print("Correctness Tests:")
        for test in result['artifacts']['correctness_tests']:
            status = "✓" if test['passed'] else "✗"
            print(f"  {status} {test['name']}")
        print()

    if 'performance_results' in result['artifacts']:
        print("Performance Benchmarks:")
        print(f"{'n':>6} {'window':>6} {'Evolved':>10} {'Pandas':>10} {'Speedup':>8}")
        print("-" * 50)
        for r in result['artifacts']['performance_results']:
            print(f"{r['n']:6d} {r['window']:6d} {r['evolved_time']:9.4f}s {r['pandas_time']:9.4f}s {r['speedup']:7.2f}x")
        print()
        print(f"Average Speedup: {result['artifacts']['avg_speedup']:.2f}x")
        print(f"Target: Match pandas (1.0x) or better")
