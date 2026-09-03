"""
JSONSchema Equality Checking - Baseline Implementation
================================================================

This file provides the baseline implementation of JSON Schema's equality
checking functions from python-jsonschema library.

PERFORMANCE ISSUE:
Issue #1304: Python 3.12 shows 33% performance degradation compared to 3.10
- Profiling shows _mapping_equal() consumes 32.6% of total execution time
- Called 56.4 million times during validation of 16MB SBOM file
- O(n*m) complexity for nested structures leads to billions of comparisons

KEY CHALLENGES:
1. Must handle Python's bool/int type hierarchy (False != 0, True != 1 in JSON Schema)
2. Recursive comparisons through nested structures (mappings, sequences)
3. Must preserve exact equality semantics for JSON Schema compliance
4. Called millions of times - even small optimizations matter

OPPORTUNITY FOR IMPROVEMENT:
- Memoization/caching of comparison results
- Hash-based quick rejection before deep comparison
- Type-based short-circuit evaluation
- Algorithmic restructuring of comparison strategy
- Early termination strategies

Source: https://github.com/python-jsonschema/jsonschema/blob/main/jsonschema/_utils.py
"""

from collections.abc import Mapping, Sequence


def unbool(element, true=object(), false=object()):
    """
    A hack to make True and 1 and False and 0 unique for equality checking.

    This is necessary because in Python:
    - True == 1 evaluates to True
    - False == 0 evaluates to True

    But in JSON Schema, these are distinct values and should not be equal.
    """
    if element is True:
        return true
    elif element is False:
        return false
    return element


# EVOLVE-BLOCK-START
def _mapping_equal(one, two):
    """
    Check if two mappings are equal using the semantics of `equal`.

    This is the PRIMARY BOTTLENECK identified in profiling:
    - 32.6% of total execution time
    - O(n * m) where n is number of keys, m is average value complexity
    - No caching or memoization of results
    - Recursive calls for nested structures multiply the cost

    BASELINE PERFORMANCE:
    - Python 3.10: 27.8s cumulative time (56.4M calls)
    - Python 3.12: 37.2s cumulative time (56.4M calls)
    - 33% regression in Python 3.12

    OPPORTUNITY:
    This function has significant room for optimization through:
    - Caching comparison results
    - Hash-based quick rejection
    - Structural comparison before value comparison
    - Type-aware early termination
    """
    if len(one) != len(two):
        return False
    return all(
        key in two and equal(value, two[key])
        for key, value in one.items()
    )


def _sequence_equal(one, two):
    """
    Check if two sequences are equal using the semantics of `equal`.

    Part of the recursive equality checking system.
    Also shows up in profiling but less critical than _mapping_equal.
    """
    if len(one) != len(two):
        return False
    return all(equal(i, j) for i, j in zip(one, two))


def equal(one, two):
    """
    Check if two things are equal evading some Python type hierarchy semantics.

    Specifically in JSON Schema, evade `bool` inheriting from `int`,
    recursing into sequences to do the same.

    This is the main entry point for equality checking and dispatches to
    specialized functions based on type. The isinstance() checks themselves
    show up as bottlenecks in Python 3.12 profiling.

    PROFILING DATA:
    - Total cumulative time: 33.2s (Python 3.12) vs 28.4s (Python 3.10)
    - isinstance() calls: 34.9s (Python 3.12) vs 23.1s (Python 3.10)

    The cascading isinstance() checks are expensive when called millions of times.
    """
    if one is two:
        return True
    if isinstance(one, str) or isinstance(two, str):
        return one == two
    if isinstance(one, Sequence) and isinstance(two, Sequence):
        return _sequence_equal(one, two)
    if isinstance(one, Mapping) and isinstance(two, Mapping):
        return _mapping_equal(one, two)
    return unbool(one) == unbool(two)
# EVOLVE-BLOCK-END


def benchmark_equality_checking(data, num_comparisons=1000):
    """
    Benchmark the equality checking functions.

    This simulates the type of comparisons that happen during JSON Schema
    validation of large, nested documents.
    """
    import time
    import copy

    # Create a deep copy for comparison
    data_copy = copy.deepcopy(data)

    # Warmup
    for _ in range(10):
        equal(data, data_copy)

    # Benchmark
    start = time.time()
    for _ in range(num_comparisons):
        result = equal(data, data_copy)
    elapsed = time.time() - start

    return {
        'num_comparisons': num_comparisons,
        'elapsed_time': elapsed,
        'comparisons_per_second': num_comparisons / elapsed if elapsed > 0 else float('inf'),
        'result': result
    }


if __name__ == "__main__":
    print("JSONSchema Equality Checking - Baseline Implementation")
    print("=" * 70)
    print()
    print("Testing baseline performance on nested structures...")
    print()

    # Test with increasingly complex nested structures
    test_cases = [
        {
            'name': 'Small nested dict',
            'data': {'a': 1, 'b': 2, 'c': {'d': 3, 'e': 4}},
            'comparisons': 10000
        },
        {
            'name': 'Medium nested structure',
            'data': {
                'metadata': {'version': '1.0', 'author': 'test'},
                'data': [1, 2, 3, 4, 5],
                'nested': {'a': {'b': {'c': {'d': 1}}}}
            },
            'comparisons': 5000
        },
        {
            'name': 'Large flat dict',
            'data': {f'key_{i}': i for i in range(100)},
            'comparisons': 1000
        }
    ]

    print(f"{'Test Case':<25} {'Comparisons':>12} {'Time':>10} {'Comp/sec':>12}")
    print("-" * 65)

    for test in test_cases:
        result = benchmark_equality_checking(test['data'], test['comparisons'])
        print(f"{test['name']:<25} {result['num_comparisons']:>12} "
              f"{result['elapsed_time']:>9.4f}s {result['comparisons_per_second']:>11.0f}")

    print()
    print("Baseline established. Room for optimization:")
    print("- Memoization of comparison results")
    print("- Hash-based quick rejection")
    print("- Type-aware early termination")
    print("- Structural pre-checks")
