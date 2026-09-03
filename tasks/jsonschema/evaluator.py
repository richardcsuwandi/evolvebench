"""
Evaluator for JSONSchema Equality Checking Optimization

This evaluator tests both correctness and performance of the equality
checking functions from python-jsonschema.

CORRECTNESS TESTS:
- Bool/int distinction (False != 0, True != 1 in JSON Schema)
- Nested structure equality
- String comparisons
- Sequence comparisons
- Mapping comparisons
- Edge cases (empty containers, None, etc.)

PERFORMANCE TESTS:
- Benchmark on representative nested structures
- Simulate workload similar to 16MB SBOM validation
- Measure comparisons per second

SCORING:
- Correctness: 70% weight (must be 100% for valid solution)
- Performance: 30% weight (sigmoid scoring vs baseline)
"""

import sys
import time
import copy
import json
import math
import os
import types

# Global cache for baseline performance
_BASELINE_CACHE = None


def get_baseline_performance():
    """
    Measure the baseline performance from the jsonschema.py.

    This is cached globally so we only measure once, then all evolved programs
    are compared against this fixed baseline.
    """
    global _BASELINE_CACHE

    if _BASELINE_CACHE is not None:
        return _BASELINE_CACHE

    # Load the initial program (jsonschema.py in the same directory)
    initial_program_path = os.path.join(os.path.dirname(__file__), 'jsonschema.py')

    if not os.path.exists(initial_program_path):
        # Try current directory
        initial_program_path = 'jsonschema.py'

    if not os.path.exists(initial_program_path):
        # If we can't find it, use a reasonable default
        print("WARNING: Could not find jsonschema.py, using default baseline")
        _BASELINE_CACHE = 0.3  # Conservative baseline time
        return _BASELINE_CACHE

    try:
        with open(initial_program_path, 'r') as f:
            code = f.read()

        namespace = {}
        exec(code, namespace)
        baseline_equal = namespace['equal']

        # Create test data
        test_data = create_sbom_like_data(depth=5, breadth=10)
        test_data_copy = copy.deepcopy(test_data)

        # Warmup
        for _ in range(5):
            baseline_equal(test_data, test_data_copy)

        # Benchmark
        num_iterations = 100
        start = time.time()
        for _ in range(num_iterations):
            baseline_equal(test_data, test_data_copy)
        baseline_time = time.time() - start

        _BASELINE_CACHE = baseline_time
        print(f"✓ Measured baseline performance: {baseline_time:.4f}s for {num_iterations} iterations")
        return baseline_time

    except Exception as e:
        print(f"WARNING: Failed to measure baseline: {e}, using default")
        _BASELINE_CACHE = 0.3
        return _BASELINE_CACHE


def evaluate(program_code):
    """
    Main evaluation function called by OpenEvolve.

    Args:
        program_code (str): The program code to evaluate (can be a file path or code string)

    Returns:
        dict: Evaluation results with correctness, performance, and combined scores
    """
    import os
    import types

    # If program_code is a file path, read the file
    if os.path.exists(program_code):
        with open(program_code, 'r') as f:
            code = f.read()
    else:
        code = program_code

    # Execute the program code to get the functions
    try:
        namespace = {}
        exec(code, namespace)
        program = types.SimpleNamespace(**namespace)
    except SyntaxError as e:
        return {
            'correctness': 0.0,
            'performance': 0.0,
            'combined_score': 0.0,
            'error': f'Syntax error: {str(e)}'
        }
    except Exception as e:
        return {
            'correctness': 0.0,
            'performance': 0.0,
            'combined_score': 0.0,
            'error': f'Failed to execute program: {str(e)}'
        }

    # Get the functions we need to test
    try:
        equal = program.equal
        _mapping_equal = program._mapping_equal
        _sequence_equal = program._sequence_equal
    except AttributeError as e:
        return {
            'correctness': 0.0,
            'performance': 0.0,
            'combined_score': 0.0,
            'error': f'Missing required function: {str(e)}'
        }

    # Run correctness tests
    correctness_result = test_correctness(equal, _mapping_equal, _sequence_equal)

    if correctness_result['score'] < 1.0:
        # If correctness is not 100%, return immediately
        return {
            'correctness': correctness_result['score'],
            'performance': 0.0,
            'combined_score': correctness_result['score'] * 0.7,
            'failed_tests': correctness_result.get('failed_tests', []),
            'error': 'Correctness tests failed'
        }

    # Run performance tests
    performance_result = test_performance(equal)

    # Calculate combined score
    # Correctness: 70%, Performance: 30%
    combined_score = (correctness_result['score'] * 0.7 +
                     performance_result['score'] * 0.3)

    return {
        'correctness': correctness_result['score'],
        'performance': performance_result['score'],
        'combined_score': combined_score,
        'performance_details': performance_result,
        'artifacts': {
            'baseline_time': performance_result.get('baseline_time'),
            'optimized_time': performance_result.get('optimized_time'),
            'speedup': performance_result.get('speedup')
        }
    }


def test_correctness(equal, _mapping_equal, _sequence_equal):
    """
    Test that the equality functions maintain correct semantics.

    Critical: JSON Schema requires False != 0 and True != 1
    """
    tests = []
    failed_tests = []

    # Test 1: Bool/int distinction
    tests.append(('bool_int_false_0', not equal(False, 0), 'False should not equal 0'))
    tests.append(('bool_int_true_1', not equal(True, 1), 'True should not equal 1'))
    tests.append(('bool_false_false', equal(False, False), 'False should equal False'))
    tests.append(('bool_true_true', equal(True, True), 'True should equal True'))
    tests.append(('int_0_0', equal(0, 0), '0 should equal 0'))
    tests.append(('int_1_1', equal(1, 1), '1 should equal 1'))

    # Test 2: Identity optimization
    obj = {'a': 1}
    tests.append(('identity', equal(obj, obj), 'Object should equal itself'))

    # Test 3: String comparisons
    tests.append(('string_equal', equal('hello', 'hello'), 'Equal strings'))
    tests.append(('string_not_equal', not equal('hello', 'world'), 'Different strings'))
    tests.append(('string_vs_int', not equal('1', 1), 'String vs int'))

    # Test 4: Simple sequences
    tests.append(('list_equal', equal([1, 2, 3], [1, 2, 3]), 'Equal lists'))
    tests.append(('list_not_equal', not equal([1, 2, 3], [1, 2, 4]), 'Different lists'))
    tests.append(('list_length', not equal([1, 2], [1, 2, 3]), 'Different length lists'))

    # Test 5: Simple mappings
    tests.append(('dict_equal', equal({'a': 1, 'b': 2}, {'a': 1, 'b': 2}), 'Equal dicts'))
    tests.append(('dict_not_equal', not equal({'a': 1}, {'a': 2}), 'Different values'))
    tests.append(('dict_keys', not equal({'a': 1}, {'b': 1}), 'Different keys'))
    tests.append(('dict_length', not equal({'a': 1}, {'a': 1, 'b': 2}), 'Different lengths'))

    # Test 6: Nested structures
    nested1 = {'a': [1, 2, {'b': 3}]}
    nested2 = {'a': [1, 2, {'b': 3}]}
    nested3 = {'a': [1, 2, {'b': 4}]}
    tests.append(('nested_equal', equal(nested1, nested2), 'Equal nested structures'))
    tests.append(('nested_not_equal', not equal(nested1, nested3), 'Different nested structures'))

    # Test 7: Boolean edge cases in nested structures
    tests.append(('nested_bool_1', not equal({'a': [False]}, {'a': [0]}), 'Nested False vs 0'))
    tests.append(('nested_bool_2', not equal({'a': [True]}, {'a': [1]}), 'Nested True vs 1'))

    # Test 8: Empty containers
    tests.append(('empty_list', equal([], []), 'Empty lists'))
    tests.append(('empty_dict', equal({}, {}), 'Empty dicts'))
    tests.append(('empty_vs_nonempty_list', not equal([], [1]), 'Empty vs non-empty list'))
    tests.append(('empty_vs_nonempty_dict', not equal({}, {'a': 1}), 'Empty vs non-empty dict'))

    # Test 9: Mixed types
    tests.append(('list_vs_dict', not equal([1, 2], {'0': 1, '1': 2}), 'List vs dict'))
    tests.append(('int_vs_float', equal(1, 1.0), 'Int vs float with same value'))

    # Test 10: Complex real-world-like structure
    schema_like = {
        'type': 'object',
        'properties': {
            'name': {'type': 'string'},
            'age': {'type': 'integer', 'minimum': 0},
            'address': {
                'type': 'object',
                'properties': {
                    'street': {'type': 'string'},
                    'city': {'type': 'string'}
                }
            }
        },
        'required': ['name', 'age']
    }
    schema_like_copy = copy.deepcopy(schema_like)
    tests.append(('schema_like', equal(schema_like, schema_like_copy), 'Schema-like structure'))

    # Count passed tests
    passed = 0
    for name, result, description in tests:
        if result:
            passed += 1
        else:
            failed_tests.append({'name': name, 'description': description})

    score = passed / len(tests) if tests else 0.0

    return {
        'score': score,
        'passed': passed,
        'total': len(tests),
        'failed_tests': failed_tests
    }


def test_performance(equal):
    """
    Test performance on representative workloads.

    Simulates the type of nested structure comparisons that happen during
    JSON Schema validation of large documents like 16MB SBOMs.

    Baseline is the original implementation performance.
    We use sigmoid scoring: significant speedup gets high score, slowdowns get low score.
    """
    # Create test data that simulates SBOM-like structure
    test_data = create_sbom_like_data(depth=5, breadth=10)
    test_data_copy = copy.deepcopy(test_data)

    # Warmup
    for _ in range(5):
        equal(test_data, test_data_copy)

    # Benchmark the evolved implementation
    num_iterations = 100
    start = time.time()
    for _ in range(num_iterations):
        result = equal(test_data, test_data_copy)
    optimized_time = time.time() - start

    # Get the TRUE baseline from jsonschema.py
    baseline_time = get_baseline_performance()

    # Calculate speedup
    speedup = baseline_time / optimized_time if optimized_time > 0 else 1.0

    # Sigmoid scoring: 1.0 = no change, >1.0 = speedup, <1.0 = slowdown
    # Score function: 1 / (1 + exp(-5 * (speedup - 1)))
    # This gives:
    # - speedup 1.0x = score 0.5
    # - speedup 1.2x = score 0.73
    # - speedup 1.5x = score 0.92
    # - speedup 2.0x = score 0.99
    # - speedup 0.8x = score 0.27
    # - speedup 0.5x = score 0.01
    relative_speedup = speedup - 1.0
    score = 1.0 / (1.0 + math.exp(-5 * relative_speedup))

    return {
        'score': score,
        'baseline_time': baseline_time,
        'optimized_time': optimized_time,
        'speedup': speedup,
        'num_iterations': num_iterations
    }


def create_sbom_like_data(depth=5, breadth=10):
    """
    Create a nested structure similar to an SBOM (Software Bill of Materials).

    Args:
        depth: Maximum nesting depth
        breadth: Number of items at each level

    Returns:
        dict: A deeply nested structure resembling an SBOM
    """
    if depth == 0:
        return {
            'name': 'component',
            'version': '1.0.0',
            'type': 'library',
            'licenses': ['MIT', 'Apache-2.0']
        }

    components = []
    for i in range(breadth):
        components.append({
            'name': f'component-{depth}-{i}',
            'version': f'{depth}.{i}.0',
            'type': 'library',
            'licenses': ['MIT'],
            'dependencies': create_sbom_like_data(depth - 1, max(2, breadth // 2))
        })

    return {
        'bomFormat': 'CycloneDX',
        'specVersion': '1.4',
        'version': 1,
        'metadata': {
            'timestamp': '2024-01-01T00:00:00Z',
            'tools': [{'vendor': 'test', 'name': 'test-tool'}]
        },
        'components': components
    }


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: python evaluator.py <program_path>')
        sys.exit(1)

    program_path = sys.argv[1]
    result = evaluate(program_path)

    print('=' * 70)
    print('JSONSchema Equality Checking - Evaluation Results')
    print('=' * 70)
    print()
    print(f'Correctness: {result["correctness"]*100:.1f}%')
    print(f'Performance: {result["performance"]*100:.1f}%')
    print(f'Combined Score: {result["combined_score"]*100:.1f}%')
    print()

    if 'performance_details' in result:
        details = result['performance_details']
        print('Performance Details:')
        print(f'  Baseline time:  {details.get("baseline_time", 0):.4f}s')
        print(f'  Optimized time: {details.get("optimized_time", 0):.4f}s')
        print(f'  Speedup:        {details.get("speedup", 0):.2f}x')
        print(f'  Iterations:     {details.get("num_iterations", 0)}')
        print()

    if 'failed_tests' in result and result['failed_tests']:
        print('Failed Correctness Tests:')
        for test in result['failed_tests']:
            print(f'  - {test["name"]}: {test["description"]}')
        print()

    if 'error' in result:
        print(f'Error: {result["error"]}')
        print()

    # Print JSON for OpenEvolve
    print('JSON Output for OpenEvolve:')
    print(json.dumps({
        'correctness': result['correctness'],
        'performance': result['performance'],
        'combined_score': result['combined_score']
    }, indent=2))
