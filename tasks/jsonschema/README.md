# Python-JSONSchema Equality Checking Optimization

This task demonstrates OpenEvolve's ability to discover type-aware optimization techniques for addressing Python 3.12 performance regressions in widely-used validation libraries.

## Background

[python-jsonschema](https://github.com/python-jsonschema/jsonschema) is the most widely-used JSON Schema validator for Python with **19.6k+ GitHub stars**. It's used in production systems for API validation, configuration validation, and critical applications across thousands of projects.

**Issue #1304**: Python 3.12 Performance Regression

### The Optimization Opportunity

Python-jsonschema's equality checking functions (`equal()`, `_mapping_equal()`, `_sequence_equal()`) experienced a **33% slowdown** when upgrading from Python 3.10 to 3.12.

**Source**: `jsonschema/_utils.py` lines 106-154

**Profiling Data** (16MB SBOM validation):

| Function | Python 3.10 | Python 3.12 | Regression |
|----------|-------------|-------------|------------|
| `_mapping_equal()` | 27.8s | 37.2s | +33.8% |
| `equal()` | 28.4s | 33.2s | +16.9% |
| `isinstance()` | 23.1s | 34.9s | +51.1% |

**Root Cause**: The equality functions are called **56.4 million times** during validation. Python 3.12's slower ABC (Abstract Base Class) implementation for `isinstance()` compounds to massive slowdowns.

**Key Constraint**: Must preserve JSON Schema semantics:
- `False != 0` (even though `False == 0` in Python)
- `True != 1` (even though `True == 1` in Python)
- This is why the `unbool()` helper exists

## The Optimization

OpenEvolve successfully discovered a **2.3-2.4x speedup** through type-aware dispatch optimization:

### Before (Baseline)

```python
def _mapping_equal(one, two):
    """O(n) comparison with generator expression"""
    if len(one) != len(two):
        return False
    return all(
        key in two and equal(value, two[key])
        for key, value in one.items()
    )

def _sequence_equal(one, two):
    """O(n) comparison with generator expression"""
    if len(one) != len(two):
        return False
    return all(equal(i, j) for i, j in zip(one, two))

def equal(one, two):
    """Recursive equality with cascading isinstance() checks"""
    if one is two:
        return True
    if isinstance(one, str) or isinstance(two, str):
        return one == two
    if isinstance(one, Sequence) and isinstance(two, Sequence):
        return _sequence_equal(one, two)
    if isinstance(one, Mapping) and isinstance(two, Mapping):
        return _mapping_equal(one, two)
    return unbool(one) == unbool(two)
```

### After (Optimized)

```python
def _mapping_equal(one, two):
    """Explicit loop for better short-circuiting"""
    if len(one) != len(two):
        return False

    for key, value in one.items():
        try:
            if not equal(value, two[key]):
                return False
        except KeyError:
            return False

    return True

def _sequence_equal(one, two):
    """Index-based iteration for better performance"""
    len_one = len(one)
    if len_one != len(two):
        return False

    for i in range(len_one):
        if not equal(one[i], two[i]):
            return False

    return True

def equal(one, two):
    """Type-aware fast path optimization"""
    if one is two:
        return True

    type_one = type(one)
    type_two = type(two)

    # Fast path for identical types
    if type_one is type_two:
        if type_one is str:
            return one == two
        if type_one is dict:
            return _mapping_equal(one, two)
        if type_one is list:
            return _sequence_equal(one, two)

        # Only use unbool for primitives, not sequences/mappings
        if not isinstance(one, (Sequence, Mapping)):
            return unbool(one) == unbool(two)

    # Handle type mismatches
    if type_one is str or type_two is str:
        return False

    if isinstance(one, Sequence) and isinstance(two, Sequence):
        return _sequence_equal(one, two)

    if isinstance(one, Mapping) and isinstance(two, Mapping):
        return _mapping_equal(one, two)

    return unbool(one) == unbool(two)
```

## Results

Based on comprehensive testing against python-jsonschema's official test suite:

| Metric | Value |
|--------|-------|
| **Correctness** | 100% (329/329 tests pass) |
| **Performance** | 2.3-2.4x speedup on nested structures |
| **Test Suite** | test_utils.py (26 tests) + test_validators.py (303 tests) |

### Key Optimizations Discovered

1. **Type Caching**: Store `type(one)` and `type(two)` once instead of repeated `isinstance()` calls
2. **Fast Path for Identical Types**: Use `type_one is type_two` for common cases (str, dict, list)
3. **Explicit Loops**: Replace generator expressions for better short-circuiting
4. **Try/Except for KeyError**: Faster than `key in dict` for dictionary lookups
5. **Index-Based Iteration**: `range(len)` instead of `zip()` for sequences
6. **Careful Sequence Handling**: Avoid `unbool()` for sequences to maintain correctness

### Algorithm Comparison

**Complexity**: Both O(n) for recursive structures, but with **drastically reduced constant factors**

**Key Innovation**: The evolved implementation minimizes expensive `isinstance()` calls by:
- Caching types once at the start
- Using `is` comparison for exact type matching (fast path)
- Only falling back to `isinstance()` for type hierarchy checks when necessary

## Verification Against Repository Tests

The optimized code was verified against python-jsonschema's complete test suite:

```
Test Suite: jsonschema/tests/test_utils.py + test_validators.py
Total Tests: 329
Passed: 329
Failed: 0
Success Rate: 100%
```

### Test Coverage

**test_utils.py (26 tests)**:
- ✅ Equal function tests
- ✅ Dictionary equality tests
- ✅ List equality tests
- ✅ Nested structure tests
- ✅ NaN handling tests
- ✅ None handling tests

**test_validators.py (303 tests)**:
- ✅ All validator implementations (Draft3-Draft202012)
- ✅ uniqueItems validation with sequences
- ✅ Complex nested validation scenarios
- ✅ Reference resolution tests
- ✅ Custom type checker tests

## Critical Bug Found and Fixed

During verification, OpenEvolve's initial solution had an over-optimization bug with `deque` handling:

**Bug**: Fast path used `unbool()` for all matching types, including sequences like `deque`:
```python
if type_one is type_two:
    # ... handle str, dict, list ...
    return unbool(one) == unbool(two)  # ❌ WRONG for deque([False]) vs deque([0])
```

**Fix**: Add guard to prevent sequences from using `unbool()`:
```python
if type_one is type_two:
    # ... handle str, dict, list ...
    if not isinstance(one, (Sequence, Mapping)):
        return unbool(one) == unbool(two)  # ✅ Only for primitives
```

This demonstrates OpenEvolve's evolution process combined with comprehensive testing catches edge cases.

## Files in This Task

- **`jsonschema.py`**: Baseline implementation from python-jsonschema
- **`best_program.py`**: Optimized version (corrected after deque fix)
- **`evaluator.py`**: Correctness and performance test harness
- **`config_handwritten.yaml`**: Configuration for handwritten evaluator
- **`config_llm_generated.yaml`**: Configuration for LLM-generated evaluator
- **`task.yaml`**: Task metadata and configuration

## Running the Evaluator

To test the optimization:

```bash
cd tasks/jsonschema

# Test baseline
python evaluator.py jsonschema.py

# Test optimized version
python evaluator.py best_program.py
```

Expected output:
```
======================================================================
JSONSchema Equality Checking - Evaluation Results
======================================================================

Correctness: 100.0%
Performance: 98.5%
Combined Score: 99.6%

Performance Details:
  Baseline time:  0.3034s
  Optimized time: 0.1224s
  Speedup:        2.48x
  Iterations:     100
```

## Running Evolution

To reproduce the optimization from scratch:

```bash
cd tasks/jsonschema

# Run evolution for 50 iterations
python ../../openevolve-run.py jsonschema.py evaluator.py \
  --config config_handwritten.yaml \
  --iterations 50
```

## Key Takeaways

1. **Python 3.12 regressions are real**: The slower `isinstance()` implementation created new optimization opportunities

2. **Type-aware dispatch wins**: Caching types and using exact type matching (`is`) before falling back to `isinstance()` provides massive speedups

3. **Constant factors matter**: Even with the same O(n) complexity, the evolved version is 2.3-2.4x faster

4. **Comprehensive testing is essential**: The deque bug was only caught by running against the full repository test suite

5. **Production-ready optimization**: 100% correctness maintained while achieving significant speedup

6. **Contribution-ready**: This optimization addresses Issue #1304 and could be submitted as a patch

## Production Impact

If integrated into python-jsonschema, this optimization would:
- Restore Python 3.12 performance to Python 3.10 levels (33% regression eliminated)
- Speed up validation by 2.3-2.4x for nested structures
- Reduce CI/CD pipeline times for projects using JSON Schema validation
- Improve API response times for systems using jsonschema
- Benefit thousands of downstream projects (OpenAPI, Kubernetes configs, SBOMs)

### Companies/Projects Using python-jsonschema

- OpenAPI/Swagger validation tools
- Kubernetes configuration validators
- SBOM generation (CycloneDX, SPDX)
- Configuration management systems
- API testing frameworks
- Cloud infrastructure tools

## Performance Analysis

### Why Type Caching Helps

The baseline implementation calls `isinstance()` multiple times per comparison:
1. `isinstance(one, str)` - Check if either is string
2. `isinstance(two, str)` - Check if either is string
3. `isinstance(one, Sequence)` - Check for sequence
4. `isinstance(two, Sequence)` - Check for sequence
5. `isinstance(one, Mapping)` - Check for mapping
6. `isinstance(two, Mapping)` - Check for mapping

**In Python 3.12**, each `isinstance()` with ABC is **~50% slower** than Python 3.10.

The optimized version:
1. `type_one = type(one)` - O(1) type lookup
2. `type_two = type(two)` - O(1) type lookup
3. `type_one is type_two` - O(1) pointer comparison
4. Falls back to `isinstance()` only when types differ

**Result**: For the common case (same types), we reduce 6 `isinstance()` calls to 2 `type()` calls + 1 `is` check.

### Benchmark Workload

The evaluator creates SBOM-like nested structures:
- Depth: 5 levels
- Breadth: 10 items per level
- Total structure: ~100,000 nested comparisons
- Represents realistic JSON Schema validation workloads

## Configuration Details

- **LLM**: Gemini 2.5 Flash Preview (via OpenRouter)
- **Population size**: 120 programs across multiple islands
- **Temperature**: 0.6 (moderate creativity for type dispatch solutions)
- **Evolution strategy**: Diff-based (only evolves code within EVOLVE-BLOCK markers)
- **Evaluation**: Correctness 70%, Performance 30%

## Related Issues

- **Issue #1304**: Python 3.12 performance regression (this task) - [Link](https://github.com/python-jsonschema/jsonschema/issues/1304)
- **Issue #853**: Previous 5x regression (partially fixed, room for improvement)
- **Issue #277**: Large file validation performance

## License

This optimization case study is based on python-jsonschema (MIT License). The
upstream license text is included at `LICENSE` in this task directory. EvolveBench's
own harness and task code are licensed separately — see the repository root [LICENSE](../../LICENSE).
