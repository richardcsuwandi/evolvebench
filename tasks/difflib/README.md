# Difflib Optimization Example

This example demonstrates OpenEvolve's ability to discover performance optimizations in Python's standard library `difflib` module.

## Background

Python's `difflib.Differ._fancy_replace` method has documented performance issues:
- **GitHub Issue #119105**: O(N³) complexity causing recursion errors on files with 500+ similar lines
- **GitHub Issue #6931**: "Dreadfully slow" performance on certain file comparisons

The method searches for similar lines when replacing one block with another, using a sliding WINDOW to find best matches. The original implementation had a subtle inefficiency: it called the expensive `ratio()` function twice in the matching loop.

## The Optimization

OpenEvolve discovered at **iteration 10** (out of 40) that the `ratio()` function was being called redundantly:

### Before (Initial Program)
```python
for i in arange:
    cruncher.set_seq1(a[i])
    if (crqr() > best_ratio
          and cqr() > best_ratio
          and cr() > best_ratio):        # ← Call 1
        best_i, best_j, best_ratio = i, j, cr()  # ← Call 2 (redundant!)
```

### After (Optimized Program)
```python
for i in arange:
    cruncher.set_seq1(a[i])
    if crqr() > best_ratio and cqr() > best_ratio:
        r = cr()                          # ← Single call
        if r > best_ratio:                # ← Reuse the result
            best_ratio = r
            best_i = i
```

## Results

Based on comprehensive testing (1,400 comparisons across 7 sizes with 200 iterations each):

- **Mean performance improvement**: 5.24% faster (1.055x speedup)
- **Improvement range**: 4.0% to 6.4% faster across different file sizes
- **Consistency**: Standard deviation of 0.009x (very consistent improvement)
- **Evolution score**: 0.9928 → 0.9929
- **Stability**: This solution remained the best through all remaining 30 iterations

See benchmark_comparison.py for detailed statistics.

## Files in This Example

- **`initial_program.py`**: Original difflib Differ implementation with EVOLVE-BLOCK markers
- **`best_program.py`**: Optimized version discovered by OpenEvolve
- **`evaluator.py`**: Evaluation function that tests correctness and performance
- **`config.yaml`**: OpenEvolve configuration used for evolution
- **`best_program_info.json`**: Metadata about the best program (iteration, parent, metrics)
- **`benchmark_comparison.py`**: Script to compare initial vs optimized _fancy_replace performance
- **`benchmark_get_close_matches.py`**: Script demonstrating the same optimization applies to get_close_matches()

## Running the Benchmarks

To see the performance difference yourself:

```bash
cd examples/difflib_optimization

# Benchmark the main _fancy_replace optimization (5.24% speedup)
python benchmark_comparison.py

# Benchmark the get_close_matches optimization (0.62% speedup)
python benchmark_get_close_matches.py
```

Expected output:
```
================================================================================
OpenEvolve Difflib Optimization - Performance Comparison
================================================================================

Pathological Case (GitHub issue #119105):
Comparing files with many similar lines differing by one character
Running 200 iterations per test size for statistical accuracy
Total comparisons: 1400

  Size │  Initial (sec) │     Best (sec) │   Saved (sec) │    Speedup
─────────────────────────────────────────────────────────────────────────────
   100 │         0.130 │         0.122 │        0.008 │   1.07x ✅ (+6.4%)
   200 │         0.256 │         0.244 │        0.012 │   1.05x    (+4.7%)
   500 │         0.655 │         0.616 │        0.039 │   1.06x ✅ (+5.9%)
  1000 │         1.289 │         1.215 │        0.074 │   1.06x ✅ (+5.7%)
  1500 │         1.931 │         1.824 │        0.107 │   1.06x ✅ (+5.5%)
  2000 │         2.559 │         2.451 │        0.108 │   1.04x    (+4.2%)
  3000 │         3.965 │         3.807 │        0.158 │   1.04x    (+4.0%)
─────────────────────────────────────────────────────────────────────────────
 TOTAL │        10.785 │        10.278 │        0.507 │   1.05x

Statistical Summary:
  Mean speedup:   1.055x (+5.24% faster)
  Min speedup:    1.042x (+4.00% faster)
  Max speedup:    1.068x (+6.41% faster)
  Std deviation:  0.009x
```

## Running Evolution Yourself

To reproduce the optimization from scratch:

```bash
cd examples/difflib_optimization

# Run evolution for 40 iterations (takes ~60-80 minutes)
python ../../openevolve-run.py initial_program.py evaluator.py \
  --config config.yaml \
  --iterations 40
```

## Key Takeaways

1. **Subtle optimizations matter**: Even simple code changes like eliminating redundant function calls can yield meaningful performance improvements.

2. **LLMs find real optimizations**: The optimization discovered by Gemini 2.5 Pro is a classic performance pattern that could be submitted as a patch to CPython.

3. **Stable solutions emerge early**: The best solution was found at iteration 10 and never beaten in 30 more iterations, suggesting it discovered a genuine local optimum.

4. **Standard library optimization**: This demonstrates that even well-tested, mature code in Python's standard library can benefit from automated optimization.

## Configuration Details

- **LLM**: Gemini 2.5 Pro (via OpenRouter)
- **Population size**: 120 programs
- **Islands**: 4 parallel evolution populations
- **Temperature**: 0.7 (higher for creative algorithmic solutions)
- **Evolution strategy**: Diff-based (only evolves code within EVOLVE-BLOCK markers)

## Additional Finding: get_close_matches()

After discovering the optimization in `_fancy_replace`, we found that `get_close_matches()` (lines 703-708) has the **exact same bug**:

```python
# BEFORE:
if (s.real_quick_ratio() >= cutoff
      and s.quick_ratio() >= cutoff
      and s.ratio() >= cutoff):        # ← Call 1
    result.append((s.ratio(), x))      # ← Call 2 (redundant!)

# AFTER:
if s.real_quick_ratio() >= cutoff and s.quick_ratio() >= cutoff:
    r = s.ratio()                      # ← Single call
    if r >= cutoff:
        result.append((r, x))          # ← Reuse result
```

However, the impact is much smaller:
- **Mean speedup**: 0.62% (vs 5.24% for _fancy_replace)
- **Reason**: get_close_matches() does simpler string matching, so ratio() is less dominant

See `benchmark_get_close_matches.py` for details.

## Potential Impact

If the `_fancy_replace` optimization were integrated into CPython's standard library, it would:
- Speed up diff operations across all Python installations globally (5.24% faster)
- Reduce computational costs for tools that rely on difflib (code review, version control, etc.)
- Improve user experience in interactive diff tools
- Set a precedent for using LLM-based optimization on standard library code

The `get_close_matches()` fix is technically correct but has minimal real-world impact (~0.6%).

This example shows OpenEvolve's potential for discovering meaningful optimizations in production code!

## License

This task is based on code from CPython's `difflib` module, which is distributed
under the Python Software Foundation License. The upstream license text is included
at `LICENSE` in this task directory. EvolveBench's own harness and task code are
licensed separately — see the repository root [LICENSE](../../LICENSE).
