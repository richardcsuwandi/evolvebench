"""
Pandas Rolling Rank - Baseline Using Pandas' Default Implementation
====================================================================

This serves as the baseline for a potential pandas PR. We use pandas' own
rolling_rank implementation and will evolve an optimized version for small
window sizes that's faster while maintaining 100% compatibility.

Current pandas implementation uses a C-based skiplist for ALL window sizes.
Our goal: Discover that Numba JIT is faster for typical windows (w < 300).

Potential PR: Add fast path for small windows using Numba JIT.
"""

import time
import numpy as np
import pandas as pd


class RollingRank:
    """
    Wrapper around pandas' default rolling rank implementation.

    This represents the current state-of-the-art in pandas.
    We'll use OpenEvolve to discover optimizations for small windows.
    """

    def __init__(self, window_size, method='average', ascending=True, pct=False):
        """
        Initialize rolling rank calculator using pandas' implementation.

        Parameters
        ----------
        window_size : int
            Size of the rolling window
        method : str, default 'average'
            How to rank duplicates:
            - 'average': average rank of tied values
            - 'min': minimum rank of tied values
            - 'max': maximum rank of tied values
        ascending : bool, default True
            Whether to rank in ascending order
        pct : bool, default False
            If True, return percentile rank (rank / count)
        """
        self.window_size = window_size
        self.method = method
        self.ascending = ascending
        self.pct = pct

    # EVOLVE-BLOCK-START id="rolling-rank-small-windows"
    def compute(self, values):
        """
        Compute rolling rank using pandas' C-based skiplist implementation.

        BASELINE PERFORMANCE:
        - Uses O(n log w) skiplist algorithm
        - Implemented in Cython for speed
        - Works well for all window sizes

        OPPORTUNITY FOR IMPROVEMENT:
        - For small windows (w < 300), JIT compilation can be faster
        - Simpler algorithm with lower constant factors
        - Potential 2-3x speedup for typical use cases
        """
        values = np.asarray(values, dtype=np.float64)

        # Use pandas' implementation directly
        series = pd.Series(values)
        result = series.rolling(
            window=self.window_size,
            min_periods=self.window_size
        ).rank(
            method=self.method,
            ascending=self.ascending,
            pct=self.pct
        ).values

        return result
    # EVOLVE-BLOCK-END


def benchmark_rolling_rank(n_values, window_size):
    """
    Benchmark pandas' default rolling rank implementation.
    """
    # Generate random data
    np.random.seed(42)
    values = np.random.randn(n_values)

    # Create rolling rank calculator
    roller = RollingRank(window_size=window_size, method='average')

    # Time the computation
    start = time.time()
    result = roller.compute(values)
    elapsed = time.time() - start

    return {
        'n_values': n_values,
        'window_size': window_size,
        'elapsed_time': elapsed,
        'result_size': len(result),
        'result_sample': result[-10:].tolist() if len(result) >= 10 else result.tolist()
    }


if __name__ == "__main__":
    print("Pandas Rolling Rank - Baseline (Current pandas Implementation)")
    print("=" * 70)
    print()
    print("This uses pandas' C-based skiplist, which is the current default.")
    print("Goal: Evolve a faster version for small windows (w < 300)")
    print()

    # Test different problem sizes
    test_cases = [
        (1000, 50),
        (5000, 50),
        (10000, 50),
        (10000, 100),
        (10000, 200),
    ]

    print(f"{'n':>6} {'window':>6} {'Time (s)':>10}")
    print("-" * 30)

    for n_values, window_size in test_cases:
        result = benchmark_rolling_rank(n_values, window_size)
        print(f"{result['n_values']:6d} {result['window_size']:6d} {result['elapsed_time']:9.4f}")

    print()
    print("Baseline: pandas C skiplist O(n log w)")
    print("Target: Discover JIT optimization for 2-3x speedup on small windows!")
