"""
SymPy Min/Max _find_localzeros Optimization
============================================

This program contains the O(n²) _find_localzeros algorithm from SymPy that finds
the minimal/maximal elements (localzeros) in a set of potentially comparable values.

Problem: The current implementation compares every value against all previously added
localzeros, resulting in O(n²) complexity.

Opportunity: Use transitivity to reduce comparisons. If (x < y) and (y < z), then
we know (x < z) without comparing them directly.

GitHub Issue: https://github.com/sympy/sympy/issues/16249
PR #27758: Achieved 3.3x speedup but didn't change algorithmic complexity
"""

import time
from sympy import symbols, S, sympify, Expr
from sympy.core.exprtools import factor_terms

# EVOLVE-BLOCK-START id="performance-min-max-localzeros"
class MinMaxBase:
    """
    Simplified MinMaxBase class focusing on the _find_localzeros algorithm.

    The goal is to optimize the O(n²) algorithm that sequentially allocates values
    to localzeros by finding which values are more extreme than others.
    """

    @classmethod
    def _find_localzeros(cls, values, **options):
        """
        Sequentially allocate values to localzeros.

        When a value is identified as being more extreme than another member it
        replaces that member; if this is never true, then the value is simply
        appended to the localzeros.

        CURRENT COMPLEXITY: O(n²) - compares each value against all existing localzeros
        TARGET COMPLEXITY: O(n log n) or O(n) using transitivity
        """
        localzeros = set()
        # This is the O(n²) bottleneck that needs optimization
        # The algorithm compares every new value v against all existing localzeros
        # Optimization opportunity: Use transitivity to avoid redundant comparisons
        # If (x, y) and (y, z) have been compared, then (x, z) doesn't need testing

        for v in values:
            is_newzero = True
            localzeros_ = list(localzeros)  # O(n) conversion on each iteration
            for z in localzeros_:            # Nested loop = O(n²) total
                if id(v) == id(z):
                    is_newzero = False
                else:
                    con = cls._is_connected(v, z)
                    if con:
                        is_newzero = False
                        if con is True or con == cls:
                            localzeros.remove(z)
                            localzeros.update([v])
            if is_newzero:
                localzeros.update([v])

        return localzeros
# EVOLVE-BLOCK-END

    @classmethod
    def _is_connected(cls, x, y):
        """
        Check if x and y are connected somehow.

        Returns:
        - True if x == y
        - Max if x > y
        - Min if x < y
        - False if not comparable
        """
        for i in range(2):
            if x == y:
                return True
            t, f = Max, Min
            for op in "><":
                for j in range(2):
                    try:
                        if op == ">":
                            v = x >= y
                        else:
                            v = x <= y
                    except TypeError:
                        return False  # non-real arg
                    if not hasattr(v, 'is_Relational') or not v.is_Relational:
                        return t if v else f
                    t, f = f, t
                    x, y = y, x
                x, y = y, x  # run next pass with reversed order relative to start
            # simplification can be expensive, so be conservative
            # in what is attempted
            x = factor_terms(x - y)
            y = S.Zero

        return False


class Max(MinMaxBase):
    """Max class - represents maximum function"""
    pass


class Min(MinMaxBase):
    """Min class - represents minimum function"""
    pass


# Set the class references for _is_connected to work
MinMaxBase.Max = Max
MinMaxBase.Min = Min


def benchmark_min_construction(num_symbols):
    """
    Benchmark Min(*symbols('x:N')) construction time.

    This is the test case from Issue #16249 that takes 2-3 seconds for 50 symbols.
    """
    syms = symbols(f'x:{num_symbols}')

    start_time = time.time()
    result = Min._find_localzeros(syms)
    elapsed_time = time.time() - start_time

    return {
        'num_symbols': num_symbols,
        'elapsed_time': elapsed_time,
        'result_size': len(result)
    }


if __name__ == "__main__":
    print("SymPy Min/Max _find_localzeros Baseline Performance")
    print("=" * 60)
    print()

    test_sizes = [10, 20, 30, 40, 50]

    for size in test_sizes:
        result = benchmark_min_construction(size)
        print(f"Symbols: {result['num_symbols']:3d} | "
              f"Time: {result['elapsed_time']:.4f}s | "
              f"Localzeros: {result['result_size']}")

    print()
    print("Note: For 50 symbols, baseline is ~3.0s (post-PR #27758)")
    print("Target: <1.0s for 50 symbols using transitivity optimization")
