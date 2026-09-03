#!/usr/bin/env python3
"""
Evaluator for pymoo non-dominated sorting optimization task.

Metrics: correctness, performance, combined_score.
"""

import types
import time
import numpy as np
from typing import Dict
from pymoo.util.dominator import Dominator


def original_fast_non_dominated_sort(F, dominator=Dominator(), **kwargs):
    if "dominator" in kwargs:
        M = Dominator.calc_domination_matrix(F)
    else:
        M = dominator.calc_domination_matrix(F)

    n = M.shape[0]
    fronts = []
    if n == 0:
        return fronts

    n_ranked = 0
    ranked = np.zeros(n, dtype=int)
    is_dominating = [[] for _ in range(n)]
    n_dominated = np.zeros(n)
    current_front = []

    for i in range(n):
        for j in range(i + 1, n):
            rel = M[i, j]
            if rel == 1:
                is_dominating[i].append(j)
                n_dominated[j] += 1
            elif rel == -1:
                is_dominating[j].append(i)
                n_dominated[i] += 1
        if n_dominated[i] == 0:
            current_front.append(i)
            ranked[i] = 1.0
            n_ranked += 1

    fronts.append(current_front)
    while n_ranked < n:
        next_front = []
        for i in current_front:
            for j in is_dominating[i]:
                n_dominated[j] -= 1
                if n_dominated[j] == 0:
                    next_front.append(j)
                    ranked[j] = 1.0
                    n_ranked += 1
        fronts.append(next_front)
        current_front = next_front
    return fronts


def evaluate(code: str) -> Dict[str, float]:
    """Evaluate evolved fast_non_dominated_sort implementation.

    The evolved code must define `fast_non_dominated_sort(F, dominator=Dominator(), **kwargs)`.
    Correctness compares Pareto fronts with the reference implementation on several sizes.
    Performance measures relative speedup and maps it to [0,1].
    """
    try:
        # Handle both file paths and code content
        import os
        if os.path.isfile(code.strip()):
            with open(code.strip(), 'r') as f:
                code_content = f.read()
        else:
            # Try relative path from the task directory
            relative_path = os.path.join(os.path.dirname(__file__), code.strip())
            if os.path.isfile(relative_path):
                with open(relative_path, 'r') as f:
                    code_content = f.read()
            else:
                code_content = code

        # Check if this contains the target optimization block
        if "performance-non-dominated-sorting" in code_content:
            # Import the module directly and use the function
            import sys
            import importlib.util
            
            # Create a temporary module from the file
            file_path = code.strip() if os.path.isfile(code.strip()) else os.path.join(os.path.dirname(__file__), code.strip())
            spec = importlib.util.spec_from_file_location("temp_module", file_path)
            if spec and spec.loader:
                temp_module = importlib.util.module_from_spec(spec)
                sys.modules["temp_module"] = temp_module
                spec.loader.exec_module(temp_module)
                
                if hasattr(temp_module, "fast_non_dominated_sort"):
                    evo = temp_module.fast_non_dominated_sort
                else:
                    return {"correctness": 0.0, "performance": 0.0, "combined_score": 0.0}
            else:
                return {"correctness": 0.0, "performance": 0.0, "combined_score": 0.0}
        else:
            return {"correctness": 0.0, "performance": 0.0, "combined_score": 0.0}

        # Small bi-objective test sizes (where speedups are most impactful)
        rng = np.random.default_rng(42)
        sizes = [50, 100, 200]

        def gen(n):
            return rng.random((n, 2)) * 100.0

        def norm(fronts):
            return [sorted(front) for front in fronts]

        # Correctness
        ok = 0
        total = 0
        for n in sizes:
            F = gen(n)
            ref = original_fast_non_dominated_sort(F)
            test = evo(F)
            if norm(ref) == norm(test):
                ok += 1
            total += 1
        correctness = ok / total if total else 0.0

        # Performance (relative speedup)
        def bench(f, F, runs=2):
            ts = []
            for _ in range(runs):
                s = time.perf_counter()
                _ = f(F)
                ts.append(time.perf_counter() - s)
            return sum(ts) / len(ts)

        ratios = []
        for n in sizes:
            F = gen(n)
            tr = bench(original_fast_non_dominated_sort, F)
            te = bench(evo, F)
            if te > 0:
                ratios.append(tr / te)

        if ratios:
            avg = sum(ratios) / len(ratios)
            # Map speedup to [0,1] around baseline 1.0x ~ 0.5
            performance = max(0.0, min(1.0, 0.5 + 0.25 * (avg - 1.0)))
        else:
            performance = 0.0

        combined = 0.6 * correctness + 0.4 * performance
        
        return {
            "correctness": float(correctness),
            "performance": float(performance),
            "combined_score": float(combined),
        }
    except Exception as e:
        return {"correctness": 0.0, "performance": 0.0, "combined_score": 0.0}