#!/usr/bin/env python3
"""
Evaluator for LMCache LFU Cache Policy Optimization

This evaluator tests both correctness and performance of the LFU cache policy.

Correctness: Output must match expected LFU behavior
Performance: Tested on various cache sizes and access patterns

Scoring:
- Stage 1 (quick): Basic correctness + small cache performance (100 entries)
- Stage 2 (comprehensive): All test cases including large caches

Baseline performance (original SortedDict implementation):
- Operations are O(log N) complexity
- Target: Achieve O(1) operations with min-frequency tracking
"""

import sys
import time
import math
import tempfile
from typing import Dict, Any, Optional


def sigmoid_performance_score(actual_time: float, target_time: float) -> float:
    """
    Calculate performance score using sigmoid function.

    Score = 1 / (1 + exp(5 * relative_slowdown))

    Where relative_slowdown = (actual_time - target_time) / target_time
    
    Balanced scoring: reasonable penalty factor for good differentiation
    """
    if target_time <= 0:
        return 1.0 if actual_time <= 0 else 0.0

    relative_slowdown = (actual_time - target_time) / target_time
    return 1.0 / (1.0 + math.exp(5 * relative_slowdown))


def test_correctness_basic(evolved_module) -> Dict[str, Any]:
    """
    Basic correctness test: LFU eviction behavior.

    Tests:
    1. Basic LFU eviction (least frequently used first)
    2. Frequency tie-breaking with FIFO
    3. Pinned entries (can_evict=False)
    4. Force eviction
    """
    errors = []

    # Import classes from evolved module
    CacheEngineKey = evolved_module.CacheEngineKey
    CacheEntry = evolved_module.CacheEntry
    create_policy = evolved_module.create_policy

    # Test 1: Basic LFU eviction
    policy = create_policy()
    cache_dict = {}

    key1 = CacheEngineKey(1)
    key2 = CacheEngineKey(2)
    key3 = CacheEngineKey(3)

    cache_dict[key1] = CacheEntry("val1")
    cache_dict[key2] = CacheEntry("val2")
    cache_dict[key3] = CacheEntry("val3")

    policy.update_on_put(key1)
    policy.update_on_put(key2)
    policy.update_on_put(key3)

    # Access pattern: key3=3 times, key2=2 times, key1=1 time
    policy.update_on_hit(key3, cache_dict)
    policy.update_on_hit(key3, cache_dict)
    policy.update_on_hit(key2, cache_dict)
    policy.update_on_hit(key2, cache_dict)
    policy.update_on_hit(key1, cache_dict)

    # Evict 2 candidates: should be key1 (freq=2), then key3 (freq=3, inserted first)
    candidates = policy.get_evict_candidates(cache_dict, num_candidates=2)

    if candidates != [key1, key3]:
        errors.append(f"Test 1: Expected [key1, key3], got {candidates}")

    # Test 2: FIFO tie-breaking for same frequency
    policy = create_policy()
    cache_dict = {}

    key1 = CacheEngineKey(1)
    key2 = CacheEngineKey(2)
    key3 = CacheEngineKey(3)

    cache_dict[key1] = CacheEntry("val1")
    cache_dict[key2] = CacheEntry("val2")
    cache_dict[key3] = CacheEntry("val3")

    policy.update_on_put(key1)
    policy.update_on_put(key2)
    policy.update_on_put(key3)

    # All have frequency=1, should evict in FIFO order
    candidates = policy.get_evict_candidates(cache_dict, num_candidates=2)

    if candidates != [key1, key2]:
        errors.append(f"Test 2: Expected [key1, key2] (FIFO), got {candidates}")

    # Test 3: Pinned entries (can_evict=False)
    policy = create_policy()
    cache_dict = {}

    key1 = CacheEngineKey(1)
    key2 = CacheEngineKey(2)
    key3 = CacheEngineKey(3)

    cache_dict[key1] = CacheEntry("val1", can_evict=False)  # Pinned
    cache_dict[key2] = CacheEntry("val2")
    cache_dict[key3] = CacheEntry("val3")

    policy.update_on_put(key1)
    policy.update_on_put(key2)
    policy.update_on_put(key3)

    policy.update_on_hit(key3, cache_dict)
    policy.update_on_hit(key3, cache_dict)
    policy.update_on_hit(key2, cache_dict)
    policy.update_on_hit(key2, cache_dict)
    policy.update_on_hit(key1, cache_dict)

    # key1 has lowest effective freq but is pinned, should skip it
    candidates = policy.get_evict_candidates(cache_dict, num_candidates=2)

    if candidates != [key3, key2]:
        errors.append(f"Test 3: Expected [key3, key2] (skip pinned), got {candidates}")

    # Test 4: Force eviction
    policy = create_policy()
    cache_dict = {}

    key1 = CacheEngineKey(1)
    key2 = CacheEngineKey(2)

    cache_dict[key1] = CacheEntry("val1")
    cache_dict[key2] = CacheEntry("val2")

    policy.update_on_put(key1)
    policy.update_on_put(key2)
    policy.update_on_hit(key1, cache_dict)

    policy.update_on_force_evict(key1)

    # After force evicting key1, only key2 should remain
    candidates = policy.get_evict_candidates(cache_dict, num_candidates=1)

    if key1 in candidates:
        errors.append(f"Test 4: key1 should be force-evicted, got {candidates}")

    return {
        "passed": len(errors) == 0,
        "errors": errors,
        "num_tests": 4,
        "num_passed": 4 - len(errors)
    }


def test_performance_cache_operations(evolved_module, cache_size: int, num_operations: int) -> Dict[str, Any]:
    """
    Test performance on cache operations.

    Simulates a realistic workload with:
    - Puts (cache population)
    - Hits (cache accesses following Zipf distribution)
    - Evictions (when cache is full)
    """
    CacheEngineKey = evolved_module.CacheEngineKey
    CacheEntry = evolved_module.CacheEntry
    create_policy = evolved_module.create_policy

    policy = create_policy()
    cache_dict = {}

    import random
    random.seed(42)

    start_time = time.time()

    try:
        # Phase 1: Populate cache
        for i in range(cache_size):
            key = CacheEngineKey(i)
            cache_dict[key] = CacheEntry(f"value_{i}")
            policy.update_on_put(key)

        # Phase 2: Access pattern (Zipf-like: some keys accessed much more frequently)
        # This stresses the frequency tracking
        for _ in range(num_operations):
            # 80% of accesses go to 20% of keys (hot keys)
            if random.random() < 0.8:
                key_id = random.randint(0, cache_size // 5)
            else:
                key_id = random.randint(0, cache_size - 1)

            key = CacheEngineKey(key_id)
            if key in cache_dict:
                policy.update_on_hit(key, cache_dict)

        # Phase 3: Evictions
        num_evictions = cache_size // 10  # Evict 10% of cache
        for _ in range(num_evictions):
            candidates = policy.get_evict_candidates(cache_dict, num_candidates=1)
            if candidates:
                evicted_key = candidates[0]
                del cache_dict[evicted_key]

        elapsed = time.time() - start_time

        return {
            "success": True,
            "elapsed_time": elapsed,
            "cache_size": cache_size,
            "num_operations": num_operations,
            "error": None
        }

    except Exception as e:
        elapsed = time.time() - start_time
        return {
            "success": False,
            "elapsed_time": elapsed,
            "cache_size": cache_size,
            "num_operations": num_operations,
            "error": str(e)
        }


def evaluate(program_code: str) -> Dict[str, Any]:
    """
    Main evaluation function called by OpenEvolve.

    This is the entry point for OpenEvolve's evaluation system.
    """
    return eval_initial_program(program_code)


def eval_initial_program(program_code: str) -> Dict[str, Any]:
    """
    Stage 1: Quick validation (used during evolution)

    Tests:
    1. Basic correctness
    2. Small cache performance (cache_size=100, 1000 ops)

    Returns combined correctness and performance score.
    """
    print("\n🔍 LMCACHE LFU EVALUATOR - Stage 1 (Quick validation)")

    # If program_code is a file path, read the file
    import os
    if os.path.exists(program_code):
        with open(program_code, 'r') as f:
            code = f.read()
    else:
        code = program_code

    # Load the evolved program
    try:
        # Enable postponed evaluation of annotations (PEP 563)
        # This prevents NameError when type hints reference classes defined later
        future_annotations_code = "from __future__ import annotations\n"
        
        # Create namespace with necessary imports
        namespace = {
            '__builtins__': __builtins__,
            'Any': Any,
            'Optional': Optional,
        }
        
        # Add common imports that might be needed
        try:
            from sortedcontainers import SortedDict
            namespace['SortedDict'] = SortedDict
        except ImportError:
            pass
        
        try:
            from collections import defaultdict, OrderedDict
            namespace['defaultdict'] = defaultdict
            namespace['OrderedDict'] = OrderedDict
        except ImportError:
            pass
        
        # Add typing imports
        try:
            from typing import Dict, List, Set, Tuple
            namespace.update({
                'Dict': Dict,
                'List': List,
                'Set': Set,
                'Tuple': Tuple,
            })
        except ImportError:
            pass
        
        # Prepend future annotations if not already present
        if 'from __future__ import annotations' not in code:
            code_to_exec = future_annotations_code + code
        else:
            code_to_exec = code
        
        exec(code_to_exec, namespace)
        import types
        evolved_module = types.SimpleNamespace(**namespace)
        
        # Check if this is a partial code block (missing class definitions)
        # If so, load the base classes from lmcache.py
        if not hasattr(evolved_module, 'CacheEngineKey') or not hasattr(evolved_module, 'CacheEntry'):
            # Load base classes from lmcache.py
            import os
            lmcache_path = os.path.join(os.path.dirname(__file__), 'lmcache.py')
            if os.path.exists(lmcache_path):
                with open(lmcache_path, 'r') as f:
                    base_code = f.read()
                
                # Execute base code to get class definitions
                base_namespace = {
                    '__builtins__': __builtins__,
                    'Any': Any,
                    'Optional': Optional,
                }
                
                # Add imports
                try:
                    from sortedcontainers import SortedDict
                    base_namespace['SortedDict'] = SortedDict
                except ImportError:
                    pass
                
                try:
                    from collections import defaultdict, OrderedDict
                    base_namespace['defaultdict'] = defaultdict
                    base_namespace['OrderedDict'] = OrderedDict
                except ImportError:
                    pass
                
                try:
                    from typing import Dict, List, Set, Tuple
                    base_namespace.update({
                        'Dict': Dict,
                        'List': List,
                        'Set': Set,
                        'Tuple': Tuple,
                    })
                except ImportError:
                    pass
                
                if 'from __future__ import annotations' not in base_code:
                    base_code = future_annotations_code + base_code
                
                exec(base_code, base_namespace)
                
                # Copy base classes to evolved namespace
                if 'CacheEngineKey' in base_namespace:
                    namespace['CacheEngineKey'] = base_namespace['CacheEngineKey']
                if 'CacheEntry' in base_namespace:
                    namespace['CacheEntry'] = base_namespace['CacheEntry']
                
                # Recreate evolved_module with updated namespace
                evolved_module = types.SimpleNamespace(**namespace)
        
        # Verify required classes/functions exist
        if not hasattr(evolved_module, 'CacheEngineKey'):
            raise AttributeError("CacheEngineKey class not found in evolved code or base file")
        if not hasattr(evolved_module, 'CacheEntry'):
            raise AttributeError("CacheEntry class not found in evolved code or base file")
        if not hasattr(evolved_module, 'create_policy'):
            # If create_policy is missing, create a factory function
            if hasattr(evolved_module, 'LFUCachePolicy'):
                namespace['create_policy'] = lambda: namespace['LFUCachePolicy']()
                evolved_module = types.SimpleNamespace(**namespace)
            else:
                raise AttributeError("create_policy function and LFUCachePolicy class not found in evolved code")
            
    except SyntaxError as e:
        print(f"  ❌ Syntax error in program: {e}")
        if hasattr(e, 'lineno') and hasattr(e, 'text'):
            print(f"     Line {e.lineno}: {e.text}")
        return {
            "correctness": 0.0,
            "performance": 0.0,
            "combined_score": 0.0,
            "error": f"Syntax error: {e}"
        }
    except Exception as e:
        print(f"  ❌ Failed to load program: {e}")
        import traceback
        traceback.print_exc()
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

    # Test performance on small cache
    perf_result = test_performance_cache_operations(
        evolved_module,
        cache_size=100,
        num_operations=5000  # 5x more operations for more challenging test
    )

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
    # Target: 0.005s for 100-entry cache with 5000 ops (more reasonable but still challenging)
    target_time = 0.005
    elapsed = perf_result["elapsed_time"]
    performance_score = sigmoid_performance_score(elapsed, target_time)

    print(f"  ⏱️  Performance: {elapsed:.4f}s for {perf_result['cache_size']} entries, {perf_result['num_operations']} ops (target: {target_time:.3f}s)")
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
    2. Multiple cache sizes: 100, 1000, 10000 entries
    3. Various operation counts

    Baseline scores to beat:
    - Size 100: Target 0.005s (challenging but achievable)
    - Size 1000: Target 0.050s (challenging but achievable)
    - Size 10000: Target 0.500s (challenging but achievable)
    - Size 50000: Target 2.500s (challenging but achievable)
    """
    print("\n🔍 LMCACHE LFU EVALUATOR - Stage 2 (Comprehensive)")

    # If program_code is a file path, read the file
    import os
    if os.path.exists(program_code):
        with open(program_code, 'r') as f:
            code = f.read()
    else:
        code = program_code

    # Load the evolved program
    try:
        # Enable postponed evaluation of annotations (PEP 563)
        # This prevents NameError when type hints reference classes defined later
        future_annotations_code = "from __future__ import annotations\n"
        
        # Create namespace with necessary imports
        namespace = {
            '__builtins__': __builtins__,
            'Any': Any,
            'Optional': Optional,
        }
        
        # Add common imports that might be needed
        try:
            from sortedcontainers import SortedDict
            namespace['SortedDict'] = SortedDict
        except ImportError:
            pass
        
        try:
            from collections import defaultdict, OrderedDict
            namespace['defaultdict'] = defaultdict
            namespace['OrderedDict'] = OrderedDict
        except ImportError:
            pass
        
        # Add typing imports
        try:
            from typing import Dict, List, Set, Tuple
            namespace.update({
                'Dict': Dict,
                'List': List,
                'Set': Set,
                'Tuple': Tuple,
            })
        except ImportError:
            pass
        
        # Prepend future annotations if not already present
        if 'from __future__ import annotations' not in code:
            code_to_exec = future_annotations_code + code
        else:
            code_to_exec = code
        
        exec(code_to_exec, namespace)
        import types
        evolved_module = types.SimpleNamespace(**namespace)
        
        # Check if this is a partial code block (missing class definitions)
        # If so, load the base classes from lmcache.py
        if not hasattr(evolved_module, 'CacheEngineKey') or not hasattr(evolved_module, 'CacheEntry'):
            # Load base classes from lmcache.py
            import os
            lmcache_path = os.path.join(os.path.dirname(__file__), 'lmcache.py')
            if os.path.exists(lmcache_path):
                with open(lmcache_path, 'r') as f:
                    base_code = f.read()
                
                # Execute base code to get class definitions
                base_namespace = {
                    '__builtins__': __builtins__,
                    'Any': Any,
                    'Optional': Optional,
                }
                
                # Add imports
                try:
                    from sortedcontainers import SortedDict
                    base_namespace['SortedDict'] = SortedDict
                except ImportError:
                    pass
                
                try:
                    from collections import defaultdict, OrderedDict
                    base_namespace['defaultdict'] = defaultdict
                    base_namespace['OrderedDict'] = OrderedDict
                except ImportError:
                    pass
                
                try:
                    from typing import Dict, List, Set, Tuple
                    base_namespace.update({
                        'Dict': Dict,
                        'List': List,
                        'Set': Set,
                        'Tuple': Tuple,
                    })
                except ImportError:
                    pass
                
                if 'from __future__ import annotations' not in base_code:
                    base_code = future_annotations_code + base_code
                
                exec(base_code, base_namespace)
                
                # Copy base classes to evolved namespace
                if 'CacheEngineKey' in base_namespace:
                    namespace['CacheEngineKey'] = base_namespace['CacheEngineKey']
                if 'CacheEntry' in base_namespace:
                    namespace['CacheEntry'] = base_namespace['CacheEntry']
                
                # Recreate evolved_module with updated namespace
                evolved_module = types.SimpleNamespace(**namespace)
        
        # Verify required classes/functions exist
        if not hasattr(evolved_module, 'CacheEngineKey'):
            raise AttributeError("CacheEngineKey class not found in evolved code or base file")
        if not hasattr(evolved_module, 'CacheEntry'):
            raise AttributeError("CacheEntry class not found in evolved code or base file")
        if not hasattr(evolved_module, 'create_policy'):
            # If create_policy is missing, create a factory function
            if hasattr(evolved_module, 'LFUCachePolicy'):
                namespace['create_policy'] = lambda: namespace['LFUCachePolicy']()
                evolved_module = types.SimpleNamespace(**namespace)
            else:
                raise AttributeError("create_policy function and LFUCachePolicy class not found in evolved code")
            
    except Exception as e:
        print(f"  ❌ Failed to load program: {e}")
        import traceback
        traceback.print_exc()
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

    # Test performance on multiple cache sizes
    test_configs = [
        (100, 1000, 0.005),      # Small cache (more reasonable but challenging)
        (1000, 10000, 0.050),    # Medium cache (more reasonable but challenging)
        (10000, 100000, 0.500),  # Large cache (more reasonable but challenging)
        (50000, 500000, 2.500),  # Very large cache (challenging but achievable)
    ]

    scores = []
    total_time = 0.0

    for cache_size, num_ops, target_time in test_configs:
        perf_result = test_performance_cache_operations(
            evolved_module,
            cache_size=cache_size,
            num_operations=num_ops
        )

        if not perf_result["success"]:
            print(f"  ❌ Size {cache_size:5d}: FAILED - {perf_result['error']}")
            scores.append(0.0)
        else:
            elapsed = perf_result["elapsed_time"]
            total_time += elapsed
            score = sigmoid_performance_score(elapsed, target_time)
            scores.append(score)
            print(f"  ✅ Size {cache_size:5d}: {elapsed:7.4f}s (target: {target_time:6.3f}s, score: {score:.3f})")

    # Calculate overall performance score
    performance_score = sum(scores) / len(scores) if scores else 0.0
    combined_score = correctness_score * performance_score

    print(f"\n  📊 Final Scores:")
    print(f"     Correctness: {correctness_score:.3f}")
    print(f"     Performance: {performance_score:.3f}")
    print(f"     Combined: {combined_score:.3f}")
    print(f"     Total time: {total_time:.4f}s")

    return {
        "correctness": correctness_score,
        "performance": performance_score,
        "combined_score": combined_score,
        "total_time": total_time,
        "test_scores": scores
    }


if __name__ == "__main__":
    if len(sys.argv) > 1:
        result = evaluate(sys.argv[1])
        print(f"\n✅ Evaluation complete: {result}")
    else:
        print("Usage: python evaluator.py <program_file>")
