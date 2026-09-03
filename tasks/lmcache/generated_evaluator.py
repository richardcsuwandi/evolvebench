import math
import time
import importlib.util
import sys
import random
from typing import Any


def sigmoid_performance_score(actual_value: float, target_value: float, steepness: float = 2.0) -> float:
    """
    Creates a sigmoid scoring curve for smooth gradient optimization.
    
    At target_value: score = 0.5
    Better than target: score approaches 1.0
    Worse than target: score approaches 0.0
    
    Steepness controls how quickly score changes around target (default 2.0).
    Higher steepness = sharper transitions, lower = smoother transitions.
    """
    if actual_value <= 0:
        return 0.99  # Near-perfect for instantaneous/zero time
    
    # For "lower is better" metrics (like execution time)
    relative_performance = (actual_value - target_value) / target_value
    return 1.0 / (1.0 + math.exp(steepness * relative_performance))


# Baseline measurements from profiling
baseline_measurements = {
    "test_config_small": {
        "target_time": 0.000754,
        "cache_size": 100,
        "num_operations": 1000,
        "num_evictions": 50,
        "hit_ratio": 0.700000,
        "ops_per_second": 1392973.100000,
        "description": "Small test with 100 cache entries, 1000 operations",
    },
    "test_config_medium": {
        "target_time": 0.007917,
        "cache_size": 1000,
        "num_operations": 10000,
        "num_evictions": 500,
        "hit_ratio": 0.700000,
        "ops_per_second": 1326310.240000,
        "description": "Medium test with 1000 cache entries, 10000 operations",
    },
    "test_config_large": {
        "target_time": 0.044421,
        "cache_size": 5000,
        "num_operations": 50000,
        "num_evictions": 2500,
        "hit_ratio": 0.700000,
        "ops_per_second": 1181862.570000,
        "description": "Large test with 5000 cache entries, 50000 operations",
    },
}


def load_module(program_path: str):
    """Load the program module from the given path."""
    spec = importlib.util.spec_from_file_location("lfu_module", program_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["lfu_module"] = module
    spec.loader.exec_module(module)
    return module


def run_cache_simulation(module, cache_size: int, num_operations: int, hit_ratio: float = 0.7):
    """
    Run a cache simulation with the given parameters.
    Returns: (elapsed_time, num_evictions, correctness_passed)
    """
    try:
        # Get classes from module
        CacheEngineKey = module.CacheEngineKey
        CacheEntry = module.CacheEntry
        
        # Create policy instance
        policy = module.create_policy()
        cache_dict = {}
        
        # Track state
        num_evictions = 0
        key_frequencies = {}  # Ground truth frequency tracking
        
        # Phase 1: Fill cache to capacity
        for i in range(cache_size):
            key = CacheEngineKey(i)
            cache_dict[key] = CacheEntry(f"value_{i}")
            policy.update_on_put(key)
            key_frequencies[key] = 1
        
        # Phase 2: Run mixed operations
        start_time = time.perf_counter()
        
        for op_idx in range(num_operations):
            # Decide operation: hit or miss (requiring eviction)
            if random.random() < hit_ratio:
                # Cache hit - access existing key
                key_id = random.randint(0, cache_size - 1)
                key = CacheEngineKey(key_id)
                if key in cache_dict:
                    policy.update_on_hit(key, cache_dict)
                    key_frequencies[key] = key_frequencies.get(key, 1) + 1
            else:
                # Cache miss - need to evict and add new key
                new_key_id = cache_size + num_evictions
                new_key = CacheEngineKey(new_key_id)
                
                # Get eviction candidate
                candidates = policy.get_evict_candidates(cache_dict, num_candidates=1)
                
                if candidates:
                    evicted_key = candidates[0]
                    # Remove from cache
                    del cache_dict[evicted_key]
                    if evicted_key in key_frequencies:
                        del key_frequencies[evicted_key]
                    num_evictions += 1
                
                # Add new entry
                cache_dict[new_key] = CacheEntry(f"value_{new_key_id}")
                policy.update_on_put(new_key)
                key_frequencies[new_key] = 1
        
        elapsed_time = time.perf_counter() - start_time
        
        # Correctness check: verify LFU property
        # Get eviction candidates and verify they have minimum frequency
        if len(cache_dict) > 0:
            candidates = policy.get_evict_candidates(cache_dict, num_candidates=min(5, len(cache_dict)))
            
            if candidates:
                # Check that candidates have among the lowest frequencies
                candidate_freqs = [key_frequencies.get(k, 1) for k in candidates]
                all_freqs = sorted(key_frequencies.values())
                min_freq = min(all_freqs) if all_freqs else 1
                
                # All candidates should have frequency <= median frequency
                median_freq = all_freqs[len(all_freqs) // 2] if all_freqs else 1
                correctness_passed = all(f <= median_freq for f in candidate_freqs)
            else:
                correctness_passed = True
        else:
            correctness_passed = True
        
        return elapsed_time, num_evictions, correctness_passed
    
    except Exception as e:
        print(f"Error in cache simulation: {e}")
        return float('inf'), 0, False


def test_basic_operations(module):
    """Test basic LFU operations for correctness."""
    try:
        CacheEngineKey = module.CacheEngineKey
        CacheEntry = module.CacheEntry
        policy = module.create_policy()
        cache_dict = {}
        
        # Add entries
        keys = [CacheEngineKey(i) for i in range(5)]
        for key in keys:
            cache_dict[key] = CacheEntry(f"value_{key.key_id}")
            policy.update_on_put(key)
        
        # Access key 0 twice, key 1 once
        policy.update_on_hit(keys[0], cache_dict)
        policy.update_on_hit(keys[0], cache_dict)
        policy.update_on_hit(keys[1], cache_dict)
        
        # Get eviction candidates - should be keys 2, 3, or 4 (freq=1)
        candidates = policy.get_evict_candidates(cache_dict, num_candidates=2)
        
        # Verify candidates are not key 0 or key 1
        if len(candidates) != 2:
            return False
        
        for candidate in candidates:
            if candidate.key_id in [0, 1]:
                return False
        
        return True
    except Exception as e:
        print(f"Error in basic operations test: {e}")
        return False


def test_eviction_order(module):
    """Test that eviction follows LFU order."""
    try:
        CacheEngineKey = module.CacheEngineKey
        CacheEntry = module.CacheEntry
        policy = module.create_policy()
        cache_dict = {}
        
        # Add 3 entries with different frequencies
        key1 = CacheEngineKey(1)
        key2 = CacheEngineKey(2)
        key3 = CacheEngineKey(3)
        
        cache_dict[key1] = CacheEntry("v1")
        cache_dict[key2] = CacheEntry("v2")
        cache_dict[key3] = CacheEntry("v3")
        
        policy.update_on_put(key1)
        policy.update_on_put(key2)
        policy.update_on_put(key3)
        
        # Access key1 3 times, key2 2 times, key3 1 time
        for _ in range(3):
            policy.update_on_hit(key1, cache_dict)
        for _ in range(2):
            policy.update_on_hit(key2, cache_dict)
        
        # First eviction should be key3 (lowest frequency)
        candidates = policy.get_evict_candidates(cache_dict, num_candidates=1)
        if len(candidates) != 1 or candidates[0].key_id != 3:
            return False
        
        return True
    except Exception as e:
        print(f"Error in eviction order test: {e}")
        return False


def test_pinned_entries(module):
    """Test that pinned entries are not evicted."""
    try:
        CacheEngineKey = module.CacheEngineKey
        CacheEntry = module.CacheEntry
        policy = module.create_policy()
        cache_dict = {}
        
        # Add entries, some pinned
        key1 = CacheEngineKey(1)
        key2 = CacheEngineKey(2)
        key3 = CacheEngineKey(3)
        
        cache_dict[key1] = CacheEntry("v1", can_evict=False)  # Pinned
        cache_dict[key2] = CacheEntry("v2", can_evict=True)
        cache_dict[key3] = CacheEntry("v3", can_evict=True)
        
        policy.update_on_put(key1)
        policy.update_on_put(key2)
        policy.update_on_put(key3)
        
        # Get eviction candidates - should not include key1
        candidates = policy.get_evict_candidates(cache_dict, num_candidates=3)
        
        for candidate in candidates:
            if candidate.key_id == 1:
                return False
        
        return True
    except Exception as e:
        print(f"Error in pinned entries test: {e}")
        return False


def test_force_evict(module):
    """Test force eviction functionality."""
    try:
        CacheEngineKey = module.CacheEngineKey
        CacheEntry = module.CacheEntry
        policy = module.create_policy()
        cache_dict = {}
        
        # Add entries
        key1 = CacheEngineKey(1)
        key2 = CacheEngineKey(2)
        
        cache_dict[key1] = CacheEntry("v1")
        cache_dict[key2] = CacheEntry("v2")
        
        policy.update_on_put(key1)
        policy.update_on_put(key2)
        
        # Force evict key1
        policy.update_on_force_evict(key1)
        del cache_dict[key1]
        
        # Get eviction candidates - should only return key2
        candidates = policy.get_evict_candidates(cache_dict, num_candidates=2)
        
        if len(candidates) != 1:
            return False
        if candidates[0].key_id != 2:
            return False
        
        return True
    except Exception as e:
        print(f"Error in force evict test: {e}")
        return False


def evaluate_stage1(program_path: str) -> dict:
    """Quick validation with 5 diverse test cases."""
    try:
        module = load_module(program_path)
        
        correctness_tests = []
        performance_scores = []
        
        # Test 1: Basic operations
        correctness_tests.append(test_basic_operations(module))
        
        # Test 2: Eviction order
        correctness_tests.append(test_eviction_order(module))
        
        # Test 3: Pinned entries
        correctness_tests.append(test_pinned_entries(module))
        
        # Test 4: Force evict
        correctness_tests.append(test_force_evict(module))
        
        # Test 5: Small performance test
        elapsed, num_evictions, correct = run_cache_simulation(
            module, 
            cache_size=100, 
            num_operations=1000, 
            hit_ratio=0.7
        )
        correctness_tests.append(correct)
        
        target_time = baseline_measurements["test_config_small"]["target_time"]
        perf_score = sigmoid_performance_score(elapsed, target_time, steepness=2.0)
        performance_scores.append(perf_score)
        
        # Calculate scores
        correctness = sum(correctness_tests) / len(correctness_tests)
        performance = sum(performance_scores) / len(performance_scores) if performance_scores else 0.0
        combined_score = 0.5 * correctness + 0.5 * performance
        
        return {
            "correctness": correctness,
            "performance": performance,
            "combined_score": combined_score,
            "stage": 1.0
        }
    
    except Exception as e:
        print(f"Error in stage1 evaluation: {e}")
        return {
            "correctness": 0.0,
            "performance": 0.0,
            "combined_score": 0.0,
            "stage": 1.0
        }


def evaluate_stage2(program_path: str) -> dict:
    """Comprehensive testing with 10+ test cases including edge cases."""
    try:
        module = load_module(program_path)
        
        correctness_tests = []
        performance_scores = []
        
        # Correctness tests
        # Test 1: Basic operations
        correctness_tests.append(test_basic_operations(module))
        
        # Test 2: Eviction order
        correctness_tests.append(test_eviction_order(module))
        
        # Test 3: Pinned entries
        correctness_tests.append(test_pinned_entries(module))
        
        # Test 4: Force evict
        correctness_tests.append(test_force_evict(module))
        
        # Test 5: Empty cache
        try:
            CacheEngineKey = module.CacheEngineKey
            CacheEntry = module.CacheEntry
            policy = module.create_policy()
            cache_dict = {}
            candidates = policy.get_evict_candidates(cache_dict, num_candidates=1)
            correctness_tests.append(len(candidates) == 0)
        except:
            correctness_tests.append(False)
        
        # Test 6: Single entry
        try:
            policy = module.create_policy()
            cache_dict = {}
            key = CacheEngineKey(1)
            cache_dict[key] = CacheEntry("v1")
            policy.update_on_put(key)
            candidates = policy.get_evict_candidates(cache_dict, num_candidates=1)
            correctness_tests.append(len(candidates) == 1 and candidates[0].key_id == 1)
        except:
            correctness_tests.append(False)
        
        # Test 7: All pinned entries
        try:
            policy = module.create_policy()
            cache_dict = {}
            for i in range(3):
                key = CacheEngineKey(i)
                cache_dict[key] = CacheEntry(f"v{i}", can_evict=False)
                policy.update_on_put(key)
            candidates = policy.get_evict_candidates(cache_dict, num_candidates=5)
            correctness_tests.append(len(candidates) == 0)
        except:
            correctness_tests.append(False)
        
        # Performance tests with baseline measurements
        # Test 8: Small cache simulation
        elapsed, num_evictions, correct = run_cache_simulation(
            module,
            cache_size=100,
            num_operations=1000,
            hit_ratio=0.7
        )
        correctness_tests.append(correct)
        target_time = baseline_measurements["test_config_small"]["target_time"]
        perf_score = sigmoid_performance_score(elapsed, target_time, steepness=2.0)
        performance_scores.append(perf_score)
        
        # Test 9: Medium cache simulation
        elapsed, num_evictions, correct = run_cache_simulation(
            module,
            cache_size=1000,
            num_operations=10000,
            hit_ratio=0.7
        )
        correctness_tests.append(correct)
        target_time = baseline_measurements["test_config_medium"]["target_time"]
        perf_score = sigmoid_performance_score(elapsed, target_time, steepness=2.0)
        performance_scores.append(perf_score)
        
        # Test 10: Large cache simulation
        elapsed, num_evictions, correct = run_cache_simulation(
            module,
            cache_size=5000,
            num_operations=50000,
            hit_ratio=0.7
        )
        correctness_tests.append(correct)
        target_time = baseline_measurements["test_config_large"]["target_time"]
        perf_score = sigmoid_performance_score(elapsed, target_time, steepness=2.0)
        performance_scores.append(perf_score)
        
        # Test 11: High hit ratio
        elapsed, num_evictions, correct = run_cache_simulation(
            module,
            cache_size=500,
            num_operations=5000,
            hit_ratio=0.9
        )
        correctness_tests.append(correct)
        # Scale target time based on medium test
        target_time = baseline_measurements["test_config_medium"]["target_time"] * 0.5
        perf_score = sigmoid_performance_score(elapsed, target_time, steepness=2.0)
        performance_scores.append(perf_score)
        
        # Test 12: Low hit ratio (more evictions)
        elapsed, num_evictions, correct = run_cache_simulation(
            module,
            cache_size=500,
            num_operations=5000,
            hit_ratio=0.3
        )
        correctness_tests.append(correct)
        # Scale target time based on medium test
        target_time = baseline_measurements["test_config_medium"]["target_time"] * 0.5
        perf_score = sigmoid_performance_score(elapsed, target_time, steepness=2.0)
        performance_scores.append(perf_score)
        
        # Calculate scores
        correctness = sum(correctness_tests) / len(correctness_tests)
        performance = sum(performance_scores) / len(performance_scores) if performance_scores else 0.0
        combined_score = 0.5 * correctness + 0.5 * performance
        
        return {
            "correctness": correctness,
            "performance": performance,
            "combined_score": combined_score,
            "stage": 2.0
        }
    
    except Exception as e:
        print(f"Error in stage2 evaluation: {e}")
        import traceback
        traceback.print_exc()
        return {
            "correctness": 0.0,
            "performance": 0.0,
            "combined_score": 0.0,
            "stage": 2.0
        }


def evaluate(program_path: str) -> dict:
    """Main evaluation function."""
    return evaluate_stage2(program_path)
