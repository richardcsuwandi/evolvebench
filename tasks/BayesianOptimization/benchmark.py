"""
This test compares the solutions generated using handwritten evaluator and LLM-generated evaluator 
against the original sampling method from acquisition.py, measuring correctness and speedup for each.
"""

import time
import sys
import os
import numpy as np
from bayes_opt import BayesianOptimization
from bayes_opt.acquisition import UpperConfidenceBound
from numpy.random import RandomState
import warnings
from pathlib import Path


class Tee:
    """A class that writes to both stdout/stderr and a file."""
    def __init__(self, file_path, stream):
        self.file = open(file_path, 'w', encoding='utf-8')
        self.stream = stream
    
    def write(self, data):
        self.stream.write(data)
        self.file.write(data)
        self.file.flush()
    
    def flush(self):
        self.stream.flush()
        self.file.flush()
    
    def close(self):
        self.file.close()

def extract_code_from_evolve_block(file_path):
    """
    Extract code from EVOLVE-BLOCK format files.
    Returns the code content from the FILE section or EVOLVE-BLOCK section.
    """
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Check if this is a multi-file EVOLVE-BLOCK format file (best_program files)
    if '### FILE:' in content:
        # Extract code between ### FILE: ... ### and ### END FILE ###
        start_marker = '### FILE:'
        end_marker = '### END FILE ###'
        
        start_idx = content.find(start_marker)
        if start_idx == -1:
            return content
        
        # Find the start of actual code (after filename line)
        code_start = content.find('\n', start_idx) + 1
        
        # Find the end marker
        end_idx = content.find(end_marker, code_start)
        if end_idx == -1:
            # If no end marker, take everything after the FILE marker
            return content[code_start:]
        
        return content[code_start:end_idx].strip()
    
    # Check if this file has EVOLVE-BLOCK markers (like source.py)
    if '# EVOLVE-BLOCK-START' in content:
        import re
        # Extract code between EVOLVE-BLOCK-START and EVOLVE-BLOCK-END
        pattern = r'# EVOLVE-BLOCK-START[^\n]*\n(.*?)# EVOLVE-BLOCK-END'
        match = re.search(pattern, content, re.DOTALL)
        if match:
            return match.group(1).strip()
    
    return content

def load_acquisition_function_from_file(file_path):
    """Load AcquisitionFunction class from a file, handling both regular files and EVOLVE-BLOCK format"""
    # Extract code (handles EVOLVE-BLOCK format if present)
    code = extract_code_from_evolve_block(file_path)
    
    # Create namespace with necessary imports
    future_annotations_code = "from __future__ import annotations\n"
    
    namespace = {
        '__builtins__': __builtins__,
    }
    
    # Add common imports that might be needed
    try:
        from typing import TYPE_CHECKING, Any, Literal, NoReturn
        namespace.update({
            'TYPE_CHECKING': TYPE_CHECKING,
            'Any': Any,
            'Literal': Literal,
            'NoReturn': NoReturn,
        })
    except ImportError:
        pass
    
    try:
        import abc
        namespace['abc'] = abc
    except ImportError:
        pass
    
    try:
        import warnings as warnings_module
        namespace['warnings'] = warnings_module
    except ImportError:
        pass
    
    try:
        from copy import deepcopy
        namespace['deepcopy'] = deepcopy
    except ImportError:
        pass
    
    try:
        from packaging import version
        namespace['version'] = version
    except ImportError:
        pass
    
    try:
        import scipy
        from scipy import __version__ as scipy_version
        from scipy.optimize._differentialevolution import DifferentialEvolutionSolver, minimize
        from scipy.special import softmax
        from scipy.stats import norm
        namespace.update({
            'scipy': scipy,
            'scipy_version': scipy_version,
            'DifferentialEvolutionSolver': DifferentialEvolutionSolver,
            'minimize': minimize,
            'softmax': softmax,
            'norm': norm,
        })
    except ImportError:
        pass
    
    try:
        from sklearn.gaussian_process import GaussianProcessRegressor
        namespace['GaussianProcessRegressor'] = GaussianProcessRegressor
    except ImportError:
        pass
    
    try:
        from bayes_opt.exception import (
            ConstraintNotSupportedError,
            NoValidPointRegisteredError,
            TargetSpaceEmptyError,
        )
        namespace.update({
            'ConstraintNotSupportedError': ConstraintNotSupportedError,
            'NoValidPointRegisteredError': NoValidPointRegisteredError,
            'TargetSpaceEmptyError': TargetSpaceEmptyError,
        })
    except ImportError:
        pass
    
    try:
        from bayes_opt.target_space import TargetSpace
        namespace['TargetSpace'] = TargetSpace
    except ImportError:
        pass
    
    try:
        from bayes_opt.util import ensure_rng
        namespace['ensure_rng'] = ensure_rng
    except ImportError:
        pass
    
    # Add numpy and RandomState
    namespace['np'] = np
    namespace['RandomState'] = RandomState
    
    # Prepend future annotations if not already present
    if 'from __future__ import annotations' not in code:
        code_to_exec = future_annotations_code + code
    else:
        code_to_exec = code
    
    exec(code_to_exec, namespace)
    
    # Return the AcquisitionFunction class
    if 'AcquisitionFunction' in namespace:
        return namespace['AcquisitionFunction']
    else:
        raise ValueError(f"AcquisitionFunction class not found in {file_path}")

# Global variables to store loaded implementations
_original_acq_func_class = None
_handwritten_acq_func_class = None
_llm_generated_acq_func_class = None

def _create_dummy_acquisition_instance(acq_func_class):
    """Create a concrete instance of AcquisitionFunction by creating a simple subclass"""
    # Check if UpperConfidenceBound is available in the loaded class's module
    # If not, create a minimal concrete subclass
    try:
        # Try to get UpperConfidenceBound from the namespace
        if hasattr(acq_func_class, '__module__'):
            module = sys.modules.get(acq_func_class.__module__)
            if module and hasattr(module, 'UpperConfidenceBound'):
                return module.UpperConfidenceBound()
    except:
        pass
    
    # Create a minimal concrete subclass
    class ConcreteAcquisitionFunction(acq_func_class):
        def base_acq(self, mean, std):
            # Dummy implementation - won't be called for _random_sample_minimize
            return mean + std
    
    return ConcreteAcquisitionFunction()

def original_sampling_method(space, acq, n_random, n_x_seeds, random_state):
    """
    Original implementation from acquisition.py using np.argsort (O(N log N)).
    This extracts the _random_sample_minimize method from the original AcquisitionFunction class.
    """
    global _original_acq_func_class
    if _original_acq_func_class is None:
        # Use the installed package's UpperConfidenceBound directly
        dummy_acq_func = UpperConfidenceBound()
        return dummy_acq_func._random_sample_minimize(acq, space, random_state, n_random, n_x_seeds)
    
    dummy_acq_func = _create_dummy_acquisition_instance(_original_acq_func_class)
    return dummy_acq_func._random_sample_minimize(acq, space, random_state, n_random, n_x_seeds)

def handwritten_sampling_method(space, acq, n_random, n_x_seeds, random_state):
    """
    Implementation from best_program_handwritten.py.
    Uses optimized version with argpartition and sorting for better performance.
    """
    global _handwritten_acq_func_class
    if _handwritten_acq_func_class is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        handwritten_path = os.path.join(script_dir, 'best_program_handwritten.py')
        _handwritten_acq_func_class = load_acquisition_function_from_file(handwritten_path)
    
    dummy_acq_func = _create_dummy_acquisition_instance(_handwritten_acq_func_class)
    return dummy_acq_func._random_sample_minimize(acq, space, random_state, n_random, n_x_seeds)

def llm_generated_sampling_method(space, acq, n_random, n_x_seeds, random_state):
    """
    Implementation from best_program_llm_generated.py.
    Uses optimized version with argpartition for better performance.
    """
    global _llm_generated_acq_func_class
    if _llm_generated_acq_func_class is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        llm_generated_path = os.path.join(script_dir, 'best_program_llm_generated.py')
        _llm_generated_acq_func_class = load_acquisition_function_from_file(llm_generated_path)
    
    dummy_acq_func = _create_dummy_acquisition_instance(_llm_generated_acq_func_class)
    return dummy_acq_func._random_sample_minimize(acq, space, random_state, n_random, n_x_seeds)

def create_real_acquisition_function():
    """Create a real acquisition function using BayesianOptimization."""
    # Set up BayesianOptimization
    pbounds = {'x1': (-3, 3), 'x2': (-3, 3), 'x3': (-3, 3)}

    def test_function(x1, x2, x3):
        return -(x1**2 + x2**2 + x3**2)

    optimizer = BayesianOptimization(f=test_function, pbounds=pbounds)

    # Add some observations to train the GP
    optimizer.probe(params={'x1': 1.0, 'x2': 0.5, 'x3': -1.0}, lazy=False)
    optimizer.probe(params={'x1': -0.5, 'x2': 1.5, 'x3': 0.2}, lazy=False)
    optimizer.probe(params={'x1': 2.1, 'x2': -1.2, 'x3': 1.8}, lazy=False)

    # Fit the Gaussian Process
    optimizer._gp.fit(optimizer.space.params, optimizer.space.target)

    space = optimizer.space
    gp = optimizer._gp
    acquisition = optimizer._acquisition_function
    dim = space.dim

    def acq_func(x_tries):
        """Real acquisition function using GP predictions."""
        x_tries = x_tries.reshape(-1, dim)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mean, std = gp.predict(x_tries, return_std=True)

        # Use the acquisition function's base_acq method
        acq_values = acquisition.base_acq(mean, std)

        # Return negative values since we minimize in the algorithm
        return -1 * acq_values

    return space, acq_func


def benchmark_methods(n_random_values, n_runs=5):
    """Benchmark both methods against the original across different sample sizes."""
    print("Comparing handwritten evaluator and LLM-generated evaluator against original method")
    print("Using real BayesianOptimization acquisition function")
    print()

    space, acq_func = create_real_acquisition_function()
    n_x_seeds = 10  # Typical number of seeds

    results = []

    for n_random in n_random_values:
        print(f"Testing with {n_random:,} samples, {n_x_seeds} seeds:")
        print("-" * 50)

        # Test original method (baseline)
        original_times = []
        for run in range(n_runs):
            random_state = np.random.RandomState(42 + run)

            start_time = time.perf_counter()
            _ = original_sampling_method(
                space, acq_func, n_random, n_x_seeds, random_state
            )
            original_time = time.perf_counter() - start_time
            original_times.append(original_time)

        # Test handwritten method
        handwritten_times = []
        for run in range(n_runs):
            random_state = np.random.RandomState(42 + run)

            start_time = time.perf_counter()
            _ = handwritten_sampling_method(
                space, acq_func, n_random, n_x_seeds, random_state
            )
            handwritten_time = time.perf_counter() - start_time
            handwritten_times.append(handwritten_time)

        # Test LLM-generated method
        llm_generated_times = []
        for run in range(n_runs):
            random_state = np.random.RandomState(42 + run)

            start_time = time.perf_counter()
            _ = llm_generated_sampling_method(
                space, acq_func, n_random, n_x_seeds, random_state
            )
            llm_generated_time = time.perf_counter() - start_time
            llm_generated_times.append(llm_generated_time)

        # Verify correctness - compare against original with same random state
        random_state = np.random.RandomState(42)
        x_min_orig, min_acq_orig, x_seeds_orig = original_sampling_method(
            space, acq_func, n_random, n_x_seeds, random_state
        )

        random_state = np.random.RandomState(42)
        x_min_hand, min_acq_hand, x_seeds_hand = handwritten_sampling_method(
            space, acq_func, n_random, n_x_seeds, random_state
        )

        random_state = np.random.RandomState(42)
        x_min_llm, min_acq_llm, x_seeds_llm = llm_generated_sampling_method(
            space, acq_func, n_random, n_x_seeds, random_state
        )

        # Normalize x_seeds for comparison (handle list vs array differences)
        if isinstance(x_seeds_orig, list):
            x_seeds_orig = np.array(x_seeds_orig) if len(x_seeds_orig) > 0 else np.array([])
        if isinstance(x_seeds_hand, list):
            x_seeds_hand = np.array(x_seeds_hand) if len(x_seeds_hand) > 0 else np.array([])
        if isinstance(x_seeds_llm, list):
            x_seeds_llm = np.array(x_seeds_llm) if len(x_seeds_llm) > 0 else np.array([])

        # Check correctness of handwritten method against original
        x_min_match_hand = (
            (x_min_orig is None and x_min_hand is None) or
            (x_min_orig is not None and x_min_hand is not None and np.allclose(x_min_orig, x_min_hand, atol=1e-10))
        )
        min_acq_match_hand = np.allclose(min_acq_orig, min_acq_hand, atol=1e-10)
        x_seeds_match_hand = (
            (len(x_seeds_orig) == 0 and len(x_seeds_hand) == 0) or
            (len(x_seeds_orig) == len(x_seeds_hand) and len(x_seeds_orig) > 0 and 
             np.allclose(x_seeds_orig, x_seeds_hand, atol=1e-10))
        )
        handwritten_match = x_min_match_hand and min_acq_match_hand and x_seeds_match_hand

        # Check correctness of LLM-generated method against original
        x_min_match_llm = (
            (x_min_orig is None and x_min_llm is None) or
            (x_min_orig is not None and x_min_llm is not None and np.allclose(x_min_orig, x_min_llm, atol=1e-10))
        )
        min_acq_match_llm = np.allclose(min_acq_orig, min_acq_llm, atol=1e-10)
        x_seeds_match_llm = (
            (len(x_seeds_orig) == 0 and len(x_seeds_llm) == 0) or
            (len(x_seeds_orig) == len(x_seeds_llm) and len(x_seeds_orig) > 0 and 
             np.allclose(x_seeds_orig, x_seeds_llm, atol=1e-10))
        )
        llm_match = x_min_match_llm and min_acq_match_llm and x_seeds_match_llm

        # Calculate averages and speedups
        avg_original = np.mean(original_times)
        avg_handwritten = np.mean(handwritten_times)
        avg_llm_generated = np.mean(llm_generated_times)

        handwritten_speedup = avg_original / avg_handwritten
        llm_speedup = avg_original / avg_llm_generated

        handwritten_improvement_pct = (handwritten_speedup - 1) * 100
        llm_improvement_pct = (llm_speedup - 1) * 100

        # Print results
        print(f"Original method:               {avg_original:.4f}s ± {np.std(original_times):.4f}s")
        print(f"Handwritten evaluator:         {avg_handwritten:.4f}s ± {np.std(handwritten_times):.4f}s")
        print(f"  → Speedup: {handwritten_speedup:.2f}x ({handwritten_improvement_pct:.1f}% {'faster' if handwritten_speedup > 1 else 'slower'})")
        print(f"  → Correctness: {'✓ MATCH' if handwritten_match else '✗ MISMATCH'}")
        print(f"LLM-generated evaluator:       {avg_llm_generated:.4f}s ± {np.std(llm_generated_times):.4f}s")
        print(f"  → Speedup: {llm_speedup:.2f}x ({llm_improvement_pct:.1f}% {'faster' if llm_speedup > 1 else 'slower'})")
        print(f"  → Correctness: {'✓ MATCH' if llm_match else '✗ MISMATCH'}")
        print()

        results.append({
            'n_random': n_random,
            'original_time': avg_original,
            'handwritten_time': avg_handwritten,
            'llm_generated_time': avg_llm_generated,
            'handwritten_speedup': handwritten_speedup,
            'llm_speedup': llm_speedup,
            'handwritten_improvement_pct': handwritten_improvement_pct,
            'llm_improvement_pct': llm_improvement_pct,
            'handwritten_match': handwritten_match,
            'llm_match': llm_match
        })

    return results


def main():
    """Run the performance comparison."""
    print("=" * 80)
    print("BAYESIAN OPTIMIZATION ACQUISITION SAMPLING BENCHMARK")
    print("=" * 80)
    
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Load programs
    baseline_path = os.path.join(script_dir, 'bayes_opt', 'acquisition.py')
    handwritten_path = os.path.join(script_dir, 'best_program_handwritten.py')
    llm_generated_path = os.path.join(script_dir, 'best_program_llm_generated.py')

    print(f"\n📂 Loading programs:")
    print(f"   Baseline (initial): {baseline_path}")
    print(f"   Handwritten (best): {handwritten_path}")
    print(f"   LLM Generated (best): {llm_generated_path}")
    
    # Pre-load all implementations to catch any import errors early
    print(f"\n🔬 Loading implementations...")
    try:
        global _original_acq_func_class, _handwritten_acq_func_class, _llm_generated_acq_func_class
        # For baseline, we'll use the installed package directly (set to None to use default)
        _original_acq_func_class = None  # Will use UpperConfidenceBound directly
        print(f"   ✅ Baseline (using installed package)")
        _handwritten_acq_func_class = load_acquisition_function_from_file(handwritten_path)
        print(f"   ✅ Handwritten loaded")
        _llm_generated_acq_func_class = load_acquisition_function_from_file(llm_generated_path)
        print(f"   ✅ LLM Generated loaded")
    except Exception as e:
        print(f"   ❌ Error loading implementations: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test with different sample sizes
    n_random_values = [1000, 5000, 10000, 20000]

    results = benchmark_methods(n_random_values)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    total_handwritten_improvement = 0
    total_llm_improvement = 0
    all_handwritten_match = True
    all_llm_match = True

    for result in results:
        print(f"\nN={result['n_random']:,}:")
        print(f"  Handwritten: {result['handwritten_speedup']:.2f}x speedup "
              f"({result['handwritten_improvement_pct']:.1f}% {'faster' if result['handwritten_speedup'] > 1 else 'slower'}) "
              f"- {'✓ CORRECT' if result['handwritten_match'] else '✗ INCORRECT'}")
        print(f"  LLM-generated: {result['llm_speedup']:.2f}x speedup "
              f"({result['llm_improvement_pct']:.1f}% {'faster' if result['llm_speedup'] > 1 else 'slower'}) "
              f"- {'✓ CORRECT' if result['llm_match'] else '✗ INCORRECT'}")
        
        total_handwritten_improvement += result['handwritten_improvement_pct']
        total_llm_improvement += result['llm_improvement_pct']
        all_handwritten_match = all_handwritten_match and result['handwritten_match']
        all_llm_match = all_llm_match and result['llm_match']

    avg_handwritten_improvement = total_handwritten_improvement / len(results)
    avg_llm_improvement = total_llm_improvement / len(results)

    print(f"\n" + "-" * 70)
    print(f"Average speedup (handwritten): {avg_handwritten_improvement:.1f}%")
    print(f"Average speedup (LLM-generated): {avg_llm_improvement:.1f}%")
    print(f"\nOverall correctness:")
    print(f"  Handwritten: {'✓ ALL CORRECT' if all_handwritten_match else '✗ SOME INCORRECT'}")
    print(f"  LLM-generated: {'✓ ALL CORRECT' if all_llm_match else '✗ SOME INCORRECT'}")

if __name__ == "__main__":
    # Capture output to both terminal and output.txt
    script_dir = Path(__file__).parent
    output_file = script_dir / "output.txt"
    
    # Save original stdout and stderr
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    
    # Create Tee objects to write to both terminal and file
    tee_stdout = Tee(output_file, original_stdout)
    tee_stderr = Tee(output_file, original_stderr)
    
    try:
        sys.stdout = tee_stdout
        sys.stderr = tee_stderr
        main()
    finally:
        # Restore original stdout and stderr
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        tee_stdout.close()
        tee_stderr.close()