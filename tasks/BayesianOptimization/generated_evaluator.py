import sys
import importlib.util
import time
import numpy as np
from typing import Any, Callable
from numpy.random import RandomState
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C


def load_module(program_path: str):
    """Load the module from the given path."""
    spec = importlib.util.spec_from_file_location("solution_module", program_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["solution_module"] = module
    spec.loader.exec_module(module)
    return module


class MockTargetSpace:
    """Mock TargetSpace for testing."""
    def __init__(self, params, target, bounds, constraint=None, constraint_values=None):
        self.params = params
        self.target = target
        self.bounds = bounds
        self.constraint = constraint
        self._constraint_values = constraint_values
        self._dim = params.shape[1] if len(params.shape) > 1 else 1
    
    def __len__(self):
        return len(self.params)
    
    def random_sample(self, n, random_state=None):
        """Generate random samples within bounds."""
        if random_state is None:
            random_state = np.random.RandomState()
        samples = []
        for _ in range(n):
            sample = []
            for low, high in self.bounds:
                sample.append(random_state.uniform(low, high))
            samples.append(sample)
        return np.array(samples)


class ConcreteAcquisitionFunction:
    """Concrete implementation for testing."""
    def __init__(self, random_state=None):
        self.i = 0
    
    def base_acq(self, mean, std):
        """Simple acquisition function: mean + std."""
        return mean + std
    
    def _fit_gp(self, gp, target_space):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gp.fit(target_space.params, target_space.target)
    
    def _get_acq(self, gp, constraint=None):
        dim = gp.X_train_.shape[1]
        if constraint is not None:
            def acq(x):
                x = x.reshape(-1, dim)
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    mean, std = gp.predict(x, return_std=True)
                    p_constraints = constraint.predict(x)
                return -1 * self.base_acq(mean, std) * p_constraints
        else:
            def acq(x):
                x = x.reshape(-1, dim)
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    mean, std = gp.predict(x, return_std=True)
                return -1 * self.base_acq(mean, std)
        return acq


def create_test_case(n_samples=10, n_dims=2, seed=42):
    """Create a test case with mock data."""
    rs = RandomState(seed)
    params = rs.uniform(-5, 5, (n_samples, n_dims))
    target = np.sum(params**2, axis=1)  # Simple quadratic function
    bounds = [(-5, 5) for _ in range(n_dims)]
    
    return {
        "input": {
            "params": params,
            "target": target,
            "bounds": bounds,
            "n_random": 100,
            "n_x_seeds": 5,
            "seed": seed
        },
        "output": {
            "returns_tuple": True,
            "tuple_length": 3,
            "x_min_shape": (n_dims,),
            "min_acq_is_float": True,
            "x_seeds_shape": (5, n_dims)
        }
    }


def evaluate_stage1(program_path: str) -> dict:
    """Quick validation with 5 diverse test cases."""
    try:
        module = load_module(program_path)
        
        # Get the function to test
        if hasattr(module, 'AcquisitionFunction'):
            AcqClass = module.AcquisitionFunction
        else:
            return {"correctness": 0.0, "performance": 0.0, "combined_score": 0.0, "stage": 1.0}
        
        # Create concrete subclass
        class TestAcq(AcqClass):
            def base_acq(self, mean, std):
                return mean + std
            
            def get_acquisition_params(self):
                return {}
            
            def set_acquisition_params(self, **params):
                pass
        
        test_cases = [
            create_test_case(n_samples=10, n_dims=2, seed=42),
            create_test_case(n_samples=20, n_dims=3, seed=123),
            create_test_case(n_samples=15, n_dims=1, seed=456),
            create_test_case(n_samples=5, n_dims=4, seed=789),
            create_test_case(n_samples=30, n_dims=2, seed=999),
        ]
        
        passed = 0
        total_time = 0.0
        
        for test_case in test_cases:
            try:
                inp = test_case["input"]
                expected = test_case["output"]
                
                # Create mock objects
                target_space = MockTargetSpace(
                    inp["params"], inp["target"], inp["bounds"]
                )
                
                # Create GP
                kernel = C(1.0) * RBF(1.0)
                gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=0)
                gp.fit(inp["params"], inp["target"])
                
                # Create acquisition function instance
                acq_func = TestAcq()
                
                # Get the acquisition function
                acq = acq_func._get_acq(gp, constraint=None)
                
                # Test _random_sample_minimize
                start_time = time.time()
                result = acq_func._random_sample_minimize(
                    acq, target_space, RandomState(inp["seed"]),
                    inp["n_random"], inp["n_x_seeds"]
                )
                elapsed = time.time() - start_time
                total_time += elapsed
                
                # Validate output
                if not isinstance(result, tuple) or len(result) != 3:
                    continue
                
                x_min, min_acq, x_seeds = result
                
                if x_min is None or not isinstance(min_acq, (float, np.floating)):
                    continue
                
                if x_min.shape != expected["x_min_shape"]:
                    continue
                
                if x_seeds.shape != expected["x_seeds_shape"]:
                    continue
                
                passed += 1
                
            except Exception:
                continue
        
        correctness = passed / len(test_cases)
        avg_time = total_time / len(test_cases) if len(test_cases) > 0 else 1.0
        performance = 1.0 / (1.0 + avg_time * 10)
        combined_score = 0.7 * correctness + 0.3 * performance
        
        return {
            "correctness": correctness,
            "performance": performance,
            "combined_score": combined_score,
            "stage": 1.0
        }
        
    except Exception:
        return {"correctness": 0.0, "performance": 0.0, "combined_score": 0.0, "stage": 1.0}


def evaluate_stage2(program_path: str) -> dict:
    """Comprehensive testing with 10+ test cases including edge cases."""
    try:
        module = load_module(program_path)
        
        # Get the function to test
        if hasattr(module, 'AcquisitionFunction'):
            AcqClass = module.AcquisitionFunction
        else:
            return {"correctness": 0.0, "performance": 0.0, "combined_score": 0.0, "stage": 2.0}
        
        # Create concrete subclass
        class TestAcq(AcqClass):
            def base_acq(self, mean, std):
                return mean + std
            
            def get_acquisition_params(self):
                return {}
            
            def set_acquisition_params(self, **params):
                pass
        
        test_cases = [
            create_test_case(n_samples=10, n_dims=2, seed=42),
            create_test_case(n_samples=20, n_dims=3, seed=123),
            create_test_case(n_samples=15, n_dims=1, seed=456),
            create_test_case(n_samples=5, n_dims=4, seed=789),
            create_test_case(n_samples=30, n_dims=2, seed=999),
            create_test_case(n_samples=50, n_dims=5, seed=111),
            create_test_case(n_samples=8, n_dims=3, seed=222),
            create_test_case(n_samples=25, n_dims=2, seed=333),
            create_test_case(n_samples=12, n_dims=6, seed=444),
            create_test_case(n_samples=40, n_dims=3, seed=555),
        ]
        
        # Edge cases
        edge_cases = [
            create_test_case(n_samples=3, n_dims=1, seed=666),  # Minimal samples
            create_test_case(n_samples=100, n_dims=10, seed=777),  # Large dimensions
        ]
        test_cases.extend(edge_cases)
        
        passed = 0
        total_time = 0.0
        
        for test_case in test_cases:
            try:
                inp = test_case["input"]
                expected = test_case["output"]
                
                # Create mock objects
                target_space = MockTargetSpace(
                    inp["params"], inp["target"], inp["bounds"]
                )
                
                # Create GP
                kernel = C(1.0) * RBF(1.0)
                gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=0)
                gp.fit(inp["params"], inp["target"])
                
                # Create acquisition function instance
                acq_func = TestAcq()
                
                # Get the acquisition function
                acq = acq_func._get_acq(gp, constraint=None)
                
                # Test _random_sample_minimize
                start_time = time.time()
                result = acq_func._random_sample_minimize(
                    acq, target_space, RandomState(inp["seed"]),
                    inp["n_random"], inp["n_x_seeds"]
                )
                elapsed = time.time() - start_time
                total_time += elapsed
                
                # Validate output
                if not isinstance(result, tuple) or len(result) != 3:
                    continue
                
                x_min, min_acq, x_seeds = result
                
                if x_min is None or not isinstance(min_acq, (float, np.floating)):
                    continue
                
                if x_min.shape != expected["x_min_shape"]:
                    continue
                
                if x_seeds.shape != expected["x_seeds_shape"]:
                    continue
                
                # Additional validation: check that x_min is within bounds
                valid_bounds = True
                for i, (low, high) in enumerate(inp["bounds"]):
                    if not (low <= x_min[i] <= high):
                        valid_bounds = False
                        break
                
                if not valid_bounds:
                    continue
                
                passed += 1
                
            except Exception:
                continue
        
        correctness = passed / len(test_cases)
        avg_time = total_time / len(test_cases) if len(test_cases) > 0 else 1.0
        performance = 1.0 / (1.0 + avg_time * 10)
        combined_score = 0.7 * correctness + 0.3 * performance
        
        return {
            "correctness": correctness,
            "performance": performance,
            "combined_score": combined_score,
            "stage": 2.0
        }
        
    except Exception:
        return {"correctness": 0.0, "performance": 0.0, "combined_score": 0.0, "stage": 2.0}


def evaluate(program_path: str) -> dict:
    """Main evaluation function."""
    return evaluate_stage2(program_path)
