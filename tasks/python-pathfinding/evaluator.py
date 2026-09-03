"""
Evaluator for Python Pathfinding Multi-File Evolution

This evaluator tests the pathfinding repository performance by importing
and testing the actual modified files. It measures both correctness
(successful pathfinding) and performance (speed).

NOTE: For multi-file evolution, the 'code' parameter is ignored.
The evaluator tests the repository files directly.
"""

import sys
import time
import importlib
import numpy as np
from pathlib import Path
from typing import Dict

# The pathfinding directory path should be set via --directory in the CLI
# The evaluator will test whichever pathfinding installation is in the Python path


def create_random_grid(size: int = 50, obstacle_ratio: float = 0.3):
    """Create a random grid with obstacles for testing"""
    try:
        from pathfinding.core.grid import Grid
        matrix = np.random.random((size, size)) > obstacle_ratio
        return Grid(matrix=matrix)
    except ImportError as e:
        raise ImportError(
            "Could not import pathfinding.core.grid.Grid. "
            "Ensure python-pathfinding is installed and in the Python path."
        ) from e


def find_random_path(grid):
    """Find a path between random start and end points"""
    from pathfinding.finder.a_star import AStarFinder
    
    size = grid.width
    
    # Find walkable nodes
    walkable_nodes = []
    for x in range(size):
        for y in range(size):
            if grid.walkable(x, y):
                walkable_nodes.append((x, y))
    
    if len(walkable_nodes) < 2:
        return None, 0
    
    # Pick random start and end
    start_idx = np.random.randint(0, len(walkable_nodes))
    end_idx = np.random.randint(0, len(walkable_nodes))
    while end_idx == start_idx:
        end_idx = np.random.randint(0, len(walkable_nodes))
    
    start = grid.node(*walkable_nodes[start_idx])
    end = grid.node(*walkable_nodes[end_idx])
    
    # Find path and measure time
    finder = AStarFinder()
    start_time = time.perf_counter()
    path, runs = finder.find_path(start, end, grid)
    elapsed = time.perf_counter() - start_time
    
    return path, elapsed


def reload_pathfinding_modules():
    """Reload pathfinding modules to pick up file changes"""
    pathfinding_modules = [
        name for name in list(sys.modules.keys()) 
        if name.startswith('pathfinding')
    ]
    
    for name in pathfinding_modules:
        if name in sys.modules:
            try:
                importlib.reload(sys.modules[name])
            except Exception:
                # Some modules may not reload cleanly, that's ok
                pass


def evaluate_stage1(code: str) -> Dict[str, float]:
    """
    Stage 1: Quick validation with small grids
    
    Args:
        code: Ignored for multi-file evolution
        
    Returns:
        Dict with correctness and performance metrics
    """
    try:
        # Reload modules to pick up any changes
        reload_pathfinding_modules()
        
        # Test on small grids with multiple trials
        total_time = 0
        success_count = 0
        total_trials = 5
        
        for _ in range(total_trials):
            grid = create_random_grid(size=15, obstacle_ratio=0.2)
            path, elapsed = find_random_path(grid)
            
            if path is not None and len(path) > 0:
                success_count += 1
                total_time += elapsed
        
        if success_count == 0:
            return {"correctness": 0.0, "performance": 0.0}
        
        # Calculate metrics
        correctness = success_count / total_trials
        avg_time = total_time / success_count
        
        # Performance score (lower time is better, normalize to 0-1)
        # Baseline: ~0.002s per path, Target: ~0.001s per path
        best_time = 0.001
        worst_time = 0.004
        
        if avg_time <= best_time:
            performance = 1.0
        elif avg_time >= worst_time:
            performance = 0.0
        else:
            performance = 1.0 - (avg_time - best_time) / (worst_time - best_time)
        
        # Calculate combined score using weights from task.yaml (correctness: 0.5, performance: 0.5)
        combined_score = (correctness * 0.5) + (performance * 0.5)
        
        return {
            "correctness": correctness,
            "performance": max(0.0, min(1.0, performance)),
            "combined_score": combined_score
        }
        
    except Exception as e:
        print(f"Stage 1 evaluation error: {e}")
        return {"correctness": 0.0, "performance": 0.0, "combined_score": 0.0}


def evaluate_stage2(code: str) -> Dict[str, float]:
    """
    Stage 2: Comprehensive performance testing with various grid sizes
    
    Args:
        code: Ignored for multi-file evolution
        
    Returns:
        Dict with correctness and performance metrics
    """
    try:
        # Reload modules to pick up any changes
        reload_pathfinding_modules()
        
        # Test on various grid sizes and obstacle densities
        test_configs = [
            (20, 0.2, 3),   # Small, sparse
            (20, 0.4, 3),   # Small, dense
            (40, 0.3, 5),   # Medium
            (60, 0.3, 3),   # Large
        ]
        
        total_time = 0
        total_paths = 0
        successful_paths = 0
        
        for size, obstacle_ratio, trials in test_configs:
            for _ in range(trials):
                grid = create_random_grid(size=size, obstacle_ratio=obstacle_ratio)
                path, elapsed = find_random_path(grid)
                
                if path is not None and len(path) > 0:
                    total_time += elapsed
                    successful_paths += 1
                total_paths += 1
        
        if successful_paths == 0:
            return {"correctness": 0.0, "performance": 0.0}
        
        # Calculate metrics
        correctness = successful_paths / total_paths
        avg_time = total_time / successful_paths
        
        # Performance score (lower time is better, normalize to 0-1)
        # Comprehensive testing baseline
        best_time = 0.002
        worst_time = 0.008
        
        if avg_time <= best_time:
            performance = 1.0
        elif avg_time >= worst_time:
            performance = 0.0
        else:
            performance = 1.0 - (avg_time - best_time) / (worst_time - best_time)
        
        # Calculate combined score using weights from task.yaml (correctness: 0.5, performance: 0.5)
        combined_score = (correctness * 0.5) + (performance * 0.5)
        
        return {
            "correctness": correctness,
            "performance": max(0.0, min(1.0, performance)),
            "combined_score": combined_score
        }
        
    except Exception as e:
        print(f"Stage 2 evaluation error: {e}")
        return {"correctness": 0.0, "performance": 0.0, "combined_score": 0.0}


def evaluate(code: str) -> Dict[str, float]:
    """
    Main evaluation function for compatibility
    
    Args:
        code: Ignored for multi-file evolution
        
    Returns:
        Dict with correctness and performance metrics
    """
    # For multi-file evolution, we skip stage1 and go directly to stage2
    # since the repository should already be functional
    stage1_result = evaluate_stage1(code)
    
    # If stage1 correctness is too low, return early
    if stage1_result["correctness"] < 0.5:
        return stage1_result
    
    # Otherwise run comprehensive stage2 evaluation
    return evaluate_stage2(code)