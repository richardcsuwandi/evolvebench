import math
import time
import importlib.util
import sys
import os
import random
from typing import Dict, Any, List, Tuple


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


# Baseline measurements - to be populated after first run
BASELINE_MEASUREMENTS = {}


def load_program_from_string(program_code: str):
    """Load the program module from a string."""
    import tempfile
    
    # Create a temporary directory structure
    temp_dir = tempfile.mkdtemp()
    
    # Parse the multi-file program
    files = {}
    current_file = None
    current_content = []
    
    for line in program_code.split('\n'):
        if line.startswith('### FILE:'):
            if current_file:
                files[current_file] = '\n'.join(current_content)
            current_file = line.split('### FILE:')[1].strip().split('###')[0].strip()
            current_content = []
        elif line.startswith('### END FILE ###'):
            if current_file:
                files[current_file] = '\n'.join(current_content)
            current_file = None
            current_content = []
        elif current_file:
            current_content.append(line)
    
    # Write files to temp directory
    for filepath, content in files.items():
        full_path = os.path.join(temp_dir, filepath)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, 'w') as f:
            f.write(content)
    
    # Add temp directory to sys.path
    sys.path.insert(0, temp_dir)
    
    # Import required modules
    from pathfinding.core import heap, grid, graph
    
    return heap, grid, graph


def load_program(program_path: str):
    """Load the program module from the given path."""
    # Read the program file
    with open(program_path, 'r') as f:
        program_code = f.read()
    
    return load_program_from_string(program_code)


def create_test_heap(heap_module, grid_module, graph_module, grid_type='grid', size=10):
    """Create a test heap with the given grid type."""
    if grid_type == 'grid':
        test_grid = grid_module.Grid(width=size, height=size)
        start_node = test_grid.node(0, 0)
    elif grid_type == 'graph':
        test_graph = graph_module.Graph()
        # Add nodes to graph
        for i in range(size):
            test_graph.nodes[i] = graph_module.Node(identifier=i, x=i, y=0)
        start_node = test_graph.node(0)
        test_grid = test_graph
    else:
        raise ValueError(f"Unknown grid type: {grid_type}")
    
    # Import the SimpleHeap class from the module
    SimpleHeap = heap_module.SimpleHeap
    test_heap = SimpleHeap(start_node, test_grid)
    return test_heap, test_grid, start_node


def test_heap_operations(heap_module, grid_module, graph_module, grid_type='grid', num_operations=100):
    """Test heap operations and measure performance."""
    test_heap, test_grid, start_node = create_test_heap(heap_module, grid_module, graph_module, grid_type, size=20)
    
    start_time = time.perf_counter()
    
    # Push multiple nodes with different f values
    nodes_to_push = []
    for i in range(num_operations):
        if grid_type == 'grid':
            x, y = i % test_grid.width, i // test_grid.width
            if x < test_grid.width and y < test_grid.height:
                node = test_grid.node(x, y)
                node.f = random.randint(1, 100)
                nodes_to_push.append(node)
                test_heap.push_node(node)
        else:
            if i < len(test_grid.nodes):
                node = test_grid.node(i)
                node.f = random.randint(1, 100)
                nodes_to_push.append(node)
                test_heap.push_node(node)
    
    # Remove some nodes
    for i in range(min(10, len(nodes_to_push))):
        node = nodes_to_push[i]
        test_heap.remove_node(node, node.f)
    
    # Pop nodes and verify they come out in order
    popped_nodes = []
    prev_f = -1
    while len(test_heap.open_list) > len(test_heap.removed_node_tuples):
        try:
            node = test_heap.pop_node()
            popped_nodes.append(node)
            if node.f < prev_f:
                return False, 0  # Not in correct order
            prev_f = node.f
        except IndexError:
            break
    
    elapsed_time = time.perf_counter() - start_time
    
    return True, elapsed_time


def test_pathfinding_with_heap(heap_module, grid_module, graph_module, grid_size=20):
    """Test pathfinding using the heap implementation."""
    try:
        # Create a grid
        test_grid = grid_module.Grid(width=grid_size, height=grid_size)
        start = test_grid.node(0, 0)
        end = test_grid.node(grid_size - 1, grid_size - 1)
        
        start.g = 0
        start.f = 0
        start.opened = True
        
        SimpleHeap = heap_module.SimpleHeap
        open_list = SimpleHeap(start, test_grid)
        
        start_time = time.perf_counter()
        
        # Simulate A* pathfinding
        visited = set()
        found = False
        
        while len(open_list.open_list) > len(open_list.removed_node_tuples):
            try:
                node = open_list.pop_node()
            except IndexError:
                break
            
            node_id = (node.x, node.y)
            if node_id in visited:
                continue
            visited.add(node_id)
            
            if node == end:
                found = True
                break
            
            neighbors = test_grid.neighbors(node)
            for neighbor in neighbors:
                neighbor_id = (neighbor.x, neighbor.y)
                if neighbor_id not in visited:
                    ng = node.g + 1
                    if not neighbor.opened or ng < neighbor.g:
                        old_f = neighbor.f
                        neighbor.g = ng
                        neighbor.h = abs(neighbor.x - end.x) + abs(neighbor.y - end.y)
                        neighbor.f = neighbor.g + neighbor.h
                        neighbor.parent = node
                        
                        if not neighbor.opened:
                            open_list.push_node(neighbor)
                            neighbor.opened = True
                        else:
                            open_list.remove_node(neighbor, old_f)
                            open_list.push_node(neighbor)
        
        elapsed_time = time.perf_counter() - start_time
        
        return found, elapsed_time
    except Exception as e:
        return False, float('inf')


def measure_baseline_performance(program_path: str):
    """Measure baseline performance if not already done."""
    global BASELINE_MEASUREMENTS
    
    if BASELINE_MEASUREMENTS:
        return
    
    try:
        heap_module, grid_module, graph_module = load_program(program_path)
        
        # Test 1: Small heap operations (grid)
        _, time1 = test_heap_operations(heap_module, grid_module, graph_module, 'grid', 50)
        BASELINE_MEASUREMENTS['heap_ops_grid_small'] = {'target_time': max(time1, 0.0001)}
        
        # Test 2: Medium heap operations (grid)
        _, time2 = test_heap_operations(heap_module, grid_module, graph_module, 'grid', 100)
        BASELINE_MEASUREMENTS['heap_ops_grid_medium'] = {'target_time': max(time2, 0.0001)}
        
        # Test 3: Small heap operations (graph)
        _, time3 = test_heap_operations(heap_module, grid_module, graph_module, 'graph', 50)
        BASELINE_MEASUREMENTS['heap_ops_graph_small'] = {'target_time': max(time3, 0.0001)}
        
        # Test 4: Small pathfinding
        _, time4 = test_pathfinding_with_heap(heap_module, grid_module, graph_module, 10)
        BASELINE_MEASUREMENTS['pathfinding_small'] = {'target_time': max(time4, 0.0001)}
        
        # Test 5: Medium pathfinding
        _, time5 = test_pathfinding_with_heap(heap_module, grid_module, graph_module, 20)
        BASELINE_MEASUREMENTS['pathfinding_medium'] = {'target_time': max(time5, 0.0001)}
        
    except Exception as e:
        # Set default baselines if measurement fails
        BASELINE_MEASUREMENTS = {
            'heap_ops_grid_small': {'target_time': 0.001},
            'heap_ops_grid_medium': {'target_time': 0.002},
            'heap_ops_graph_small': {'target_time': 0.001},
            'pathfinding_small': {'target_time': 0.002},
            'pathfinding_medium': {'target_time': 0.005},
        }


def evaluate_stage1(program_path: str) -> dict:
    """Quick validation with 5 diverse test cases."""
    try:
        # Measure baseline if needed
        measure_baseline_performance(program_path)
        
        heap_module, grid_module, graph_module = load_program(program_path)
        
        correctness_scores = []
        performance_scores = []
        
        # Test 1: Basic heap operations with grid
        try:
            correct, elapsed = test_heap_operations(heap_module, grid_module, graph_module, 'grid', 50)
            correctness_scores.append(1.0 if correct else 0.0)
            target_time = BASELINE_MEASUREMENTS['heap_ops_grid_small']['target_time']
            performance_scores.append(sigmoid_performance_score(elapsed, target_time, 2.0))
        except Exception:
            correctness_scores.append(0.0)
            performance_scores.append(0.0)
        
        # Test 2: Basic heap operations with graph
        try:
            correct, elapsed = test_heap_operations(heap_module, grid_module, graph_module, 'graph', 50)
            correctness_scores.append(1.0 if correct else 0.0)
            target_time = BASELINE_MEASUREMENTS['heap_ops_graph_small']['target_time']
            performance_scores.append(sigmoid_performance_score(elapsed, target_time, 2.0))
        except Exception:
            correctness_scores.append(0.0)
            performance_scores.append(0.0)
        
        # Test 3: Small pathfinding
        try:
            correct, elapsed = test_pathfinding_with_heap(heap_module, grid_module, graph_module, 10)
            correctness_scores.append(1.0 if correct else 0.0)
            target_time = BASELINE_MEASUREMENTS['pathfinding_small']['target_time']
            performance_scores.append(sigmoid_performance_score(elapsed, target_time, 2.0))
        except Exception:
            correctness_scores.append(0.0)
            performance_scores.append(0.0)
        
        # Test 4: Medium heap operations
        try:
            correct, elapsed = test_heap_operations(heap_module, grid_module, graph_module, 'grid', 100)
            correctness_scores.append(1.0 if correct else 0.0)
            target_time = BASELINE_MEASUREMENTS['heap_ops_grid_medium']['target_time']
            performance_scores.append(sigmoid_performance_score(elapsed, target_time, 2.0))
        except Exception:
            correctness_scores.append(0.0)
            performance_scores.append(0.0)
        
        # Test 5: Medium pathfinding
        try:
            correct, elapsed = test_pathfinding_with_heap(heap_module, grid_module, graph_module, 20)
            correctness_scores.append(1.0 if correct else 0.0)
            target_time = BASELINE_MEASUREMENTS['pathfinding_medium']['target_time']
            performance_scores.append(sigmoid_performance_score(elapsed, target_time, 2.0))
        except Exception:
            correctness_scores.append(0.0)
            performance_scores.append(0.0)
        
        correctness = sum(correctness_scores) / len(correctness_scores) if correctness_scores else 0.0
        performance = sum(performance_scores) / len(performance_scores) if performance_scores else 0.0
        combined_score = 0.5 * correctness + 0.5 * performance
        
        return {
            'correctness': correctness,
            'performance': performance,
            'combined_score': combined_score,
            'stage': 1.0
        }
    
    except Exception as e:
        return {
            'correctness': 0.0,
            'performance': 0.0,
            'combined_score': 0.0,
            'stage': 1.0,
            'error': str(e)
        }


def evaluate_stage2(program_path: str) -> dict:
    """Comprehensive testing with 10+ test cases including edge cases."""
    try:
        # Measure baseline if needed
        measure_baseline_performance(program_path)
        
        heap_module, grid_module, graph_module = load_program(program_path)
        
        correctness_scores = []
        performance_scores = []
        
        # Test 1: Small heap operations with grid
        try:
            correct, elapsed = test_heap_operations(heap_module, grid_module, graph_module, 'grid', 50)
            correctness_scores.append(1.0 if correct else 0.0)
            target_time = BASELINE_MEASUREMENTS['heap_ops_grid_small']['target_time']
            performance_scores.append(sigmoid_performance_score(elapsed, target_time, 2.0))
        except Exception:
            correctness_scores.append(0.0)
            performance_scores.append(0.0)
        
        # Test 2: Medium heap operations with grid
        try:
            correct, elapsed = test_heap_operations(heap_module, grid_module, graph_module, 'grid', 100)
            correctness_scores.append(1.0 if correct else 0.0)
            target_time = BASELINE_MEASUREMENTS['heap_ops_grid_medium']['target_time']
            performance_scores.append(sigmoid_performance_score(elapsed, target_time, 2.0))
        except Exception:
            correctness_scores.append(0.0)
            performance_scores.append(0.0)
        
        # Test 3: Large heap operations with grid
        try:
            correct, elapsed = test_heap_operations(heap_module, grid_module, graph_module, 'grid', 200)
            correctness_scores.append(1.0 if correct else 0.0)
            target_time = BASELINE_MEASUREMENTS['heap_ops_grid_medium']['target_time'] * 2
            performance_scores.append(sigmoid_performance_score(elapsed, target_time, 2.0))
        except Exception:
            correctness_scores.append(0.0)
            performance_scores.append(0.0)
        
        # Test 4: Small heap operations with graph
        try:
            correct, elapsed = test_heap_operations(heap_module, grid_module, graph_module, 'graph', 50)
            correctness_scores.append(1.0 if correct else 0.0)
            target_time = BASELINE_MEASUREMENTS['heap_ops_graph_small']['target_time']
            performance_scores.append(sigmoid_performance_score(elapsed, target_time, 2.0))
        except Exception:
            correctness_scores.append(0.0)
            performance_scores.append(0.0)
        
        # Test 5: Medium heap operations with graph
        try:
            correct, elapsed = test_heap_operations(heap_module, grid_module, graph_module, 'graph', 100)
            correctness_scores.append(1.0 if correct else 0.0)
            target_time = BASELINE_MEASUREMENTS['heap_ops_graph_small']['target_time'] * 2
            performance_scores.append(sigmoid_performance_score(elapsed, target_time, 2.0))
        except Exception:
            correctness_scores.append(0.0)
            performance_scores.append(0.0)
        
        # Test 6: Small pathfinding
        try:
            correct, elapsed = test_pathfinding_with_heap(heap_module, grid_module, graph_module, 10)
            correctness_scores.append(1.0 if correct else 0.0)
            target_time = BASELINE_MEASUREMENTS['pathfinding_small']['target_time']
            performance_scores.append(sigmoid_performance_score(elapsed, target_time, 2.0))
        except Exception:
            correctness_scores.append(0.0)
            performance_scores.append(0.0)
        
        # Test 7: Medium pathfinding
        try:
            correct, elapsed = test_pathfinding_with_heap(heap_module, grid_module, graph_module, 20)
            correctness_scores.append(1.0 if correct else 0.0)
            target_time = BASELINE_MEASUREMENTS['pathfinding_medium']['target_time']
            performance_scores.append(sigmoid_performance_score(elapsed, target_time, 2.0))
        except Exception:
            correctness_scores.append(0.0)
            performance_scores.append(0.0)
        
        # Test 8: Large pathfinding
        try:
            correct, elapsed = test_pathfinding_with_heap(heap_module, grid_module, graph_module, 30)
            correctness_scores.append(1.0 if correct else 0.0)
            target_time = BASELINE_MEASUREMENTS['pathfinding_medium']['target_time'] * 2
            performance_scores.append(sigmoid_performance_score(elapsed, target_time, 2.0))
        except Exception:
            correctness_scores.append(0.0)
            performance_scores.append(0.0)
        
        # Test 9: Stress test - many removals
        try:
            test_heap, test_grid, start_node = create_test_heap(heap_module, grid_module, graph_module, 'grid', 20)
            nodes = []
            for i in range(100):
                x, y = i % test_grid.width, i // test_grid.width
                if x < test_grid.width and y < test_grid.height:
                    node = test_grid.node(x, y)
                    node.f = i
                    nodes.append(node)
                    test_heap.push_node(node)
            
            start_time = time.perf_counter()
            # Remove half the nodes
            for i in range(0, len(nodes), 2):
                test_heap.remove_node(nodes[i], nodes[i].f)
            
            # Pop remaining nodes
            popped = []
            while len(test_heap.open_list) > len(test_heap.removed_node_tuples):
                try:
                    popped.append(test_heap.pop_node())
                except IndexError:
                    break
            elapsed = time.perf_counter() - start_time
            
            # Verify correctness
            correct = len(popped) == len(nodes) // 2
            correctness_scores.append(1.0 if correct else 0.0)
            target_time = BASELINE_MEASUREMENTS['heap_ops_grid_medium']['target_time']
            performance_scores.append(sigmoid_performance_score(elapsed, target_time, 2.0))
        except Exception:
            correctness_scores.append(0.0)
            performance_scores.append(0.0)
        
        # Test 10: Edge case - single node
        try:
            test_heap, test_grid, start_node = create_test_heap(heap_module, grid_module, graph_module, 'grid', 5)
            start_time = time.perf_counter()
            node = test_heap.pop_node()
            elapsed = time.perf_counter() - start_time
            
            correct = node == start_node
            correctness_scores.append(1.0 if correct else 0.0)
            target_time = 0.00001
            performance_scores.append(sigmoid_performance_score(elapsed, target_time, 2.0))
        except Exception:
            correctness_scores.append(0.0)
            performance_scores.append(0.0)
        
        # Test 11: Random f values
        try:
            test_heap, test_grid, start_node = create_test_heap(heap_module, grid_module, graph_module, 'grid', 15)
            random.seed(42)
            nodes = []
            for i in range(50):
                x, y = i % test_grid.width, i // test_grid.width
                if x < test_grid.width and y < test_grid.height:
                    node = test_grid.node(x, y)
                    node.f = random.random() * 100
                    nodes.append(node)
                    test_heap.push_node(node)
            
            start_time = time.perf_counter()
            popped = []
            prev_f = -1
            correct = True
            while len(test_heap.open_list) > len(test_heap.removed_node_tuples):
                try:
                    node = test_heap.pop_node()
                    if node.f < prev_f:
                        correct = False
                        break
                    prev_f = node.f
                    popped.append(node)
                except IndexError:
                    break
            elapsed = time.perf_counter() - start_time
            
            correctness_scores.append(1.0 if correct else 0.0)
            target_time = BASELINE_MEASUREMENTS['heap_ops_grid_small']['target_time']
            performance_scores.append(sigmoid_performance_score(elapsed, target_time, 2.0))
        except Exception:
            correctness_scores.append(0.0)
            performance_scores.append(0.0)
        
        # Test 12: Multiple push/pop cycles
        try:
            test_heap, test_grid, start_node = create_test_heap(heap_module, grid_module, graph_module, 'grid', 10)
            start_time = time.perf_counter()
            correct = True
            
            for cycle in range(5):
                # Push nodes
                for i in range(10):
                    x, y = (cycle * 10 + i) % test_grid.width, (cycle * 10 + i) // test_grid.width
                    if x < test_grid.width and y < test_grid.height:
                        node = test_grid.node(x, y)
                        node.f = i
                        test_heap.push_node(node)
                
                # Pop some nodes
                for _ in range(5):
                    try:
                        test_heap.pop_node()
                    except IndexError:
                        break
            
            elapsed = time.perf_counter() - start_time
            
            correctness_scores.append(1.0 if correct else 0.0)
            target_time = BASELINE_MEASUREMENTS['heap_ops_grid_medium']['target_time']
            performance_scores.append(sigmoid_performance_score(elapsed, target_time, 2.0))
        except Exception:
            correctness_scores.append(0.0)
            performance_scores.append(0.0)
        
        correctness = sum(correctness_scores) / len(correctness_scores) if correctness_scores else 0.0
        performance = sum(performance_scores) / len(performance_scores) if performance_scores else 0.0
        combined_score = 0.5 * correctness + 0.5 * performance
        
        return {
            'correctness': correctness,
            'performance': performance,
            'combined_score': combined_score,
            'stage': 2.0
        }
    
    except Exception as e:
        return {
            'correctness': 0.0,
            'performance': 0.0,
            'combined_score': 0.0,
            'stage': 2.0,
            'error': str(e)
        }


def evaluate(program_path: str) -> dict:
    """Main evaluation function."""
    return evaluate_stage2(program_path)
