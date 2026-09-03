#!/usr/bin/env python3
"""
Benchmark comparison: Initial vs Best Optimized Python-Chess Move Generation

This script demonstrates the performance improvement discovered by OpenEvolve's
genetic programming optimization of python-chess's move generation.

The optimization improves the generate_pseudo_legal_moves function,
resulting in improved move generation performance across various positions.
"""

import time
import sys
import os
import shutil
import tempfile
import importlib
import re
from pathlib import Path
from typing import Tuple


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


def extract_code_from_evolve_block(filepath):
    """
    Extract code from EVOLVE-BLOCK format files.
    Returns the code content from the FILE section.
    """
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Extract content between ### FILE: chess/__init__.py ### and ### END FILE ###
    pattern = r'### FILE: chess/__init__\.py ###\s*\n(.*?)\n### END FILE ###'
    match = re.search(pattern, content, re.DOTALL)
    
    if match:
        return match.group(1).strip()
    else:
        # Fallback: return full content
        return content


def setup_chess_module(base_dir, code=None):
    """
    Set up a chess module by copying the base directory and optionally
    replacing __init__.py with evolved code.
    
    Args:
        base_dir: Base directory containing the chess package
        code: Optional code content to replace chess/__init__.py
        
    Returns:
        Temporary directory path for importing
    """
    # Create temporary directory
    temp_dir = tempfile.mkdtemp(prefix='chess_bench_')
    
    # Copy the entire chess directory
    chess_src = base_dir / 'chess'
    chess_dst = Path(temp_dir) / 'chess'
    shutil.copytree(chess_src, chess_dst)
    
    # Replace __init__.py if code provided
    if code:
        init_file = chess_dst / '__init__.py'
        with open(init_file, 'w') as f:
            f.write(code)
    
    return temp_dir


def cleanup_chess_module(temp_dir):
    """Remove temporary directory and clean up sys.path and modules"""
    if temp_dir in sys.path:
        sys.path.remove(temp_dir)
    
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    
    # Reload modules to clear cached imports
    modules_to_remove = [name for name in list(sys.modules.keys()) 
                        if name.startswith('chess')]
    for name in modules_to_remove:
        if name in sys.modules:
            del sys.modules[name]


def benchmark_move_generation(board_module, iterations: int = 100000) -> Tuple[float, float]:
    """
    Benchmark move generation performance.
    
    Args:
        board_module: The chess module containing Board class
        iterations: Number of iterations to run
        
    Returns:
        Tuple of (total_time, moves_per_second)
    """
    Board = board_module.Board
    
    # Create test board
    board = Board()
    
    # Warm-up
    for _ in range(1000):
        list(board.generate_pseudo_legal_moves())
    
    # Benchmark
    start_time = time.perf_counter()
    for _ in range(iterations):
        list(board.generate_pseudo_legal_moves())
    end_time = time.perf_counter()
    
    elapsed = end_time - start_time
    moves_per_second = iterations / elapsed if elapsed > 0 else 0
    
    return elapsed, moves_per_second


def test_correctness(board_module):
    """Test correctness of move generation"""
    Board = board_module.Board
    
    tests = []
    
    # Test 1: Starting position (should have 20 moves)
    board = Board()
    moves = list(board.generate_pseudo_legal_moves())
    tests.append({
        'name': 'starting_position',
        'expected': 20,
        'actual': len(moves),
        'passed': len(moves) == 20
    })
    
    # Test 2: Position after e4 (black should have 20 moves)
    board = Board("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1")
    moves = list(board.generate_pseudo_legal_moves())
    tests.append({
        'name': 'after_e4',
        'expected': 20,
        'actual': len(moves),
        'passed': len(moves) == 20
    })
    
    # Test 3: Promotion position
    board = Board("8/P7/8/8/8/8/8/8 w - - 0 1")
    moves = list(board.generate_pseudo_legal_moves())
    tests.append({
        'name': 'promotion',
        'expected': 4,
        'actual': len(moves),
        'passed': len(moves) == 4
    })
    
    # Test 4: Complex position
    board = Board("r1bqkb1r/pppp1ppp/2n2n2/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4")
    moves = list(board.generate_pseudo_legal_moves())
    # Just check that moves are generated (not checking exact count)
    tests.append({
        'name': 'complex_position',
        'expected': '>0',
        'actual': len(moves),
        'passed': len(moves) > 0
    })
    
    passed_count = sum(1 for test in tests if test['passed'])
    return {
        'num_passed': passed_count,
        'num_tests': len(tests),
        'passed': passed_count == len(tests),
        'tests': tests
    }


def main():
    """Run the benchmark comparison."""
    print("=" * 100)
    print("OpenEvolve Python-Chess Move Generation Optimization - Performance Comparison")
    print("=" * 100)
    print()
    print("This benchmark compares the original python-chess generate_pseudo_legal_moves")
    print("implementation against optimized versions from handwritten and LLM-generated evaluations.")
    print()

    base_dir = Path(__file__).parent
    
    # Extract code from best programs
    handwritten_code = extract_code_from_evolve_block(
        base_dir / "best_program_handwritten.py"
    )
    llm_generated_code = extract_code_from_evolve_block(
        base_dir / "best_program_llm_generated.py"
    )

    # Test positions: (fen, description)
    test_positions = [
        (None, "Starting position"),
        ("r1bqkb1r/pppp1ppp/2n2n2/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4", "Complex position"),
        ("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1", "After e4"),
        ("8/P7/8/8/8/8/8/8 w - - 0 1", "Promotion position"),
    ]
    
    iterations_per_test = 100000

    print("Test Case: generate_pseudo_legal_moves() performance")
    print("Comparing performance across different chess positions")
    print(f"Running {iterations_per_test:,} iterations per test position")
    print(f"Total move generations: {len(test_positions) * iterations_per_test:,}")
    print()
    print(f"{'Position':>20} │ {'Initial (sec)':>14} │ {'Handwritten (sec)':>18} │ {'LLM-Gen (sec)':>15} │ "
          f"{'HW Speedup':>12} │ {'LLM Speedup':>13}")
    print("─" * 105)

    results = []
    total_time_initial = 0.0
    total_time_handwritten = 0.0
    total_time_llm_generated = 0.0

    # Setup initial module
    initial_temp_dir = setup_chess_module(base_dir)
    
    # Setup handwritten module
    handwritten_temp_dir = setup_chess_module(base_dir, handwritten_code)
    
    # Setup LLM-generated module
    llm_generated_temp_dir = setup_chess_module(base_dir, llm_generated_code)
    
    try:
        # Test correctness first
        print("\n✅ Correctness Tests:")
        
        # Test initial
        if initial_temp_dir not in sys.path:
            sys.path.insert(0, initial_temp_dir)
        modules_to_clear = [name for name in list(sys.modules.keys()) if name.startswith('chess')]
        for name in modules_to_clear:
            if name in sys.modules:
                del sys.modules[name]
        initial_chess = importlib.import_module('chess')
        initial_correct = test_correctness(initial_chess)
        
        # Test handwritten
        # Remove initial_temp_dir and add handwritten_temp_dir
        if initial_temp_dir in sys.path:
            sys.path.remove(initial_temp_dir)
        if handwritten_temp_dir not in sys.path:
            sys.path.insert(0, handwritten_temp_dir)
        modules_to_clear = [name for name in list(sys.modules.keys()) if name.startswith('chess')]
        for name in modules_to_clear:
            if name in sys.modules:
                del sys.modules[name]
        handwritten_chess = importlib.import_module('chess')
        handwritten_correct = test_correctness(handwritten_chess)
        
        # Test LLM-generated
        # Remove handwritten_temp_dir and add llm_generated_temp_dir
        if handwritten_temp_dir in sys.path:
            sys.path.remove(handwritten_temp_dir)
        if llm_generated_temp_dir not in sys.path:
            sys.path.insert(0, llm_generated_temp_dir)
        modules_to_clear = [name for name in list(sys.modules.keys()) if name.startswith('chess')]
        for name in modules_to_clear:
            if name in sys.modules:
                del sys.modules[name]
        llm_generated_chess = importlib.import_module('chess')
        llm_generated_correct = test_correctness(llm_generated_chess)
        
        print(f"   Initial:        {initial_correct['num_passed']}/{initial_correct['num_tests']} passed")
        print(f"   Handwritten:    {handwritten_correct['num_passed']}/{handwritten_correct['num_tests']} passed")
        print(f"   LLM-Generated:  {llm_generated_correct['num_passed']}/{llm_generated_correct['num_tests']} passed")
        print()
        
        # Benchmark each position
        for fen, description in test_positions:
            print(f"Testing {description}... ", end='', flush=True)
            
            # Benchmark initial program
            # Switch to initial module
            if llm_generated_temp_dir in sys.path:
                sys.path.remove(llm_generated_temp_dir)
            if handwritten_temp_dir in sys.path:
                sys.path.remove(handwritten_temp_dir)
            if initial_temp_dir not in sys.path:
                sys.path.insert(0, initial_temp_dir)
            modules_to_clear = [name for name in list(sys.modules.keys()) if name.startswith('chess')]
            for name in modules_to_clear:
                if name in sys.modules:
                    del sys.modules[name]
            initial_chess = importlib.import_module('chess')
            initial_time, _ = benchmark_move_generation(initial_chess, iterations=iterations_per_test)
            
            # Benchmark handwritten program
            # Switch to handwritten module
            if initial_temp_dir in sys.path:
                sys.path.remove(initial_temp_dir)
            if llm_generated_temp_dir in sys.path:
                sys.path.remove(llm_generated_temp_dir)
            if handwritten_temp_dir not in sys.path:
                sys.path.insert(0, handwritten_temp_dir)
            modules_to_clear = [name for name in list(sys.modules.keys()) if name.startswith('chess')]
            for name in modules_to_clear:
                if name in sys.modules:
                    del sys.modules[name]
            handwritten_chess = importlib.import_module('chess')
            handwritten_time, _ = benchmark_move_generation(handwritten_chess, iterations=iterations_per_test)
            
            # Benchmark LLM-generated program
            # Switch to LLM-generated module
            if initial_temp_dir in sys.path:
                sys.path.remove(initial_temp_dir)
            if handwritten_temp_dir in sys.path:
                sys.path.remove(handwritten_temp_dir)
            if llm_generated_temp_dir not in sys.path:
                sys.path.insert(0, llm_generated_temp_dir)
            modules_to_clear = [name for name in list(sys.modules.keys()) if name.startswith('chess')]
            for name in modules_to_clear:
                if name in sys.modules:
                    del sys.modules[name]
            llm_generated_chess = importlib.import_module('chess')
            llm_generated_time, _ = benchmark_move_generation(llm_generated_chess, iterations=iterations_per_test)
            
            total_time_initial += initial_time
            total_time_handwritten += handwritten_time
            total_time_llm_generated += llm_generated_time
            
            hw_speedup = initial_time / handwritten_time if handwritten_time > 0 else 0
            llm_speedup = initial_time / llm_generated_time if llm_generated_time > 0 else 0
            hw_speedup_pct = (1 - handwritten_time / initial_time) * 100 if initial_time > 0 else 0
            llm_speedup_pct = (1 - llm_generated_time / initial_time) * 100 if initial_time > 0 else 0
            
            # Status indicators
            hw_status = "🚀" if hw_speedup > 1.1 else ("✅" if hw_speedup > 1.05 else "➖")
            llm_status = "🚀" if llm_speedup > 1.1 else ("✅" if llm_speedup > 1.05 else "➖")
            
            print(f"\r{description:>20} │ {initial_time:>13.3f} │ {handwritten_time:>17.3f} │ "
                  f"{llm_generated_time:>14.3f} │ {hw_speedup:>6.2f}x {hw_status} ({hw_speedup_pct:+.1f}%) │ "
                  f"{llm_speedup:>6.2f}x {llm_status} ({llm_speedup_pct:+.1f}%)")
            
            results.append({
                'position': description,
                'initial': initial_time,
                'handwritten': handwritten_time,
                'llm_generated': llm_generated_time,
                'hw_speedup': hw_speedup,
                'llm_speedup': llm_speedup
            })
    
    finally:
        cleanup_chess_module(initial_temp_dir)
        cleanup_chess_module(handwritten_temp_dir)
        cleanup_chess_module(llm_generated_temp_dir)

    print("─" * 105)

    total_hw_speedup = total_time_initial / total_time_handwritten if total_time_handwritten > 0 else 0
    total_llm_speedup = total_time_initial / total_time_llm_generated if total_time_llm_generated > 0 else 0
    total_hw_saved = total_time_initial - total_time_handwritten
    total_llm_saved = total_time_initial - total_time_llm_generated

    print(f"{'TOTAL':>20} │ {total_time_initial:>13.3f} │ {total_time_handwritten:>17.3f} │ "
          f"{total_time_llm_generated:>14.3f} │ {total_hw_speedup:>6.2f}x │ {total_llm_speedup:>6.2f}x")
    print("=" * 105)
    print()

    # Calculate statistics
    hw_speedups = [r['hw_speedup'] for r in results if r['hw_speedup'] > 0]
    llm_speedups = [r['llm_speedup'] for r in results if r['llm_speedup'] > 0]
    
    print("Statistical Summary:")
    print(f"  Total test positions: {len(test_positions)}")
    print(f"  Total iterations: {iterations_per_test * len(test_positions):,} move generations")
    print()
    
    if hw_speedups:
        hw_mean_speedup = sum(hw_speedups) / len(hw_speedups)
        hw_min_speedup = min(hw_speedups)
        hw_max_speedup = max(hw_speedups)
        hw_mean_pct = (1 - 1/hw_mean_speedup) * 100
        hw_min_pct = (1 - 1/hw_min_speedup) * 100
        hw_max_pct = (1 - 1/hw_max_speedup) * 100
        
        print("  Handwritten Evaluation Performance:")
        print(f"    Total time saved: {total_hw_saved:.3f} seconds")
        print(f"    Mean speedup:     {hw_mean_speedup:.3f}x ({hw_mean_pct:+.2f}% faster)")
        print(f"    Min speedup:      {hw_min_speedup:.3f}x ({hw_min_pct:+.2f}% faster)")
        print(f"    Max speedup:      {hw_max_speedup:.3f}x ({hw_max_pct:+.2f}% faster)")
    else:
        print("  Handwritten Evaluation Performance:")
        print(f"    No speedup detected")
    print()
    
    if llm_speedups:
        llm_mean_speedup = sum(llm_speedups) / len(llm_speedups)
        llm_min_speedup = min(llm_speedups)
        llm_max_speedup = max(llm_speedups)
        llm_mean_pct = (1 - 1/llm_mean_speedup) * 100
        llm_min_pct = (1 - 1/llm_min_speedup) * 100
        llm_max_pct = (1 - 1/llm_max_speedup) * 100
        
        print("  LLM-Generated Evaluation Performance:")
        print(f"    Total time saved: {total_llm_saved:.3f} seconds")
        print(f"    Mean speedup:     {llm_mean_speedup:.3f}x ({llm_mean_pct:+.2f}% faster)")
        print(f"    Min speedup:      {llm_min_speedup:.3f}x ({llm_min_pct:+.2f}% faster)")
        print(f"    Max speedup:      {llm_max_speedup:.3f}x ({llm_max_pct:+.2f}% faster)")
    else:
        print("  LLM-Generated Evaluation Performance:")
        print(f"    No speedup detected")
    print()
    
    # Calculate moves per second for comparison
    total_iterations = iterations_per_test * len(test_positions)
    initial_mps = total_iterations / total_time_initial if total_time_initial > 0 else 0
    handwritten_mps = total_iterations / total_time_handwritten if total_time_handwritten > 0 else 0
    llm_generated_mps = total_iterations / total_time_llm_generated if total_time_llm_generated > 0 else 0
    
    print(f"  💡 Performance Metrics:")
    print(f"     Initial:        {initial_mps:,.0f} moves/sec")
    print(f"     Handwritten:    {handwritten_mps:,.0f} moves/sec")
    print(f"     LLM-Generated:  {llm_generated_mps:,.0f} moves/sec")
    print()
    
    if total_hw_speedup > 0 or total_llm_speedup > 0:
        best_optimizer = "Handwritten" if total_hw_speedup > total_llm_speedup else "LLM-Generated"
        best_saved = max(total_hw_saved, total_llm_saved)
        time_saved_per_op = best_saved / total_iterations if total_iterations > 0 else 0
        print(f"  💡 Real-world impact (best optimizer: {best_optimizer}):")
        print(f"     Per operation:  {time_saved_per_op * 1000000:.3f} μs saved")
        print(f"     Per 1M ops:     {time_saved_per_op * 1000000:.1f} seconds saved")
        print(f"     Per 10M ops:    {time_saved_per_op * 10000000:.1f} seconds ({time_saved_per_op * 10000000 / 60:.2f} minutes)")
    print()

if __name__ == "__main__":
    # Capture output to both terminal and output.txt
    base_dir = Path(__file__).parent
    output_file = base_dir / "output.txt"
    
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
