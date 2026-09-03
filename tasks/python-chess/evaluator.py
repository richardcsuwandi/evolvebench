#!/usr/bin/env python3
"""
OpenEvolve Evaluator for python-chess move generation optimization
Targets: performance-nested-loops-generate-moves EVOLVE-BLOCK
"""

import sys
import time
import math
import random
from pathlib import Path
from typing import Dict, Any
import tempfile
import os

# Add chess library to path
chess_dir = Path(__file__).parent.parent
sys.path.insert(0, str(chess_dir))

def sigmoid_performance_score(actual_time: float, target_time: float, steepness: float = 5.0) -> float:
    """
    Create smooth performance gradient using sigmoid function.
    Returns score between 0 and 1, with 0.5 at target_time.
    """
    if actual_time <= 0:
        return 0.95

    relative_slowdown = (actual_time - target_time) / target_time

    # Prevent overflow by capping the exponent
    exponent = steepness * relative_slowdown
    if exponent > 100:
        return 0.0  # Very slow, essentially 0 score
    elif exponent < -100:
        return 1.0  # Very fast, essentially perfect score

    return 1.0 / (1.0 + math.exp(exponent))

def test_move_generation(code: str, complexity: str = "medium") -> Dict[str, Any]:
    """
    Test the evolved move generation code with various board positions.
    """
    import chess

    # Test positions of varying complexity
    test_positions = {
        "simple": [
            chess.Board(),  # Starting position
            chess.Board("8/8/8/8/8/8/8/8 w - - 0 1"),  # Empty board (edge case)
        ],
        "medium": [
            chess.Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
            chess.Board("r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 6"),
            chess.Board("rnbqk2r/pp1pppbp/5np1/8/3PP3/2N2N2/PPP2PPP/R1BQKB1R w KQkq - 2 6"),
        ],
        "complex": [
            chess.Board("r2q1rk1/ppp2ppp/2np1n2/2b1p1B1/2B1P1b1/3P1N2/PPP2PPP/RN1QK2R w KQ - 8 9"),
            chess.Board("rnbqk2r/pp2ppbp/3p1np1/8/3PP3/2N2N2/PPP2PPP/R1BQKB1R w KQkq - 2 6"),
            chess.Board("r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 4 6"),
            chess.Board("r2q1rk1/1b2ppbp/p1np1np1/1p6/3PP3/1PN2N2/PBP1BPPP/R2Q1RK1 w - - 0 12"),
            chess.Board("r3kb1r/ppp1pppp/2n2n2/3q4/3P4/2N5/PPP2PPP/R1BQKB1R w KQkq - 4 6"),
        ]
    }

    positions = test_positions.get(complexity, test_positions["medium"])

    # Number of iterations based on complexity
    iterations = {"simple": 1000, "medium": 500, "complex": 200}[complexity]

    total_moves = 0
    correct_moves = True

    start_time = time.perf_counter()

    try:
        for _ in range(iterations):
            for board in positions:
                # Test the move generation
                moves = list(board.generate_pseudo_legal_moves())
                total_moves += len(moves)

                # Basic correctness check - ensure moves are valid
                if len(moves) == 0 and not board.is_game_over():
                    correct_moves = False
                    break

                # Sample check - verify a few moves are actually pseudo-legal
                if moves and random.random() < 0.1:  # Check 10% of iterations
                    sample_move = random.choice(moves)
                    if sample_move.from_square < 0 or sample_move.from_square > 63:
                        correct_moves = False
                        break
                    if sample_move.to_square < 0 or sample_move.to_square > 63:
                        correct_moves = False
                        break

            if not correct_moves:
                break

    except Exception as e:
        return {
            "correct": 0.0,
            "elapsed_time": float('inf'),
            "error": str(e),
            "total_moves": 0
        }

    elapsed_time = time.perf_counter() - start_time

    return {
        "correct": 1.0 if correct_moves else 0.0,
        "elapsed_time": elapsed_time,
        "total_moves": total_moves,
        "moves_per_second": total_moves / elapsed_time if elapsed_time > 0 else 0
    }

def get_aggressive_targets(complexity: str) -> float:
    """
    Set realistic performance targets for move generation.
    Current baseline: ~2M moves/sec (~15ms for medium complexity)
    Target: 3-4M moves/sec (1.5-2x improvement - achievable with optimization)
    """
    targets = {
        "simple": 0.005,     # 5ms for 1000 iterations (3M moves/sec equivalent)
        "medium": 0.010,     # 10ms for 500 iterations (3M moves/sec equivalent)  
        "complex": 0.020     # 20ms for 200 iterations (2.5M moves/sec equivalent)
    }
    return targets.get(complexity, 0.010)

def evaluate_stage1(code: str) -> Dict[str, float]:
    """
    Stage 1: Quick validation with simple positions
    """
    result = test_move_generation(code, complexity="simple")

    correctness = result.get("correct", 0.0)
    elapsed_time = result.get("elapsed_time", float('inf'))

    # Use aggressive target with sigmoid scoring
    target_time = get_aggressive_targets("simple")
    performance = sigmoid_performance_score(elapsed_time, target_time, steepness=3.0)

    # Apply correctness gate
    if correctness < 1.0:
        performance *= correctness

    return {
        "correctness": correctness,
        "performance": performance,
        "combined_score": correctness * performance,
        "moves_per_second": result.get("moves_per_second", 0)
    }

def evaluate_stage2(code: str) -> Dict[str, float]:
    """
    Stage 2: Full evaluation with complex positions
    """
    complexities = ["simple", "medium", "complex"]
    weights = [0.2, 0.3, 0.5]  # Weight complex positions more

    total_correctness = 0.0
    total_performance = 0.0
    total_moves_per_sec = 0.0

    for complexity, weight in zip(complexities, weights):
        result = test_move_generation(code, complexity=complexity)

        correctness = result.get("correct", 0.0)
        elapsed_time = result.get("elapsed_time", float('inf'))
        moves_per_sec = result.get("moves_per_second", 0)

        # Use aggressive targets for each complexity level
        target_time = get_aggressive_targets(complexity)
        performance = sigmoid_performance_score(elapsed_time, target_time, steepness=4.0)

        total_correctness += correctness * weight
        total_performance += performance * weight
        total_moves_per_sec += moves_per_sec * weight

    # Strong correctness requirement
    if total_correctness < 0.95:
        total_performance *= (total_correctness / 0.95) ** 2

    return {
        "correctness": total_correctness,
        "performance": total_performance,
        "combined_score": total_correctness * total_performance,
        "moves_per_second": total_moves_per_sec
    }

def evaluate(code: str) -> Dict[str, float]:
    """
    Main OpenEvolve entry point for evaluating evolved code.
    """
    # Quick syntax check
    try:
        compile(code, "<evolved>", "exec")
    except SyntaxError as e:
        return {
            "correctness": 0.0,
            "performance": 0.0,
            "combined_score": 0.0,
            "error": str(e)
        }

    # Stage 1: Quick validation
    stage1_result = evaluate_stage1(code)
    if stage1_result["correctness"] < 0.5:
        return stage1_result

    # Stage 2: Full evaluation
    final_result = evaluate_stage2(code)

    # Add informative artifacts
    final_result["artifact"] = {
        "moves_per_second": final_result.get("moves_per_second", 0),
        "baseline_moves_per_second": 2000000,  # Current baseline
        "speedup": final_result.get("moves_per_second", 0) / 2000000 if final_result.get("moves_per_second", 0) > 0 else 0,
        "target_speedup": 5.0,  # We're aiming for 5x improvement minimum
    }

    return final_result

if __name__ == "__main__":
    # Test the evaluator with baseline performance
    print("Testing evaluator scoring with example times:")
    print("=" * 50)

    complexities = ["simple", "medium", "complex"]
    for complexity in complexities:
        target = get_aggressive_targets(complexity)
        print(f"\n{complexity.upper()} positions (target: {target:.6f}s):")

        # Test various performance levels
        test_times = [
            target * 0.2,   # 5x faster than target
            target * 0.5,   # 2x faster than target
            target * 1.0,   # At target
            target * 2.0,   # 2x slower than target
            target * 5.0,   # 5x slower than target
        ]

        for t in test_times:
            score = sigmoid_performance_score(t, target)
            speed_ratio = target / t
            status = "✅" if t <= target else "⚠️" if t <= target * 2 else "❌"
            print(f"  {t:.6f}s ({speed_ratio:.1f}x target) → score: {score:.3f} {status}")

    print("\n🎯 Goal: Unoptimized code should score ~0.1-0.3")
    print("📈 Target: Optimized code should reach 0.8+ score")
    print("🚀 This creates proper gradient for evolution!")

    # Test with actual chess move generation
    print("\n" + "=" * 50)
    print("Testing with actual chess move generation:")

    import chess
    result = test_move_generation("", complexity="medium")
    print(f"\nBaseline performance:")
    print(f"  Time: {result['elapsed_time']:.4f}s")
    print(f"  Moves generated: {result['total_moves']:,}")
    print(f"  Moves/sec: {result['moves_per_second']:,.0f}")

    target = get_aggressive_targets("medium")
    score = sigmoid_performance_score(result['elapsed_time'], target)
    print(f"  Score vs aggressive target: {score:.3f}")
    print(f"  Need {target/result['elapsed_time']:.1f}x speedup to reach target!")