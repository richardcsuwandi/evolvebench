#!/usr/bin/env python3
"""
Evaluator for nested-loops-parse-source optimization

This evaluator properly extracts and applies evolved parse_source method optimizations
by dynamically replacing the method in the Parser class.
"""

import sys
import time
import math
import random
import tempfile
import importlib.util
import os
import re
import types
from pathlib import Path
from typing import Dict, Any

# Add parent directory to path to import marko modules
sys.path.insert(0, str(Path(__file__).parent.parent))

def sigmoid_performance_score(actual_time: float, target_time: float) -> float:
    """
    Create smooth performance gradient using steeper sigmoid function.
    At target_time: score = 0.5
    2x faster: score ≈ 0.95
    2x slower: score ≈ 0.05
    """
    if actual_time <= 0:
        return 0.98  # Nearly perfect for instantaneous

    relative_slowdown = (actual_time - target_time) / target_time
    # Steeper sigmoid for better gradient
    return 1.0 / (1.0 + math.exp(5 * relative_slowdown))

def generate_test_markdown(size: str) -> str:
    """Generate diverse markdown documents for testing parser performance."""

    sizes = {
        "small": 50,
        "medium": 200,
        "large": 500
    }

    num_elements = sizes.get(size, 50)
    content = []

    # Generate diverse markdown elements to trigger worst-case parsing
    for i in range(num_elements):
        element_type = random.choice([
            "heading", "paragraph", "code_block", "list", "quote", "hr", "table"
        ])

        if element_type == "heading":
            level = random.randint(1, 6)
            content.append(f"{'#' * level} Heading {i}")

        elif element_type == "paragraph":
            content.append(f"This is paragraph {i} with some **bold** and *italic* text.")

        elif element_type == "code_block":
            lang = random.choice(["python", "javascript", "bash", ""])
            content.extend([
                f"```{lang}",
                f"# Code block {i}",
                f"def function_{i}():",
                f"    return {i}",
                "```"
            ])

        elif element_type == "list":
            content.extend([
                f"- List item {i}.1",
                f"- List item {i}.2",
                f"  - Nested item {i}.2.1"
            ])

        elif element_type == "quote":
            content.append(f"> This is a quote block {i}")

        elif element_type == "hr":
            content.append("---")

        elif element_type == "table":
            content.extend([
                f"| Column 1 | Column 2 | Column 3 |",
                f"|----------|----------|----------|",
                f"| Row {i}.1 | Data | Value |",
                f"| Row {i}.2 | More | Info |"
            ])

        # Add spacing
        if random.random() < 0.3:
            content.append("")

    return "\n".join(content)

def get_performance_target(size: str) -> float:
    """Aggressive performance targets for nested loop parser optimization."""
    # Based on test results showing 3.5x+ growth factor
    # Set targets 3x faster than typical baseline
    targets = {
        "small": 0.002,    # 2ms for 50 elements
        "medium": 0.008,   # 8ms for 200 elements
        "large": 0.025,    # 25ms for 500 elements
    }
    return targets.get(size, 0.008)

def extract_parse_source_method(evolved_code: str) -> str:
    """Extract just the parse_source method content from the evolved code."""
    # Look for the method definition within the EVOLVE-BLOCK
    evolve_block_pattern = r'# EVOLVE-BLOCK-START id="performance-nested-loops-parse-source"(.*?)# EVOLVE-BLOCK-END'
    match = re.search(evolve_block_pattern, evolved_code, re.DOTALL)

    if match:
        block_content = match.group(1).strip()
        print(f"  📦 Extracted evolve block ({len(block_content)} chars)")
        return block_content

    print(f"  ⚠️  Could not extract parse_source method from evolved code")
    return ""

def create_evolved_parser(evolved_method_code: str):
    """Create a parser with the evolved parse_source method properly applied."""
    import marko
    from marko.parser import Parser

    if not evolved_method_code.strip():
        print(f"  ⚠️  No evolved method code, using original parser")
        return marko.Markdown()

    try:
        print(f"  🔄 Applying evolved optimization to parser...")

        # Create a new parser class that inherits from the original
        class EvolvedParser(Parser):
            def parse_source(self, source):
                """Parse the source into a list of block elements with evolved optimization."""
                # Import the necessary modules that the method needs
                from marko import block

                element_list = self._build_block_element_list()
                ast = []

                # Execute the evolved code within the method context
                # This replaces the original nested loop implementation
                local_vars = {
                    'source': source,
                    'element_list': element_list,
                    'ast': ast,
                    'block': block,
                    'hasattr': hasattr
                }

                try:
                    # Execute the evolved optimization code
                    exec(evolved_method_code, globals(), local_vars)
                    return local_vars['ast']
                except Exception as e:
                    print(f"    ⚠️  Evolved code execution failed: {e}, falling back to original")
                    # Fall back to original implementation
                    return super().parse_source(source)

        # Create markdown parser with evolved parser class
        parser = marko.Markdown()
        # Replace the parser instance with our evolved version
        parser.parser = EvolvedParser()

        print(f"  ✅ Evolved parser successfully created and applied")
        return parser

    except Exception as e:
        print(f"  ⚠️  Failed to create evolved parser: {e}")
        import traceback
        traceback.print_exc()
        return marko.Markdown()

def test_parser_correctness(parser_obj) -> float:
    """Test correctness by parsing known markdown structures."""
    test_cases = [
        # Basic elements
        "# Heading\n\nParagraph with **bold** text.",

        # Code blocks
        "```python\ndef hello():\n    print('world')\n```",

        # Lists
        "- Item 1\n- Item 2\n  - Nested",

        # Mixed content
        "# Title\n\nText with *emphasis*.\n\n```\ncode\n```\n\n> Quote",

        # Complex nesting
        "# Main\n\n## Sub\n\nText\n\n- List\n  - Sub\n\n> Quote\n\n```bash\necho test\n```",

        # Edge cases
        "",  # Empty
        "   \n  \n",  # Whitespace only
        "Single line without newline",
    ]

    correct = 0
    total = len(test_cases)

    try:
        for markdown_text in test_cases:
            # Test that parsing doesn't crash and returns reasonable structure
            try:
                result = parser_obj.parse(markdown_text)
                # Basic sanity checks
                if hasattr(result, 'children') or hasattr(result, '__iter__'):
                    correct += 1
                else:
                    print(f"  Unexpected result type for: {repr(markdown_text[:30])}")
            except Exception as e:
                print(f"  Error parsing {repr(markdown_text[:30])}: {e}")

    except Exception as e:
        print(f"  Error in correctness test: {e}")
        import traceback
        traceback.print_exc()
        return 0.0

    print(f"  Correctness: {correct}/{total} = {correct/total:.3f}")
    return correct / total

def test_parser_performance(parser_obj, markdown_text: str) -> float:
    """Test parser performance with timing measurement."""
    try:
        # Warm up
        for _ in range(3):
            parser_obj.parse(markdown_text[:100])

        # Actual timing
        start_time = time.perf_counter()
        for _ in range(5):  # Multiple runs for accuracy
            result = parser_obj.parse(markdown_text)
        elapsed = (time.perf_counter() - start_time) / 5

        return elapsed

    except Exception as e:
        print(f"  Performance test error: {e}")
        return float('inf')

def evaluate_stage1(code: str) -> Dict[str, float]:
    """Stage 1: Quick validation with small dataset."""
    try:
        print(f"\n🔍 NESTED LOOPS EVALUATOR DEBUG:")
        print(f"  Type of 'code': {type(code)}")
        print(f"  Length: {len(code) if code else 'None'}")
        print(f"  First 200 chars: {repr(code[:200]) if code else 'None'}")

        # Handle both file paths AND code content
        if os.path.isfile(code.strip()):
            print("  📁 Detected file path - reading content")
            with open(code.strip(), 'r') as f:
                code_content = f.read()
            print(f"  📄 File content length: {len(code_content)}")
        else:
            print("  📝 Detected code content directly")
            code_content = code

        if not code_content.strip():
            print("  ⚠️  ERROR: Empty code content!")
            return {"correctness": 0.0, "performance": 0.0, "combined_score": 0.0, "error": "Empty code content"}

        # Check if this contains the target optimization block
        if "performance-nested-loops-parse-source" in code_content:
            print(f"  ✅ Found target optimization block in code")

            # Extract the evolved method
            evolved_method = extract_parse_source_method(code_content)

            if evolved_method:
                # Create evolved parser
                parser_obj = create_evolved_parser(evolved_method)
            else:
                print(f"  ⚠️  Could not extract evolved method, using original parser")
                import marko
                parser_obj = marko.Markdown()

        else:
            print(f"  ⚠️  Target optimization block not found, using original parser")
            import marko
            parser_obj = marko.Markdown()

        # Test correctness
        correctness = test_parser_correctness(parser_obj)

        if correctness < 0.8:  # Correctness gate
            return {"correctness": correctness, "performance": 0.0, "combined_score": 0.0}

        # Test performance
        test_markdown = generate_test_markdown("small")
        elapsed_time = test_parser_performance(parser_obj, test_markdown)
        print(f"  ⏱️  Performance: elapsed={elapsed_time:.6f}s")

        target_time = get_performance_target("small")
        performance = sigmoid_performance_score(elapsed_time, target_time)
        print(f"  📊 Scores: target={target_time:.6f}s, performance={performance:.3f}")

        combined_score = correctness * performance
        print(f"  🎯 Final: correctness={correctness:.3f} * performance={performance:.3f} = {combined_score:.3f}")

        return {
            "correctness": correctness,
            "performance": performance,
            "combined_score": combined_score,
            "elapsed_time": elapsed_time,
            "target_time": target_time
        }

    except Exception as e:
        print(f"🚨 STAGE 1 EVALUATION ERROR: {e}")
        import traceback
        print("🚨 Full traceback:")
        traceback.print_exc()
        return {"correctness": 0.0, "performance": 0.0, "combined_score": 0.0, "error": str(e)}

def evaluate_stage2(code: str) -> Dict[str, float]:
    """Stage 2: Comprehensive testing with multiple scales."""
    try:
        # Handle file paths
        if os.path.isfile(code.strip()):
            with open(code.strip(), 'r') as f:
                code_content = f.read()
        else:
            code_content = code

        # Create parser with evolved code
        if "performance-nested-loops-parse-source" in code_content:
            evolved_method = extract_parse_source_method(code_content)
            if evolved_method:
                parser_obj = create_evolved_parser(evolved_method)
            else:
                import marko
                parser_obj = marko.Markdown()
        else:
            import marko
            parser_obj = marko.Markdown()

        # Test correctness
        correctness = test_parser_correctness(parser_obj)

        if correctness < 0.8:
            return {"correctness": correctness, "performance": 0.0, "combined_score": 0.0}

        # Test performance across multiple scales
        test_sizes = ["small", "medium", "large"]
        weights = [0.2, 0.3, 0.5]  # Weight larger tests more heavily

        total_performance = 0.0
        all_times = []

        for size, weight in zip(test_sizes, weights):
            test_markdown = generate_test_markdown(size)
            elapsed_time = test_parser_performance(parser_obj, test_markdown)

            target_time = get_performance_target(size)
            performance = sigmoid_performance_score(elapsed_time, target_time)

            total_performance += performance * weight
            all_times.append(elapsed_time)

        combined_score = correctness * total_performance

        return {
            "correctness": correctness,
            "performance": total_performance,
            "combined_score": combined_score,
            "times": all_times,
            "avg_time": sum(all_times) / len(all_times)
        }

    except Exception as e:
        return {"correctness": 0.0, "performance": 0.0, "combined_score": 0.0, "error": str(e)}

def evaluate(code: str) -> Dict[str, float]:
    """
    Main OpenEvolve evaluation entry point.
    MUST return dict with: correctness, performance, combined_score
    """
    # Stage 1 filter
    stage1_result = evaluate_stage1(code)

    if stage1_result["correctness"] < 0.5:  # Quick reject for very broken code
        return {
            "correctness": stage1_result["correctness"],
            "performance": stage1_result["performance"],
            "combined_score": stage1_result["combined_score"]
        }

    # Stage 2 comprehensive evaluation
    stage2_result = evaluate_stage2(code)

    return {
        "correctness": stage2_result["correctness"],
        "performance": stage2_result["performance"],
        "combined_score": stage2_result["combined_score"]
    }

if __name__ == "__main__":
    # Test the evaluator with both original and optimized parsers
    print("Testing nested loops evaluator with original marko parser...")

    original_parser_path = "marko/parser.py"

    print("=== ORIGINAL PARSER TEST ===")
    result_original = evaluate(original_parser_path)
    print(f"Original parser evaluation:")
    print(f"  Correctness: {result_original['correctness']:.3f}")
    print(f"  Performance: {result_original['performance']:.3f}")
    print(f"  Combined Score: {result_original['combined_score']:.3f}")

    print("\n=== OPTIMIZED PARSER TEST ===")
    optimized_parser_path = "test_optimized_parser.py"
    result_optimized = evaluate(optimized_parser_path)
    print(f"Optimized parser evaluation:")
    print(f"  Correctness: {result_optimized['correctness']:.3f}")
    print(f"  Performance: {result_optimized['performance']:.3f}")
    print(f"  Combined Score: {result_optimized['combined_score']:.3f}")

    print(f"\n=== COMPARISON ===")
    improvement = result_optimized['combined_score'] - result_original['combined_score']
    print(f"Score improvement: {improvement:.3f} ({improvement/result_original['combined_score']*100:.1f}%)")

    if improvement > 0:
        print("✅ Evaluator can detect improvements!")
    else:
        print("⚠️  Evaluator may need adjustment")

    print(f"\n🎯 Target: Achieve >0.8 combined score through nested loop optimization")