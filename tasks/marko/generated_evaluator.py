import math
import sys
import time
import importlib.util
import traceback
import tempfile
import os
from typing import Any, Dict


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


def load_module_from_code(code: str):
    """Load a module from code string, handling __future__ imports properly."""
    # Create a temporary file with the code
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as tmp:
        tmp.write(code)
        tmp_path = tmp.name
    
    try:
        spec = importlib.util.spec_from_file_location("program_module", tmp_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load module from {tmp_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules["program_module"] = module
        spec.loader.exec_module(module)
        return module
    finally:
        try:
            os.unlink(tmp_path)
        except:
            pass


def extract_source_code(program_path: str) -> str:
    """Extract the actual Source class code from the program file."""
    with open(program_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find the marko/source.py section
    lines = content.split('\n')
    source_lines = []
    in_source_file = False
    
    for line in lines:
        if '### FILE: marko/source.py ###' in line:
            in_source_file = True
            continue
        elif '### END FILE ###' in line or '### FILE:' in line:
            if in_source_file:
                break
        elif in_source_file:
            source_lines.append(line)
    
    return '\n'.join(source_lines)


def create_mock_block_element():
    """Create a mock BlockElement for testing."""
    class MockBlockElement:
        def __init__(self):
            self._prefix = ""
            self._second_prefix = "  "
    return MockBlockElement()


def test_source_basic_operations(Source):
    """Test basic Source operations."""
    text = "Hello\nWorld\n"
    source = Source(text)
    
    # Test initialization
    assert source.pos == 0
    assert source._buffer == "Hello\nWorld\n"
    assert source.exhausted == False
    
    # Test next_line without prefix requirement
    line = source.next_line(require_prefix=False)
    assert line == "Hello"
    
    # Test consume
    source.consume()
    assert source.pos > 0
    
    # Test anchor and reset
    source.anchor()
    old_pos = source.pos
    source.next_line(require_prefix=False)
    source.consume()
    source.reset()
    assert source.pos == old_pos
    
    return True


def test_source_with_states(Source):
    """Test Source with state management."""
    text = "Line 1\nLine 2\n"
    source = Source(text)
    
    # Create mock states
    state1 = create_mock_block_element()
    state2 = create_mock_block_element()
    
    # Test push/pop state
    source.push_state(state1)
    assert len(source._states) == 1
    
    source.push_state(state2)
    assert len(source._states) == 2
    
    popped = source.pop_state()
    assert popped == state2
    assert len(source._states) == 1
    
    return True


def test_source_regex_matching(Source):
    """Test regex matching functionality."""
    text = "# Header\nParagraph\n"
    source = Source(text)
    
    # Test expect_re
    match = source.expect_re(r"#\s+\w+")
    if match:
        assert match.group() == "# Header"
    
    return True


def test_source_prefix_matching(Source):
    """Test prefix matching functionality."""
    # Test match_prefix static method
    result = Source.match_prefix("", "Hello")
    assert result == 0
    
    result = Source.match_prefix("  ", "  Hello")
    assert result >= 0
    
    result = Source.match_prefix(">>>", "Hello")
    assert result == -1
    
    return True


def test_source_preprocessing(Source):
    """Test text preprocessing."""
    text = "Line 1\r\nLine 2\r\n"
    source = Source(text)
    
    # Check that \r\n is converted to \n
    assert "\r\n" not in source._buffer
    assert "Line 1\nLine 2\n" == source._buffer
    
    return True


def test_source_exhausted(Source):
    """Test exhausted property."""
    text = "A"
    source = Source(text)
    
    assert not source.exhausted
    
    source.pos = len(source._buffer)
    assert source.exhausted
    
    return True


def test_source_complex_parsing(Source):
    """Test complex parsing scenario."""
    text = "# Title\n\nParagraph 1\n\nParagraph 2\n"
    source = Source(text)
    
    lines = []
    while not source.exhausted:
        line = source.next_line(require_prefix=False)
        if line:
            lines.append(line)
            source.consume()
        else:
            break
    
    assert len(lines) > 0
    return True


def evaluate_stage1(program_path: str) -> dict:
    """Quick validation with 5 diverse test cases."""
    try:
        source_code = extract_source_code(program_path)
        module = load_module_from_code(source_code)
        Source = module.Source
        
        passed = 0
        total = 5
        
        # Test 1: Basic operations
        try:
            if test_source_basic_operations(Source):
                passed += 1
        except Exception:
            pass
        
        # Test 2: State management
        try:
            if test_source_with_states(Source):
                passed += 1
        except Exception:
            pass
        
        # Test 3: Regex matching
        try:
            if test_source_regex_matching(Source):
                passed += 1
        except Exception:
            pass
        
        # Test 4: Preprocessing
        try:
            if test_source_preprocessing(Source):
                passed += 1
        except Exception:
            pass
        
        # Test 5: Exhausted property
        try:
            if test_source_exhausted(Source):
                passed += 1
        except Exception:
            pass
        
        correctness = passed / total
        
        # Performance test - simple parsing
        text = "# Header\n\n" + "\n".join([f"Line {i}" for i in range(100)])
        start = time.perf_counter()
        for _ in range(100):
            source = Source(text)
            while not source.exhausted:
                line = source.next_line(require_prefix=False)
                if line:
                    source.consume()
                else:
                    break
        elapsed = time.perf_counter() - start
        
        # Baseline target: 0.1 seconds for this test
        target_time = 0.1
        performance = sigmoid_performance_score(elapsed, target_time, steepness=2.0)
        
        combined_score = 0.5 * correctness + 0.5 * performance
        
        return {
            "correctness": correctness,
            "performance": performance,
            "combined_score": combined_score,
            "stage": 1.0
        }
        
    except Exception as e:
        return {
            "correctness": 0.0,
            "performance": 0.0,
            "combined_score": 0.0,
            "stage": 1.0,
            "error": str(e),
            "traceback": traceback.format_exc()
        }


def evaluate_stage2(program_path: str) -> dict:
    """Comprehensive testing with 10+ test cases including edge cases."""
    try:
        source_code = extract_source_code(program_path)
        module = load_module_from_code(source_code)
        Source = module.Source
        
        passed = 0
        total = 12
        
        # Test 1: Basic operations
        try:
            if test_source_basic_operations(Source):
                passed += 1
        except Exception:
            pass
        
        # Test 2: State management
        try:
            if test_source_with_states(Source):
                passed += 1
        except Exception:
            pass
        
        # Test 3: Regex matching
        try:
            if test_source_regex_matching(Source):
                passed += 1
        except Exception:
            pass
        
        # Test 4: Prefix matching
        try:
            if test_source_prefix_matching(Source):
                passed += 1
        except Exception:
            pass
        
        # Test 5: Preprocessing
        try:
            if test_source_preprocessing(Source):
                passed += 1
        except Exception:
            pass
        
        # Test 6: Exhausted property
        try:
            if test_source_exhausted(Source):
                passed += 1
        except Exception:
            pass
        
        # Test 7: Complex parsing
        try:
            if test_source_complex_parsing(Source):
                passed += 1
        except Exception:
            pass
        
        # Test 8: Empty text
        try:
            source = Source("")
            assert source.exhausted
            passed += 1
        except Exception:
            pass
        
        # Test 9: Single line
        try:
            source = Source("Single line")
            line = source.next_line(require_prefix=False)
            assert line == "Single line"
            passed += 1
        except Exception:
            pass
        
        # Test 10: Multiple newlines
        try:
            source = Source("\n\n\n")
            count = 0
            while not source.exhausted:
                line = source.next_line(require_prefix=False)
                if line is not None:
                    source.consume()
                    count += 1
                else:
                    break
            assert count > 0
            passed += 1
        except Exception:
            pass
        
        # Test 11: Under state context manager
        try:
            source = Source("Test")
            state = create_mock_block_element()
            with source.under_state(state):
                assert len(source._states) == 1
            assert len(source._states) == 0
            passed += 1
        except Exception:
            pass
        
        # Test 12: Anchor/reset multiple times
        try:
            source = Source("Line 1\nLine 2\nLine 3\n")
            source.anchor()
            source.next_line(require_prefix=False)
            source.consume()
            pos1 = source.pos
            source.reset()
            assert source.pos < pos1
            passed += 1
        except Exception:
            pass
        
        correctness = passed / total
        
        # Performance tests
        perf_scores = []
        
        # Performance test 1: Small document parsing
        text_small = "# Header\n\n" + "\n".join([f"Line {i}" for i in range(100)])
        start = time.perf_counter()
        for _ in range(100):
            source = Source(text_small)
            while not source.exhausted:
                line = source.next_line(require_prefix=False)
                if line:
                    source.consume()
                else:
                    break
        elapsed_small = time.perf_counter() - start
        perf_scores.append(sigmoid_performance_score(elapsed_small, 0.1, steepness=2.0))
        
        # Performance test 2: Medium document parsing
        text_medium = "\n".join([f"Paragraph {i}\n" for i in range(500)])
        start = time.perf_counter()
        for _ in range(50):
            source = Source(text_medium)
            while not source.exhausted:
                line = source.next_line(require_prefix=False)
                if line:
                    source.consume()
                else:
                    break
        elapsed_medium = time.perf_counter() - start
        perf_scores.append(sigmoid_performance_score(elapsed_medium, 0.2, steepness=2.0))
        
        # Performance test 3: Regex matching performance
        text_regex = "\n".join([f"# Header {i}" for i in range(200)])
        start = time.perf_counter()
        for _ in range(50):
            source = Source(text_regex)
            while not source.exhausted:
                match = source.expect_re(r"#\s+\w+\s+\d+")
                if match:
                    source.consume()
                else:
                    line = source.next_line(require_prefix=False)
                    if line:
                        source.consume()
                    else:
                        break
        elapsed_regex = time.perf_counter() - start
        perf_scores.append(sigmoid_performance_score(elapsed_regex, 0.15, steepness=2.0))
        
        # Performance test 4: State management overhead
        text_state = "\n".join([f"Line {i}" for i in range(100)])
        start = time.perf_counter()
        for _ in range(100):
            source = Source(text_state)
            state = create_mock_block_element()
            source.push_state(state)
            while not source.exhausted:
                line = source.next_line(require_prefix=False)
                if line:
                    source.consume()
                else:
                    break
            source.pop_state()
        elapsed_state = time.perf_counter() - start
        perf_scores.append(sigmoid_performance_score(elapsed_state, 0.12, steepness=2.0))
        
        performance = sum(perf_scores) / len(perf_scores)
        combined_score = 0.5 * correctness + 0.5 * performance
        
        return {
            "correctness": correctness,
            "performance": performance,
            "combined_score": combined_score,
            "stage": 2.0
        }
        
    except Exception as e:
        return {
            "correctness": 0.0,
            "performance": 0.0,
            "combined_score": 0.0,
            "stage": 2.0,
            "error": str(e),
            "traceback": traceback.format_exc()
        }


def evaluate(program_path: str) -> dict:
    """Main evaluation function."""
    return evaluate_stage2(program_path)
