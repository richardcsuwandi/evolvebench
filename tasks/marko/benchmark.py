"""
Simple performance test for OpenEvolve optimizations
This test compares the benchmark performance to verify improvements.
"""

import time
import sys
import os
import types
import importlib
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))


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

def load_source_class_from_file(file_path):
    """Load Source class from a file, handling both regular files and EVOLVE-BLOCK format"""
    # Extract code (handles EVOLVE-BLOCK format if present)
    code = extract_code_from_evolve_block(file_path)
    
    # Create namespace with necessary imports
    future_annotations_code = "from __future__ import annotations\n"
    
    namespace = {
        '__builtins__': __builtins__,
    }
    
    # Add common imports that might be needed
    try:
        from typing import TYPE_CHECKING, Generator, Match, Pattern, cast, overload
        namespace.update({
            'TYPE_CHECKING': TYPE_CHECKING,
            'Generator': Generator,
            'Match': Match,
            'Pattern': Pattern,
            'cast': cast,
            'overload': overload,
        })
    except ImportError:
        pass
    
    try:
        import functools
        namespace['functools'] = functools
    except ImportError:
        pass
    
    try:
        import re
        namespace['re'] = re
    except ImportError:
        pass
    
    try:
        import types as types_module
        namespace['types'] = types_module
    except ImportError:
        pass
    
    try:
        from contextlib import contextmanager
        namespace['contextmanager'] = contextmanager
    except ImportError:
        pass
    
    # Import marko.block for the Source class dependencies
    try:
        from marko.block import BlockElement, Document
        namespace['BlockElement'] = BlockElement
        namespace['Document'] = Document
    except ImportError:
        pass
    
    # Prepend future annotations if not already present
    if 'from __future__ import annotations' not in code:
        code_to_exec = future_annotations_code + code
    else:
        code_to_exec = code
    
    exec(code_to_exec, namespace)
    
    # Return the Source class
    if 'Source' in namespace:
        return namespace['Source']
    else:
        raise ValueError(f"Source class not found in {file_path}")

def replace_source_class(new_source_class):
    """Replace the Source class in the marko.source module"""
    import marko.source
    marko.source.Source = new_source_class
    # Reload the module to ensure changes take effect
    importlib.reload(marko.source)
    # Also reload modules that import Source
    if 'marko.parser' in sys.modules:
        importlib.reload(sys.modules['marko.parser'])
    if 'marko' in sys.modules:
        importlib.reload(sys.modules['marko'])

def create_test_content():
    """Create test markdown content that exercises optimized hotspots."""
    base_content = """
# Performance Test Document
This document exercises the empirically-identified hotspots:
## Code Blocks with Tabs (tests tab expansion optimization)
```python
def example_function():
	# Tab characters trigger optimization
	for i in range(100):
		print(f"Line {i}")
		if i % 2 == 0:
			continue
		else:
			break
```
## Complex Lists (triggers element matching and regex operations)
1. First item with **bold** and *italic* text
   - Nested item A with `inline code`
   - Nested item B with [links](http://example.com)
     - Deep nested item with more **formatting**
2. Second item with multiple patterns
   - **bold *nested italic* text**
   - `code with **bold** inside`
   - [complex link](http://example.com)
## Blockquotes (triggers prefix matching)
> This is a blockquote that requires prefix matching
> with multiple lines and extensive regex operations
>
> > Nested blockquotes
> > with more content that needs processing
>
> Back to normal level
## Tables and Complex Formatting
| Column 1 | Column 2 | Column 3 |
|----------|----------|----------|
| Data 1   | Data 2   | Data 3   |
| More     | Data     | Here     |
### More complex patterns
- [ ] Checkbox items with **formatting**
- [x] Completed items with `code`
- [ ] Links in checkboxes [example](http://test.com)
**Bold text with *nested italic* and `inline code`**
---
Horizontal rules and more content.
"""
    # Multiply for larger test
    return base_content * 20

def run_performance_test(version_name="current"):
    """Run performance test on current Marko implementation."""
    import marko
    
    test_content = create_test_content()
    iterations = 50

    # Warm up
    for _ in range(3):
        marko.convert(test_content)

    # Actual test
    start_time = time.perf_counter()

    for i in range(iterations):
        result = marko.convert(test_content)
        if i % 10 == 0 and version_name == "current":
            print(f"   Completed {i}/{iterations} iterations")

    end_time = time.perf_counter()

    total_time = end_time - start_time
    avg_time = total_time / iterations

    return {
        'total_time': total_time,
        'avg_time': avg_time,
        'iterations': iterations,
        'document_size': len(test_content)
    }

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
        
        print("=" * 80)
        print("DETAILED MARKO PERFORMANCE BENCHMARK - SOURCE CLASS COMPARISON")
        print("=" * 80)

        # Get the directory where this script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Load programs
        baseline_path = os.path.join(script_dir, 'marko', 'source.py')
        handwritten_path = os.path.join(script_dir, 'best_program_handwritten.py')
        llm_generated_path = os.path.join(script_dir, 'best_program_llm_generated.py')

        print(f"\n📂 Loading programs:")
        print(f"   Baseline (initial): {baseline_path}")
        print(f"   Handwritten (best): {handwritten_path}")
        print(f"   LLM Generated (best): {llm_generated_path}")

        # Store original Source class
        import marko.source
        original_source = marko.source.Source

        test_content = create_test_content()
        iterations = 50

        print(f"\n📊 Test parameters:")
        print(f"   Document size: {len(test_content):,} characters")
        print(f"   Iterations: {iterations}")
        print(f"   Lines: {test_content.count(chr(10)):,}")

        print("\n" + "=" * 80)
        print("PERFORMANCE RESULTS")
        print("=" * 80)

        results = {}

        # Test baseline
        print(f"\n🔬 Testing baseline (initial)...")
        try:
            baseline_source = load_source_class_from_file(baseline_path)
            replace_source_class(baseline_source)
            baseline_results = run_performance_test("baseline")
            results['baseline'] = baseline_results
            print(f"   ✅ Baseline completed")
        except Exception as e:
            print(f"   ❌ Baseline failed: {e}")
            import traceback
            traceback.print_exc()
            results['baseline'] = None

        # Test handwritten
        print(f"\n🔬 Testing handwritten (best)...")
        try:
            handwritten_source = load_source_class_from_file(handwritten_path)
            replace_source_class(handwritten_source)
            handwritten_results = run_performance_test("handwritten")
            results['handwritten'] = handwritten_results
            print(f"   ✅ Handwritten completed")
        except Exception as e:
            print(f"   ❌ Handwritten failed: {e}")
            import traceback
            traceback.print_exc()
            results['handwritten'] = None

        # Test llm_generated
        print(f"\n🔬 Testing llm_generated (best)...")
        try:
            llm_generated_source = load_source_class_from_file(llm_generated_path)
            replace_source_class(llm_generated_source)
            llm_generated_results = run_performance_test("llm_generated")
            results['llm_generated'] = llm_generated_results
            print(f"   ✅ LLM Generated completed")
        except Exception as e:
            print(f"   ❌ LLM Generated failed: {e}")
            import traceback
            traceback.print_exc()
            results['llm_generated'] = None

        # Restore original Source class
        replace_source_class(original_source)

        # Print results
        print("\n" + "=" * 80)
        print("RESULTS SUMMARY")
        print("=" * 80)

        if results['baseline']:
            baseline_stats = results['baseline']
            print(f"\n   Baseline:")
            print(f"      Total time: {baseline_stats['total_time']:.4f}s")
            print(f"      Average per iteration: {baseline_stats['avg_time']:.6f}s")
            print(f"      Throughput: {baseline_stats['document_size'] * baseline_stats['iterations'] / baseline_stats['total_time'] / 1024:.1f} KB/s")

        if results['handwritten']:
            handwritten_stats = results['handwritten']
            print(f"\n   Handwritten:")
            print(f"      Total time: {handwritten_stats['total_time']:.4f}s")
            print(f"      Average per iteration: {handwritten_stats['avg_time']:.6f}s")
            print(f"      Throughput: {handwritten_stats['document_size'] * handwritten_stats['iterations'] / handwritten_stats['total_time'] / 1024:.1f} KB/s")

        if results['llm_generated']:
            llm_generated_stats = results['llm_generated']
            print(f"\n   LLM Generated:")
            print(f"      Total time: {llm_generated_stats['total_time']:.4f}s")
            print(f"      Average per iteration: {llm_generated_stats['avg_time']:.6f}s")
            print(f"      Throughput: {llm_generated_stats['document_size'] * llm_generated_stats['iterations'] / llm_generated_stats['total_time'] / 1024:.1f} KB/s")

        # Calculate speedups
        if results['baseline'] and results['handwritten']:
            baseline_time = results['baseline']['avg_time']
            handwritten_time = results['handwritten']['avg_time']
            handwritten_speedup = (baseline_time - handwritten_time) / baseline_time * 100
            handwritten_factor = baseline_time / handwritten_time if handwritten_time > 0 else 0
            
            print(f"\n   🚀 Speedup vs Baseline:")
            print(f"      Handwritten: {handwritten_speedup:+.2f}% ({handwritten_factor:.2f}x)")

        if results['baseline'] and results['llm_generated']:
            baseline_time = results['baseline']['avg_time']
            llm_generated_time = results['llm_generated']['avg_time']
            llm_generated_speedup = (baseline_time - llm_generated_time) / baseline_time * 100
            llm_generated_factor = baseline_time / llm_generated_time if llm_generated_time > 0 else 0
            
            if results['baseline'] and results['handwritten']:
                print(f"      LLM Generated: {llm_generated_speedup:+.2f}% ({llm_generated_factor:.2f}x)")
            else:
                print(f"\n   🚀 Speedup vs Baseline:")
                print(f"      LLM Generated: {llm_generated_speedup:+.2f}% ({llm_generated_factor:.2f}x)")

        print("\n" + "=" * 80)
    finally:
        # Restore original stdout and stderr
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        tee_stdout.close()
        tee_stderr.close()