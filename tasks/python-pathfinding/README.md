# Python Pathfinding Multi-File Evolution Example

This example demonstrates multi-file evolution using OpenEvolve to optimize heap operations in the [python-pathfinding](https://github.com/brean/python-pathfinding) library.

## Overview

The python-pathfinding library implements various pathfinding algorithms including A*. This example focuses on optimizing the heap operations used in the A* algorithm across multiple files to improve pathfinding performance.

## EVOLVE-BLOCK Locations

The following files contain `EVOLVE-BLOCK` markers with `id="heap-optimization"`:

1. **`pathfinding/core/heap.py`** - 3 blocks:
   - `pop_node()` method (lines 46-60)
   - `push_node()` method (lines 68-76) 
   - `remove_node()` method (lines 88-93)

2. **`pathfinding/finder/finder.py`** - 1 block:
   - `process_node()` method (lines 140-150)

3. **`pathfinding/finder/a_star.py`** - 1 block:
   - `check_neighbors()` method (lines 55-83)

## Setup

### 1. Clone the Python Pathfinding Repository

```bash
# Clone the repository with EVOLVE-BLOCK markers already added
git clone https://github.com/brean/python-pathfinding.git
cd python-pathfinding

# Or if you have a local copy, ensure it has the EVOLVE-BLOCK markers
```

### 2. Install Dependencies

```bash
# Install the pathfinding library in development mode
pip install -e .

# Install OpenEvolve dependencies
pip install openevolve
```

## Usage

### List Available EVOLVE-BLOCKs

```bash
python openevolve-run.py \
  --directory /path/to/python-pathfinding \
  --list-blocks
```

Expected output:
```
Found 1 unique block IDs:

  ID: 'heap-optimization'
    Files: 5
      - pathfinding/core/heap.py (lines 46-60)
      - pathfinding/core/heap.py (lines 68-76)
      - pathfinding/core/heap.py (lines 88-93)
      - pathfinding/finder/finder.py (lines 140-150)
      - pathfinding/finder/a_star.py (lines 55-83)
```

### Run Multi-File Evolution

**Important**: Use the OpenEvolve virtual environment to avoid dependency issues:

```bash
# Activate OpenEvolve environment
cd /path/to/openevolve
source .venv/bin/activate

# Run evolution from the examples directory
cd examples/python_pathfinding
openevolve-run --directory /path/to/python-pathfinding --block-id heap-optimization "" evaluator.py --config config.yaml
```

Alternative with explicit paths:
```bash
# From any directory with OpenEvolve venv activated
openevolve-run \
  --directory /path/to/python-pathfinding \
  --block-id heap-optimization \
  "" \
  /path/to/openevolve/examples/python_pathfinding/evaluator.py \
  --config /path/to/openevolve/examples/python_pathfinding/config.yaml
```

### Parameters

- `--directory`: Path to the python-pathfinding repository
- `--block-id`: The ID of EVOLVE-BLOCKs to evolve (`heap-optimization`)
- `""`: Empty placeholder for initial_program (not used in multi-file mode)
- `evaluator.py`: Path to the evaluator that tests pathfinding performance
- `--config`: Configuration file with LLM settings, evolution parameters, and early stopping

## Evaluation Metrics

The evaluator tests pathfinding performance using:

### Correctness
- Percentage of successful pathfinding attempts
- Tests various grid sizes and obstacle densities
- Ensures evolved code doesn't break functionality

### Performance  
- Average time per pathfinding operation
- Normalized score where faster = better
- Baseline: ~0.002s per path, Target: <0.001s per path

## Expected Improvements

Potential optimizations the LLM might discover:

1. **Heap Operations**:
   - More efficient node tuple creation
   - Optimized removal tracking
   - Better heap ordering strategies

2. **Node Processing**:
   - Streamlined open list management
   - Reduced redundant calculations
   - Improved memory access patterns

3. **Algorithm Flow**:
   - Enhanced neighbor checking
   - Optimized backtrace handling
   - Better early termination conditions

## Output Structure

After evolution completes:

```
pathfinding_evolved/
├── pathfinding/
│   ├── core/
│   │   └── heap.py          # Optimized heap operations
│   └── finder/
│       ├── finder.py        # Optimized node processing  
│       └── a_star.py        # Optimized neighbor checking
├── logs/                    # Evolution logs
├── checkpoints/             # Evolution checkpoints
└── best/                    # Best program info
```

## Configuration

The `config.yaml` uses proven settings optimized for multi-file evolution:

- **diff_based_evolution**: Required for multi-file mode
- **Temperature 0.4**: Optimal balance of creativity vs consistency  
- **Context 16k tokens**: Sufficient for complete files
- **Artifacts enabled**: +20% improvement in results
- **4 islands**: Parallel evolution with diversity
- **Early stopping**: Automatically stops when no improvement for 20 iterations

## Troubleshooting

### Common Issues

1. **"ModuleNotFoundError: No module named 'yaml'"**:
   - **Solution**: Use the OpenEvolve virtual environment instead of python-pathfinding's venv
   - The OpenEvolve venv has all required dependencies (PyYAML, etc.)
   - Multiprocessing workers inherit the Python environment and need OpenEvolve's dependencies

2. **"Block ID not found"**:
   - Ensure the repository has EVOLVE-BLOCK markers with `id="heap-optimization"`
   - Use `--list-blocks` to verify available IDs

3. **"Import errors"**:
   - Ensure python-pathfinding is installed: `pip install -e .`
   - Check that the repository path is correct

4. **"super() errors in evaluator"**:
   - This is expected during module reloading
   - The evaluator handles these gracefully and continues

5. **"LLM connection errors"**:
   - Verify your LLM API endpoint in `config.yaml`
   - Ensure API keys are properly configured

### Performance Tips

1. **Start Small**: Use 5-10 iterations first to verify everything works
2. **Monitor Logs**: Check `logs/` directory for detailed evolution progress  
3. **Use Checkpoints**: Resume long runs with `--checkpoint` if interrupted
4. **Multiple Runs**: Run several times with different seeds for best results

## Advanced Usage

### Custom Evaluator

Modify `evaluator.py` to test different aspects:

```python
# Test specific grid sizes
sizes = [10, 20, 50, 100]

# Test different obstacle ratios  
obstacle_ratios = [0.1, 0.3, 0.5]

# Add memory usage testing
# Add specific pathfinding scenarios
```

### Configuration Tuning

Adjust `config.yaml` for your needs:

```yaml
# For faster iteration during development
max_iterations: 10
checkpoint_interval: 2

# For more exploration
database:
  exploration_ratio: 0.4  # Default: 0.3
  
# For different LLM models  
llm:
  models:
    - name: "gpt-4"
      weight: 1.0
```

## Results Analysis

After evolution, analyze results:

1. **Performance Graphs**: Plot metrics over iterations
2. **Code Diff**: Compare original vs evolved files
3. **Benchmark Testing**: Run comprehensive performance tests
4. **Code Review**: Examine discovered optimizations

The multi-file approach enables coordinated optimizations across the entire pathfinding pipeline, potentially discovering improvements that single-file evolution would miss.
## License

This task is based on code from python-pathfinding, which is licensed under the MIT
License. The upstream license text is included at `LICENSE` in this task directory.
EvolveBench's own harness and task code are licensed separately — see the repository
root [LICENSE](../../LICENSE).
