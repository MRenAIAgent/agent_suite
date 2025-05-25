# Benchmark Framework

This directory contains benchmarking frameworks for evaluating different components of the agent suite.

## Available Benchmarks

- [RAG Benchmark](./rag/README.md) - Benchmarks for Retrieval Augmented Generation
  - Evaluates retrieval accuracy, answer quality, and factual correctness
  - Supports multiple datasets (NQ, MS MARCO, HotpotQA)
  - Measures relevance, contextual quality, and hallucination rates

- [Agent Benchmark](./agent/README.md) - Benchmarks for Agent implementations
  - Compares different agent architectures (ReAct, LangChain)
  - Evaluates performance on various task types (reasoning, planning, tool use)
  - Measures accuracy, execution time, and other metrics

## Running Benchmarks

You can run all benchmarks with the provided script:

```bash
# Run all benchmarks in dry-run mode (doesn't make API calls)
python benchmark/run_all_benchmarks.py --dry-run

# Run specific benchmark type
python benchmark/run_all_benchmarks.py --include rag

# Run with verbose output
python benchmark/run_all_benchmarks.py --verbose
```

Each benchmark can also be run individually. See the README files in each subdirectory for more details.

## Directory Structure

- `rag/` - RAG benchmark implementation
  - `code/` - Benchmark code
  - `data/` - Benchmark datasets
  - `results/` - Benchmark results

- `agent/` - Agent benchmark implementation
  - `code/` - Benchmark code
  - `data/` - Benchmark datasets
  - `results/` - Benchmark results

## Adding New Benchmarks

To add a new benchmark:

1. Create a new directory with the benchmark name
2. Add subdirectories for `code`, `data`, and `results`
3. Create an `__init__.py` file in each directory
4. Implement the benchmark code in the `code/` directory
5. Update this README to include the new benchmark

## Legacy Scripts

Older benchmark scripts have been moved to the `scripts/` directory. These scripts are maintained for reference but new code should use the more structured benchmark implementations in this directory.

## Future Benchmarks

Additional benchmark frameworks will be added here to evaluate:

- Agent performance
- Tool usage efficiency
- Multi-modal capabilities
- Memory systems
- Knowledge graph operations 