# Agent Benchmark Framework

This directory contains benchmarking tools for evaluating different agent implementations.

## Available Benchmarks

- [TauBench](./code/README.md) - Benchmarks agent performance using TauBench framework
  - Compares different agent architectures (ReAct, LangChain)
  - Evaluates performance on various task types
  - Measures accuracy, execution time, and other metrics

## Running Benchmarks

Each benchmark framework has its own documentation and execution instructions. Please refer to the specific README files in each subdirectory for more information.

## Directory Structure

- `code/` - Contains the benchmark implementation code
- `data/` - Contains benchmark datasets
- `results/` - Contains benchmark results and visualizations

## Adding New Benchmarks

To add a new benchmark:

1. Create a new directory under `benchmark/agent/code/`
2. Implement the benchmark framework
3. Update this README to document the new benchmark 