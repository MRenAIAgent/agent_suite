# Agent Benchmark Data

This directory contains datasets used for agent benchmarking.

## TauBench Datasets

TauBench datasets include various task types for evaluating agent performance:

- Math tasks
- Planning tasks
- Tool use tasks
- Knowledge tasks
- Reasoning tasks

## Directory Structure

- `taubench/` - TauBench evaluation datasets
  - `datasets/` - Contains the TauBench datasets

## Adding New Datasets

To add new benchmark datasets:

1. Create a new subdirectory under the appropriate benchmark type
2. Add dataset files
3. Update the benchmark code to use the new datasets
4. Document the dataset structure and format 