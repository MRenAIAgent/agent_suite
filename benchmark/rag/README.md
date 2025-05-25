# Benchmark Framework

This directory contains the benchmarking framework for evaluating various aspects of the RAG (Retrieval Augmented Generation) and knowledge retrieval systems.

## Directory Structure

- `code/`: Contains all the benchmark code
  - `benchmarks/`: Individual benchmark implementations
  - `__init__.py`: Package initialization
  - `config.py`: Configuration classes and utilities
  - `data_generators.py`: Synthetic data generation
  - `main.py`: Main entry point for running benchmarks
  - `metrics.py`: Metrics collection and analysis
  - `mock_vector_benchmarks.py`: Mock implementations for testing
  - `rag_top_benchmark.py`: Standalone RAG benchmark
  - `runners.py`: Benchmark execution runners
  - `utils.py`: Utility functions
  - `visualization.py`: Result visualization utilities

- `data/`: Contains benchmark datasets and generated data
  - `datasets/`: Test datasets
    - `hotpotqa_sample.jsonl`: Sample from HotpotQA dataset
    - `msmarco_sample.jsonl`: Sample from MS MARCO dataset
    - `nq_sample.jsonl`: Sample from Natural Questions dataset
  - `graph/`: Storage for graph data during benchmarks
  - `vector/`: Storage for vector data during benchmarks

- `results/`: Output directory for benchmark results

## Running Benchmarks

### Basic Usage

```bash
python -m benchmark.code.main
```

### Running Specific Benchmarks

```bash
python -m benchmark.code.main --benchmarks graph vector
```

### Generating Test Datasets

```bash
python -m benchmark.code.main --generate-dataset
```

### Using Custom Configuration

```bash
python -m benchmark.code.main --config path/to/config.json
```

### Running the Standalone RAG Benchmark

```bash
python -m benchmark.code.rag_top_benchmark --datasets nq msmarco hotpotqa
```

## Adding New Benchmarks

To add a new benchmark:

1. Create a new module in the `benchmark/code/benchmarks/` directory
2. Define benchmark functions that follow the required interface
3. Register the benchmarks in the main runner

## Configuration

The benchmark framework can be configured through:

- Command-line arguments
- Configuration files
- Environment variables (for sensitive settings like API keys)

See `benchmark/code/config.py` for available configuration options. 