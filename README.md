# Knowledge Retrieval Benchmark Suite

A comprehensive benchmark suite for evaluating graph-based, vector-based, and hybrid knowledge retrieval systems.

## Overview

This benchmark suite provides tools to measure the performance, efficiency, and accuracy of different knowledge retrieval approaches. It focuses on three primary storage and retrieval paradigms:

1. **Graph Databases** - Measuring structured relationship queries
2. **Vector Databases** - Measuring semantic similarity search
3. **Hybrid Systems** - Measuring combined approaches that leverage both paradigms

## Features

- Customizable benchmark configurations
- Synthetic data generation with configurable complexity
- Comprehensive metrics collection (latency, throughput, memory usage, etc.)
- Visualization tools for benchmark results
- Benchmark specific aspects like:
  - Entity and relation creation
  - Query performance
  - Database persistence overhead
  - Semantic search accuracy
  - Scalability with increasing data size
  - Hybrid knowledge queries

## Directory Structure

```
benchmark_suite/
├── __init__.py           # Package initialization
├── config.py             # Benchmark configuration
├── data_generators.py    # Synthetic data generators
├── main.py               # Main benchmark script
├── metrics.py            # Metrics collection
├── runners.py            # Benchmark execution logic
├── utils.py              # Utility functions
├── visualization.py      # Results visualization
├── benchmarks/           # Individual benchmark implementations
│   ├── __init__.py
│   ├── graph_benchmarks.py     # Graph database benchmarks
│   ├── vector_benchmarks.py    # Vector database benchmarks
│   ├── hybrid_benchmarks.py    # Hybrid system benchmarks
├── datasets/             # Generated test datasets
├── results/              # Benchmark results storage
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/knowledge-retrieval-benchmarks.git
cd knowledge-retrieval-benchmarks
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

Run the complete benchmark suite:

```bash
python -m benchmark_suite.main
```

### Command-line Options

- `--config PATH`: Path to a custom configuration file
- `--output-dir DIR`: Directory to save benchmark results (default: `benchmark_results`)
- `--generate-data`: Generate new synthetic test data
- `--data-dir DIR`: Directory for test datasets (default: `benchmark_suite/datasets`)
- `--benchmark-type TYPE`: Type of benchmarks to run (choices: `all`, `graph`, `vector`, `hybrid`, default: `all`)
- `--visualize`: Generate visualizations of benchmark results
- `--verbose`: Enable verbose logging

### Examples

Run only graph database benchmarks:
```bash
python -m benchmark_suite.main --benchmark-type graph
```

Generate new test data and run benchmarks:
```bash
python -m benchmark_suite.main --generate-data
```

Run benchmarks with a custom configuration:
```bash
python -m benchmark_suite.main --config custom_config.json
```

Run benchmarks and generate visualizations:
```bash
python -m benchmark_suite.main --visualize
```

## Benchmark Types

### Graph Database Benchmarks

- Entity creation performance
- Relation creation performance
- Query performance (by entity type, relation type, etc.)
- Graph traversal performance
- Persistence overhead

### Vector Database Benchmarks

- Text embedding performance
- Entity storage
- Semantic search performance
- Database scalability
- Persistence overhead

### Hybrid System Benchmarks

- Document processing pipeline
- Hybrid search (combining vector and graph queries)
- Complex knowledge queries
- System comparison (hybrid vs. individual components)
- Scalability with increasing data volumes

## Configuration

You can customize the benchmark suite by creating a JSON configuration file. Example:

```json
{
  "name": "custom_benchmark",
  "description": "Custom benchmark configuration",
  "seed": 42,
  "output_dir": "custom_results",
  "dataset_size": "large",
  "num_entities": 5000,
  "num_relations": 20000,
  "num_documents": 500
}
```

## Visualization

The benchmark suite can generate visualizations to help interpret the results:

- Bar charts comparing latency across benchmarks
- Bar charts comparing throughput across benchmarks
- Line plots showing scaling performance
- Comparison plots between different approaches
- HTML report with all metrics and charts

## Extending the Suite

To add new benchmarks:

1. Create a new benchmark function in the appropriate file under `benchmarks/`
2. Follow the benchmark function signature pattern (see existing examples)
3. The function name should start with `benchmark_` to be discovered automatically

## License

This project is licensed under the MIT License - see the LICENSE file for details.
