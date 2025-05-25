# RAG Benchmarking Framework

A comprehensive benchmarking framework for evaluating Retrieval Augmented Generation (RAG) systems across multiple dimensions including accuracy, relevance, factual correctness, and performance.

## Overview

This benchmarking system provides tools to:

1. **Evaluate RAG performance** across multiple metrics
2. **Test with real or mock backends** for comprehensive assessment
3. **Generate detailed reports** with visualizations and recommendations
4. **Compare different configurations** to optimize your RAG pipeline

The framework is designed to work with the RAG system in `agent_suite`, which supports multiple storage backends including vector databases (Qdrant), graph databases, and key-value stores.

## Components

- **`rag_benchmark.py`**: Main benchmarking script for collecting performance metrics
- **`rag_benchmark_report.py`**: Report generator with visualizations and analysis
- **`rag_e2e_test.py`**: End-to-end test with real Qdrant backend
- **`rag_simple_test.py`**: Simple test with mock storage adaptors
- **`rag_reranking_test.py`**: Focused test for the reranking capability

## Key Metrics

The framework evaluates RAG systems on:

- **Retrieval Accuracy**: How well the system retrieves relevant information
- **Answer Relevance**: How relevant the generated answers are to the questions
- **Factual Correctness**: How factually accurate the answers are
- **Context Relevance**: How relevant the retrieved context is
- **Latency**: Processing time for retrieval and generation
- **Token Usage**: Number of tokens used in the process

## Getting Started

### Prerequisites

- Python 3.7+
- `qdrant-client` for vector database testing
- `sentence-transformers` for embeddings
- Dependencies: numpy, pandas, matplotlib, seaborn, tqdm

### Running a Benchmark

1. Start a Qdrant server (if using the real backend):

```bash
docker run -d --name qdrant-test -p 6333:6333 -p 6334:6334 qdrant/qdrant
```

2. Run the benchmark:

```bash
PYTHONPATH=. python examples/rag_benchmark.py
```

3. Generate a report from the results:

```bash
PYTHONPATH=. python examples/rag_benchmark_report.py benchmark_results/rag_benchmark_results_*.json
```

### Using Mock Backends

For quick testing without external dependencies:

```bash
PYTHONPATH=. python examples/rag_simple_test.py
```

## Customizing Benchmarks

You can customize the benchmark by:

- Creating your own test dataset with questions and reference answers
- Configuring different embedding models
- Adjusting the evaluation metrics and weights
- Testing with different vector database settings

## Visualizations & Reports

The reporting tool generates:

1. Summary metrics chart
2. System performance radar chart
3. Latency distribution analysis
4. Correlation heatmap between metrics
5. Per-query performance breakdown
6. Detailed HTML report with recommendations

## Contributing

Contributions to improve the benchmarking framework are welcome. Areas for enhancement include:

- Adding more sophisticated evaluation metrics
- Supporting additional storage backends
- Improving the reporting capabilities
- Adding support for LLM-based evaluation

## License

[MIT License](LICENSE)
