#!/usr/bin/env python3
"""
RAG Top Benchmark

This script runs benchmarks for the top 3 RAG metrics using the top 3 datasets.
Metrics evaluated:
1. Retrieval accuracy/context relevance
2. Answer relevance/quality
3. Factual correctness/hallucination detection

Datasets used:
1. Natural Questions (NQ)
2. MS MARCO
3. HotpotQA

NOTE: This example has been updated to work with the new location of the RAG modules.
For a simpler standalone implementation that doesn't require external services like Qdrant,
please use the standalone version at: benchmark/rag/code/rag_top_benchmark.py
"""

# Run the standalone version
if __name__ == "__main__":
    import sys
    import os

    # Import and run the standalone version
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from benchmark.rag.code.rag_top_benchmark import main
    main() 