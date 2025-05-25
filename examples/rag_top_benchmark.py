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

import sys
import os
import logging
import argparse

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def run_rag_benchmark(dry_run=False):
    """Run the RAG benchmark."""
    if dry_run:
        logger.info("RAG Benchmark dry run:")
        logger.info("  This would run the RAG benchmark with the following datasets:")
        logger.info("    - Natural Questions (NQ)")
        logger.info("    - MS MARCO")
        logger.info("    - HotpotQA")
        logger.info("  And evaluate the following metrics:")
        logger.info("    - Retrieval accuracy/context relevance")
        logger.info("    - Answer relevance/quality")
        logger.info("    - Factual correctness/hallucination detection")
        logger.info("Dry run completed. Exiting.")
        return
        
    try:
        # Import and run the standalone version
        from benchmark.rag.code.rag_top_benchmark import main
        logger.info("Running RAG benchmark...")
        main()
        logger.info("RAG benchmark completed successfully!")
    except ImportError as e:
        logger.error(f"Error importing RAG benchmark: {e}")
        logger.error("Make sure the benchmark code is properly installed.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error running RAG benchmark: {e}")
        sys.exit(1)

# Run the standalone version
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG Top Benchmark")
    parser.add_argument("--dry-run", action="store_true", help="Don't actually run the benchmark, just print information")
    args = parser.parse_args()
    
    run_rag_benchmark(dry_run=args.dry_run) 