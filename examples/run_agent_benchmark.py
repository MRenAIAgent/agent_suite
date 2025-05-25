#!/usr/bin/env python3
"""
Run Agent Benchmark

This script runs the agent benchmark from the new location.
"""

import sys
import os
import argparse
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    # Import create_and_run_benchmark function from the benchmark module
    from benchmark.agent.code.agent_benchmark import create_and_run_benchmark
    BENCHMARK_AVAILABLE = True
except ImportError as e:
    logger.error(f"Error importing agent benchmark: {e}")
    logger.error("Make sure the benchmark code is properly installed.")
    BENCHMARK_AVAILABLE = False


def main():
    """Parse arguments and run the benchmark."""
    if not BENCHMARK_AVAILABLE:
        logger.error("Agent benchmark is not available. Exiting.")
        sys.exit(1)
        
    parser = argparse.ArgumentParser(description="Agent Benchmark")
    parser.add_argument("--agents", nargs="+", default=["custom_react", "langchain_react", "langchain_structured"],
                        help="Agents to benchmark")
    parser.add_argument("--model-name", default="gpt-3.5-turbo", help="Model name")
    parser.add_argument("--llm-provider", default="openai", help="LLM provider")
    parser.add_argument("--output-dir", default="benchmark/agent/results", help="Output directory")
    parser.add_argument("--max-cases", type=int, default=2, help="Maximum number of test cases (default: 2)")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--dataset-path", default=None, help="Path to dataset")
    parser.add_argument("--dry-run", action="store_true", help="Don't actually run the benchmark, just print settings")
    
    args = parser.parse_args()
    
    # Print settings
    logger.info("Agent Benchmark Settings:")
    logger.info(f"  Agents: {args.agents}")
    logger.info(f"  Model: {args.model_name} (Provider: {args.llm_provider})")
    logger.info(f"  Output Directory: {args.output_dir}")
    logger.info(f"  Max Cases: {args.max_cases}")
    logger.info(f"  Dataset: {'Default' if args.dataset_path is None else args.dataset_path}")
    
    if args.dry_run:
        logger.info("Dry run completed. Exiting.")
        return
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run the benchmark
    try:
        create_and_run_benchmark(args)
        logger.info("Benchmark completed successfully!")
    except Exception as e:
        logger.error(f"Error running benchmark: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 