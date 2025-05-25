#!/usr/bin/env python3
"""
Benchmark Suite Main Entry Point

This module provides the main entry point for running benchmarks.
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

# Add the parent directory to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from benchmark.rag.code.config import BenchmarkConfig
from benchmark.rag.code.runners import BenchmarkRunner
from benchmark.rag.code.visualization import generate_report

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run knowledge system benchmarks")
    
    parser.add_argument(
        "--config", 
        type=str,
        help="Path to benchmark configuration file"
    )
    
    parser.add_argument(
        "--output-dir", 
        type=str,
        default=None,
        help="Directory to store benchmark results"
    )
    
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Name for this benchmark run"
    )
    
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        choices=["graph", "vector", "hybrid", "all"],
        default=["all"],
        help="Which benchmark modules to run"
    )
    
    parser.add_argument(
        "--list-benchmarks",
        action="store_true",
        help="List available benchmarks and exit"
    )
    
    parser.add_argument(
        "--generate-dataset",
        action="store_true",
        help="Generate dataset and exit"
    )
    
    parser.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="Number of iterations to run each benchmark"
    )
    
    return parser.parse_args()


def load_config(args) -> BenchmarkConfig:
    """
    Load configuration from file or create default.
    
    Args:
        args: Command line arguments
        
    Returns:
        BenchmarkConfig object
    """
    if args.config:
        logger.info(f"Loading configuration from {args.config}")
        config = BenchmarkConfig.load(args.config)
    else:
        logger.info("Using default configuration")
        config = BenchmarkConfig()
    
    # Override with command line arguments
    if args.output_dir:
        config.output_dir = args.output_dir
    
    if args.name:
        config.name = args.name
    
    if args.iterations:
        config.iterations = args.iterations
    
    return config


def get_benchmark_modules(args) -> List[str]:
    """
    Get list of benchmark modules to run.
    
    Args:
        args: Command line arguments
        
    Returns:
        List of module names
    """
    modules = []
    
    if "all" in args.benchmarks:
        return [
            "benchmark.rag.code.benchmarks.graph_benchmarks",
            "benchmark.rag.code.benchmarks.vector_benchmarks",
            "benchmark.rag.code.benchmarks.hybrid_benchmarks"
        ]
    
    if "graph" in args.benchmarks:
        modules.append("benchmark.rag.code.benchmarks.graph_benchmarks")
    
    if "vector" in args.benchmarks:
        modules.append("benchmark.rag.code.benchmarks.vector_benchmarks")
    
    if "hybrid" in args.benchmarks:
        modules.append("benchmark.rag.code.benchmarks.hybrid_benchmarks")
    
    return modules


def save_results(results: Dict[str, Any], config: BenchmarkConfig) -> str:
    """
    Save benchmark results.
    
    Args:
        results: Benchmark results
        config: Benchmark configuration
        
    Returns:
        Path to results directory
    """
    # Create timestamp
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    
    # Create results directory
    results_dir = Path(config.output_dir) / f"run_{timestamp}"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save results
    results_file = results_dir / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    
    # Save configuration
    config_file = results_dir / "config.json"
    config.save(str(config_file))
            
    logger.info(f"Results saved to {results_dir}")
    return str(results_dir)


def main():
    """Main entry point for running benchmarks."""
    args = parse_args()
    
    # List benchmarks if requested
    if args.list_benchmarks:
        runner = BenchmarkRunner()
        benchmark_modules = get_benchmark_modules(args)
        runner.load_benchmarks(benchmark_modules)
        
        print("\nAvailable benchmarks:")
        for name, benchmark in runner.benchmarks.items():
            print(f"  {name}: {benchmark.__doc__ or 'No description'}")
        
        return
    
    # Load configuration
    config = load_config(args)
    
    # Create benchmark runner
    runner = BenchmarkRunner(config)
    
    # Get benchmark modules
    benchmark_modules = get_benchmark_modules(args)
    
    # Load benchmarks
    runner.load_benchmarks(benchmark_modules)
    
    # Generate dataset if requested
    if args.generate_dataset:
        logger.info("Generating dataset...")
        runner.generate_dataset()
        logger.info(f"Dataset generated in {config.output_dir}")
        return
    
    # Run benchmarks
    logger.info("Running benchmarks...")
    results = runner.run_benchmarks()
    
    # Save results
    results_dir = save_results(results, config)
    
    # Generate report
    report_file = os.path.join(results_dir, "report.html")
    generate_report(results, config, report_file)
    
    logger.info(f"Report generated: {report_file}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 