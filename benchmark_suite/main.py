#!/usr/bin/env python3
"""
Benchmark Suite Main Script

This script runs the benchmark suite for hybrid knowledge retrieval systems.
"""

import os
import sys
import json
import time
import logging
import argparse
from typing import Dict, List, Any, Optional
import importlib

from benchmark_suite.config import BenchmarkConfig
from benchmark_suite.runners import BenchmarkRunner
from benchmark_suite.data_generators import DataGenerator
from benchmark_suite.utils import setup_logger
from benchmark_suite.visualization import visualize_benchmark_results


logger = logging.getLogger("benchmark_suite")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run knowledge retrieval benchmark suite")
    
    parser.add_argument(
        "--config", 
        type=str,
        default=None,
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--output-dir", 
        type=str,
        default="benchmark_results",
        help="Directory to save benchmark results"
    )
    
    parser.add_argument(
        "--generate-data", 
        action="store_true",
        help="Generate synthetic test data"
    )
    
    parser.add_argument(
        "--data-dir", 
        type=str,
        default="benchmark_suite/datasets",
        help="Directory for test datasets"
    )
    
    parser.add_argument(
        "--benchmark-type", 
        type=str,
        choices=["all", "graph", "vector", "hybrid"],
        default="all",
        help="Type of benchmarks to run"
    )
    
    parser.add_argument(
        "--visualize", 
        action="store_true",
        help="Generate visualizations of benchmark results"
    )
    
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Enable verbose logging"
    )
    
    return parser.parse_args()


def load_config(config_path: Optional[str] = None) -> BenchmarkConfig:
    """
    Load benchmark configuration.
    
    Args:
        config_path: Path to config file
        
    Returns:
        Loaded configuration
    """
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        return BenchmarkConfig(**config_dict)
    
    return BenchmarkConfig()


def generate_data(data_dir: str, config: BenchmarkConfig) -> str:
    """
    Generate benchmark test data.
    
    Args:
        data_dir: Directory to save generated data
        config: Benchmark configuration
        
    Returns:
        Path to generated dataset directory
    """
    logger.info("Generating synthetic benchmark data...")
    
    # Create dataset directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)
    
    # Create dataset subdirectory with timestamp
    timestamp = int(time.time())
    dataset_dir = os.path.join(data_dir, f"dataset_{timestamp}")
    os.makedirs(dataset_dir, exist_ok=True)
    
    # Generate data
    data_generator = DataGenerator()
    
    # 1. Generate entities
    num_entities = config.num_entities
    entities = data_generator.generate_entities(num_entities)
    
    with open(os.path.join(dataset_dir, "entities.json"), 'w') as f:
        json.dump(entities, f, indent=2)
    
    # 2. Generate relations
    num_relations = config.num_relations
    relations = data_generator.generate_relations(
        entities=entities,
        num_relations=num_relations
    )
    
    with open(os.path.join(dataset_dir, "relations.json"), 'w') as f:
        json.dump(relations, f, indent=2)
    
    # 3. Generate documents
    num_documents = config.num_documents
    documents = data_generator.generate_documents(
        entities=entities,
        num_documents=num_documents
    )
    
    with open(os.path.join(dataset_dir, "documents.json"), 'w') as f:
        json.dump(documents, f, indent=2)
    
    logger.info(f"Generated dataset at: {dataset_dir}")
    logger.info(f"- {len(entities)} entities")
    logger.info(f"- {len(relations)} relations")
    logger.info(f"- {len(documents)} documents")
    
    return dataset_dir


def find_benchmarks(benchmark_type: str = "all") -> Dict[str, List[str]]:
    """
    Find benchmark functions based on type.
    
    Args:
        benchmark_type: Type of benchmarks to find
        
    Returns:
        Dictionary mapping module paths to benchmark function names
    """
    benchmark_modules = {}
    
    # Define which modules to search based on benchmark type
    if benchmark_type == "all":
        modules = [
            "benchmark_suite.benchmarks.graph_benchmarks",
            "benchmark_suite.benchmarks.vector_benchmarks",
            "benchmark_suite.benchmarks.hybrid_benchmarks"
        ]
    elif benchmark_type == "graph":
        modules = ["benchmark_suite.benchmarks.graph_benchmarks"]
    elif benchmark_type == "vector":
        modules = ["benchmark_suite.benchmarks.vector_benchmarks"]
    elif benchmark_type == "hybrid":
        modules = ["benchmark_suite.benchmarks.hybrid_benchmarks"]
    else:
        modules = []
    
    # Find benchmark functions in each module
    for module_path in modules:
        try:
            module = importlib.import_module(module_path)
            
            # Find functions starting with "benchmark_"
            benchmark_functions = [
                name for name in dir(module)
                if name.startswith("benchmark_") and callable(getattr(module, name))
            ]
            
            if benchmark_functions:
                benchmark_modules[module_path] = benchmark_functions
        except ImportError as e:
            logger.warning(f"Could not import module {module_path}: {e}")
    
    return benchmark_modules


def save_benchmark_config(config: BenchmarkConfig, output_dir: str) -> str:
    """
    Save benchmark configuration.
    
    Args:
        config: Benchmark configuration
        output_dir: Directory to save configuration
        
    Returns:
        Path to saved configuration file
    """
    os.makedirs(output_dir, exist_ok=True)
    
    config_path = os.path.join(output_dir, "config.json")
    
    with open(config_path, 'w') as f:
        json.dump(config.to_dict(), f, indent=2)
    
    return config_path


def run_benchmarks(
    config: BenchmarkConfig,
    dataset_dir: str,
    output_dir: str,
    benchmark_type: str = "all"
) -> str:
    """
    Run benchmarks based on type.
    
    Args:
        config: Benchmark configuration
        dataset_dir: Path to dataset directory
        output_dir: Directory to save results
        benchmark_type: Type of benchmarks to run
        
    Returns:
        Path to results directory
    """
    logger.info(f"Running {benchmark_type} benchmarks...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Find benchmarks to run
    benchmarks = find_benchmarks(benchmark_type)
    
    if not benchmarks:
        logger.warning(f"No benchmarks found for type: {benchmark_type}")
        return output_dir
    
    # Save configuration
    save_benchmark_config(config, output_dir)
    
    # Create benchmark runner
    runner = BenchmarkRunner(config)
    runner.prepare_dataset(dataset_dir)
    
    # Run benchmarks
    for module_path, benchmark_functions in benchmarks.items():
        module = importlib.import_module(module_path)
        
        module_name = module_path.split('.')[-1]
        benchmark_category = module_name.split('_')[0]  # "graph", "vector", or "hybrid"
        
        for function_name in benchmark_functions:
            benchmark_func = getattr(module, function_name)
            
            logger.info(f"Running benchmark: {function_name}")
            
            # Run the benchmark
            result = runner.run_benchmark(
                benchmark_name=function_name,
                benchmark_func=benchmark_func,
                benchmark_type=benchmark_category
            )
            
            # Skip if the benchmark failed
            if result is None:
                logger.warning(f"Benchmark {function_name} failed, skipping result")
                continue
            
            # Save result
            result_path = os.path.join(output_dir, f"{function_name}.json")
            with open(result_path, 'w') as f:
                json.dump(result.to_dict(), f, indent=2)
            
            logger.info(f"Benchmark {function_name} completed, results saved to {result_path}")
    
    logger.info(f"All benchmarks completed, results saved to {output_dir}")
    
    return output_dir


def main():
    """Main entry point."""
    # Parse command-line arguments
    args = parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logger("benchmark_suite", log_level)
    
    # Load configuration
    config = load_config(args.config)
    
    # Override output directory if specified
    if args.output_dir:
        config.output_dir = args.output_dir
    
    # Create timestamped output directory
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_dir = os.path.join(config.output_dir, f"run_{timestamp}")
    
    logger.info(f"Benchmark run started at {timestamp}")
    logger.info(f"Output directory: {output_dir}")
    
    # Generate or use existing data
    if args.generate_data:
        dataset_dir = generate_data(args.data_dir, config)
    else:
        # Use the most recent dataset if available
        datasets = [
            os.path.join(args.data_dir, d) for d in os.listdir(args.data_dir)
            if os.path.isdir(os.path.join(args.data_dir, d)) and d.startswith("dataset_")
        ]
        
        if datasets:
            # Sort by creation time (newest first)
            datasets.sort(key=lambda d: os.path.getctime(d), reverse=True)
            dataset_dir = datasets[0]
            logger.info(f"Using existing dataset: {dataset_dir}")
        else:
            logger.info("No existing datasets found, generating new data")
            dataset_dir = generate_data(args.data_dir, config)
    
    # Run benchmarks
    results_dir = run_benchmarks(
        config=config,
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        benchmark_type=args.benchmark_type
    )
    
    # Generate visualizations if requested
    if args.visualize:
        logger.info("Generating visualizations...")
        report_path = visualize_benchmark_results(results_dir)
        if report_path:
            logger.info(f"Benchmark report generated: {report_path}")
    
    logger.info("Benchmark run completed")
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 