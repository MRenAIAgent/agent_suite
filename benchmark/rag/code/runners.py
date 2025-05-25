"""
Benchmark Suite Runners

This module provides the main benchmark runner classes that
coordinate and execute the benchmark tests.
"""

import os
import time
import json
import logging
import importlib
from typing import Dict, List, Any, Optional, Callable, Union

import numpy as np

from benchmark.rag.code.config import BenchmarkConfig
from benchmark.rag.code.metrics import MetricsCollector
from benchmark.rag.code.data_generators import generate_dataset


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("benchmark_suite")


class BenchmarkRunner:
    """Main runner for executing benchmarks."""
    
    def __init__(self, config: Optional[Union[Dict[str, Any], BenchmarkConfig]] = None):
        """
        Initialize the benchmark runner.
        
        Args:
            config: Configuration for benchmarks
        """
        # Convert dict to BenchmarkConfig if needed
        if isinstance(config, dict):
            self.config = BenchmarkConfig(**config)
        elif config is None:
            self.config = BenchmarkConfig()
        else:
            self.config = config
            
        # Initialize metrics collector
        self.metrics = MetricsCollector()
        
        # Track current dataset path
        self.dataset_path = None
        
        # Benchmark registry
        self.benchmarks = {}
        
        # Initialize random seed for reproducibility
        np.random.seed(42)
    
    def load_benchmarks(self, benchmark_modules: Optional[List[str]] = None) -> None:
        """
        Load benchmark modules dynamically.
        
        Args:
            benchmark_modules: List of module names to load
        """
        if benchmark_modules is None:
            # Default benchmark modules
            benchmark_modules = [
                "benchmark.rag.code.benchmarks.graph_benchmarks",
                "benchmark.rag.code.benchmarks.vector_benchmarks",
                "benchmark.rag.code.benchmarks.hybrid_benchmarks"
            ]
        
        for module_name in benchmark_modules:
            try:
                module = importlib.import_module(module_name)
                
                # Find all benchmark functions
                for attr_name in dir(module):
                    if attr_name.startswith("benchmark_"):
                        benchmark_func = getattr(module, attr_name)
                        if callable(benchmark_func):
                            self.benchmarks[attr_name] = benchmark_func
                            
                logger.info(f"Loaded benchmarks from {module_name}")
            except ImportError as e:
                logger.warning(f"Could not load benchmark module {module_name}: {e}")
        
        logger.info(f"Loaded {len(self.benchmarks)} benchmarks")
    
    def generate_dataset(self) -> str:
        """
        Generate benchmark dataset.
        
        Returns:
            Path to generated dataset
        """
        dataset_path = generate_dataset(self.config)
        self.dataset_path = dataset_path
        return dataset_path
    
    def run_benchmarks(self) -> Dict[str, Any]:
        """
        Run all registered benchmarks.
        
        Returns:
            Dictionary with benchmark results
        """
        if not self.benchmarks:
            logger.warning("No benchmarks loaded")
            return {}
        
        results = {}
        
        # Make sure we have a dataset
        if self.dataset_path is None:
            logger.info("No dataset loaded, generating one")
            self.dataset_path = self.generate_dataset()
        
        # Run each benchmark
        for name, benchmark_func in self.benchmarks.items():
            logger.info(f"Running benchmark: {name}")
            
            try:
                # Setup timing
                start_time = time.time()
                
                # Run warmup iterations first
                for i in range(self.config.warmup_iterations):
                    logger.info(f"Warmup iteration {i+1}/{self.config.warmup_iterations}")
                    benchmark_func(self.config, self.dataset_path)
                
                # Run actual benchmark iterations
                iteration_results = []
                for i in range(self.config.iterations):
                    logger.info(f"Benchmark iteration {i+1}/{self.config.iterations}")
                    result = benchmark_func(self.config, self.dataset_path)
                    iteration_results.append(result)
                
                # Calculate aggregate metrics
                benchmark_result = self.metrics.aggregate_results(iteration_results)
                
                # Add timing information
                elapsed_time = time.time() - start_time
                benchmark_result["elapsed_time"] = elapsed_time
                benchmark_result["iterations"] = self.config.iterations
                
                # Store result
                results[name] = benchmark_result
                
                logger.info(f"Benchmark {name} completed in {elapsed_time:.2f}s")
                
            except Exception as e:
                logger.error(f"Error running benchmark {name}: {e}")
                logger.exception(e)
        
        return results
    
    def run_benchmark(self, benchmark_name: str) -> Optional[Dict[str, Any]]:
        """
        Run a specific benchmark by name.
        
        Args:
            benchmark_name: Name of the benchmark to run
            
        Returns:
            Benchmark result or None if benchmark not found
        """
        if benchmark_name not in self.benchmarks:
            logger.error(f"Benchmark {benchmark_name} not found")
            return None
        
        # Make sure we have a dataset
        if self.dataset_path is None:
            logger.info("No dataset loaded, generating one")
            self.dataset_path = self.generate_dataset()
        
        benchmark_func = self.benchmarks[benchmark_name]
        
        try:
            # Setup timing
            start_time = time.time()
            
            # Run warmup iterations first
            for i in range(self.config.warmup_iterations):
                logger.info(f"Warmup iteration {i+1}/{self.config.warmup_iterations}")
                benchmark_func(self.config, self.dataset_path)
            
            # Run actual benchmark iterations
            iteration_results = []
            for i in range(self.config.iterations):
                logger.info(f"Benchmark iteration {i+1}/{self.config.iterations}")
                result = benchmark_func(self.config, self.dataset_path)
                iteration_results.append(result)
            
            # Calculate aggregate metrics
            benchmark_result = self.metrics.aggregate_results(iteration_results)
            
            # Add timing information
            elapsed_time = time.time() - start_time
            benchmark_result["elapsed_time"] = elapsed_time
            benchmark_result["iterations"] = self.config.iterations
            
            logger.info(f"Benchmark {benchmark_name} completed in {elapsed_time:.2f}s")
            
            return benchmark_result
            
        except Exception as e:
            logger.error(f"Error running benchmark {benchmark_name}: {e}")
            logger.exception(e)
            return None


def run_benchmarks(config_path: Optional[str] = None) -> None:
    """
    Run benchmarks from the command line.
    
    Args:
        config_path: Optional path to configuration file
    """
    # Load configuration
    if config_path and os.path.exists(config_path):
        config = BenchmarkConfig.load(config_path)
        logger.info(f"Loaded configuration from {config_path}")
    else:
        config = BenchmarkConfig()
        logger.info("Using default configuration")
    
    # Create runner
    runner = BenchmarkRunner(config)
    
    # Load benchmarks
    runner.load_benchmarks()
    
    # Prepare dataset
    runner.generate_dataset()
    
    # Run all benchmarks
    results = runner.run_benchmarks()
    
    # Save results
    runner.save_results(results)
    
    logger.info("Benchmark run completed") 