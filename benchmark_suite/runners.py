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

from benchmark_suite.config import BenchmarkConfig
from benchmark_suite.metrics import MetricsCollector
from benchmark_suite.data_generators import generate_dataset


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
        self.metrics = MetricsCollector(self.config.to_dict())
        
        # Set random seed for reproducibility
        import random
        random.seed(self.config.seed)
        try:
            import numpy as np
            np.random.seed(self.config.seed)
        except ImportError:
            pass
        
        # Benchmark registry
        self.benchmarks = {}
        
        # Dataset path
        self.dataset_path = None
    
    def load_benchmarks(self, benchmark_modules: Optional[List[str]] = None) -> None:
        """
        Load benchmark modules dynamically.
        
        Args:
            benchmark_modules: List of module names to load
        """
        if benchmark_modules is None:
            # Default benchmark modules
            benchmark_modules = [
                "benchmark_suite.benchmarks.graph_benchmarks",
                "benchmark_suite.benchmarks.vector_benchmarks",
                "benchmark_suite.benchmarks.hybrid_benchmarks"
            ]
        
        for module_name in benchmark_modules:
            try:
                # Import the module
                module = importlib.import_module(module_name)
                
                # Look for benchmark functions
                for attr_name in dir(module):
                    if attr_name.startswith("benchmark_"):
                        benchmark_func = getattr(module, attr_name)
                        if callable(benchmark_func):
                            self.benchmarks[attr_name] = benchmark_func
                
                logger.info(f"Loaded benchmarks from {module_name}")
            except ImportError as e:
                logger.error(f"Failed to import benchmark module {module_name}: {e}")
    
    def prepare_dataset(self, dataset_path: Optional[str] = None) -> str:
        """
        Prepare the dataset for benchmarks.
        
        Args:
            dataset_path: Optional path to existing dataset
            
        Returns:
            Path to the dataset directory
        """
        if dataset_path and os.path.exists(dataset_path):
            # Use existing dataset
            self.dataset_path = dataset_path
            logger.info(f"Using existing dataset at {dataset_path}")
        else:
            # Generate new dataset
            logger.info("Generating new benchmark dataset...")
            self.dataset_path = generate_dataset(self.config.to_dict())
            logger.info(f"Generated dataset at {self.dataset_path}")
        
        return self.dataset_path
    
    def run_benchmark(
        self, 
        benchmark_name: str = None, 
        benchmark_func: Optional[Callable] = None,
        benchmark_type: str = None,
        **kwargs
    ) -> Optional[Any]:
        """
        Run a specific benchmark.
        
        Args:
            benchmark_name: Name of the benchmark function
            benchmark_func: Direct reference to benchmark function (alternative to looking up by name)
            benchmark_type: Type of benchmark (graph, vector, hybrid)
            **kwargs: Additional arguments for the benchmark
            
        Returns:
            Benchmark result object or None if error
        """
        # Get the benchmark function
        if benchmark_func is None:
            if benchmark_name not in self.benchmarks:
                logger.error(f"Benchmark {benchmark_name} not found")
                return None
            benchmark_func = self.benchmarks[benchmark_name]
        
        # If name not provided but func is, use func.__name__
        if benchmark_name is None and benchmark_func is not None:
            benchmark_name = getattr(benchmark_func, "__name__", "unnamed_benchmark")
        
        # Extract benchmark type from function name if not provided
        if benchmark_type is None:
            if "graph" in benchmark_name:
                benchmark_type = "graph"
            elif "vector" in benchmark_name:
                benchmark_type = "vector"
            elif "hybrid" in benchmark_name:
                benchmark_type = "hybrid"
            else:
                benchmark_type = "other"
        
        # Create result object
        result = self.metrics.start_benchmark(benchmark_name, benchmark_type)
        
        # Set up kwargs for the benchmark
        benchmark_kwargs = {
            "config": self.config.to_dict() if hasattr(self.config, "to_dict") else self.config,
            "dataset_path": self.dataset_path,
            "metrics": self.metrics,
            "result": result
        }
        benchmark_kwargs.update(kwargs)
        
        # Run the benchmark
        logger.info(f"Running benchmark: {benchmark_name}")
        
        try:
            # Warmup iterations
            if self.config.warmup_iterations > 0:
                logger.info(f"Performing {self.config.warmup_iterations} warmup iterations...")
                for i in range(self.config.warmup_iterations):
                    benchmark_func(**benchmark_kwargs)
                    logger.info(f"Warmup iteration {i+1}/{self.config.warmup_iterations} completed")
            
            # Timed iterations
            logger.info(f"Running {self.config.iterations} benchmark iterations...")
            
            for i in range(self.config.iterations):
                start_time = time.time()
                
                # Run the benchmark
                benchmark_func(**benchmark_kwargs)
                
                elapsed_time = time.time() - start_time
                result.add_latency(elapsed_time * 1000)  # Convert to ms
                
                logger.info(f"Iteration {i+1}/{self.config.iterations} completed in {elapsed_time:.4f}s")
            
            logger.info(f"Benchmark {benchmark_name} completed")
        except Exception as e:
            logger.error(f"Error running benchmark {benchmark_name}: {e}")
            import traceback
            logger.error(traceback.format_exc())
        
        return result
    
    def run_all_benchmarks(self, subset: Optional[List[str]] = None) -> None:
        """
        Run all registered benchmarks or a subset.
        
        Args:
            subset: Optional list of benchmark names to run
        """
        # Ensure we have a dataset
        if not self.dataset_path:
            self.prepare_dataset()
        
        # Get the benchmarks to run
        if subset:
            benchmarks_to_run = [b for b in subset if b in self.benchmarks]
        else:
            benchmarks_to_run = list(self.benchmarks.keys())
        
        if not benchmarks_to_run:
            logger.error("No benchmarks found to run")
            return
        
        # Run each benchmark
        for benchmark_name in benchmarks_to_run:
            self.run_benchmark(benchmark_name)
    
    def save_results(self, output_dir: Optional[str] = None) -> List[str]:
        """
        Save benchmark results.
        
        Args:
            output_dir: Optional directory to save results
            
        Returns:
            List of saved file paths
        """
        if output_dir is None:
            output_dir = self.config.output_dir
        
        # Ensure directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Save config
        config_path = self.config.save(os.path.join(output_dir, "config.json"))
        logger.info(f"Saved benchmark configuration to {config_path}")
        
        # Save metrics
        result_paths = self.metrics.save_results(output_dir)
        logger.info(f"Saved benchmark results to {output_dir}")
        
        return [config_path] + result_paths


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
    runner.prepare_dataset()
    
    # Run all benchmarks
    runner.run_all_benchmarks()
    
    # Save results
    runner.save_results()
    
    logger.info("Benchmark run completed") 