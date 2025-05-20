"""
Benchmark Metrics Collection and Processing

This module handles the collection, calculation, and storage of
benchmark metrics such as latency, throughput, memory usage, etc.
"""

import time
import os
import json
import statistics
import psutil
import numpy as np
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass, field, asdict


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""
    
    # Benchmark information
    benchmark_name: str
    benchmark_type: str
    config: Dict[str, Any]
    timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S"))
    
    # Performance metrics
    latency_ms: List[float] = field(default_factory=list)
    throughput_ops: Optional[float] = None
    memory_mb: Optional[float] = None
    storage_bytes: Optional[int] = None
    
    # Quality metrics
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    
    # Custom metrics
    custom_metrics: Dict[str, Any] = field(default_factory=dict)
    
    def add_latency(self, latency_ms: float) -> None:
        """Add a latency measurement."""
        self.latency_ms.append(latency_ms)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary representation."""
        data = asdict(self)
        data["stats"] = self.calculate_stats()
        return data
    
    def calculate_stats(self) -> Dict[str, Any]:
        """Calculate summary statistics from collected metrics."""
        stats = {}
        
        # Latency statistics
        if self.latency_ms:
            stats["latency"] = {
                "min_ms": min(self.latency_ms),
                "max_ms": max(self.latency_ms),
                "mean_ms": statistics.mean(self.latency_ms),
                "median_ms": statistics.median(self.latency_ms),
                "p95_ms": np.percentile(self.latency_ms, 95),
                "p99_ms": np.percentile(self.latency_ms, 99),
                "stddev_ms": statistics.stdev(self.latency_ms) if len(self.latency_ms) > 1 else 0
            }
        
        # Other metrics
        if self.throughput_ops is not None:
            stats["throughput_ops"] = self.throughput_ops
        
        if self.memory_mb is not None:
            stats["memory_mb"] = self.memory_mb
            
        if self.storage_bytes is not None:
            stats["storage_bytes"] = self.storage_bytes
            stats["storage_mb"] = self.storage_bytes / (1024 * 1024)
        
        # Quality metrics
        quality_metrics = {}
        if self.accuracy is not None:
            quality_metrics["accuracy"] = self.accuracy
        if self.precision is not None:
            quality_metrics["precision"] = self.precision
        if self.recall is not None:
            quality_metrics["recall"] = self.recall
        if self.f1_score is not None:
            quality_metrics["f1_score"] = self.f1_score
            
        if quality_metrics:
            stats["quality"] = quality_metrics
        
        # Add custom metrics
        if self.custom_metrics:
            stats["custom"] = self.custom_metrics
            
        return stats
    
    def save(self, output_dir: str) -> str:
        """
        Save results to a JSON file.
        
        Args:
            output_dir: Directory to save results
            
        Returns:
            Path to the saved file
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Create filename
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{self.benchmark_name}_{timestamp}.json"
        filepath = os.path.join(output_dir, filename)
        
        # Calculate stats and prepare data
        stats = self.calculate_stats()
        data = {
            "benchmark_name": self.benchmark_name,
            "benchmark_type": self.benchmark_type,
            "timestamp": self.timestamp,
            "config": self.config,
            "stats": stats,
            "raw_data": {
                "latency_ms": self.latency_ms
            }
        }
        
        # Add other metrics to raw data
        for key, value in self.custom_metrics.items():
            data["raw_data"][key] = value
        
        # Save to file
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
            
        return filepath


class MetricsCollector:
    """Collects and processes metrics for benchmarks."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize metrics collector.
        
        Args:
            config: Benchmark configuration
        """
        self.config = config
        self.process = psutil.Process()
        self.results = {}
    
    def start_benchmark(self, benchmark_name: str, benchmark_type: str) -> BenchmarkResult:
        """
        Start collecting metrics for a benchmark.
        
        Args:
            benchmark_name: Name of the benchmark
            benchmark_type: Type of benchmark (e.g., "graph", "vector", "hybrid")
            
        Returns:
            New BenchmarkResult instance
        """
        result = BenchmarkResult(
            benchmark_name=benchmark_name,
            benchmark_type=benchmark_type,
            config=self.config
        )
        
        self.results[benchmark_name] = result
        return result
    
    def measure_latency(self, func: Callable, *args, **kwargs) -> float:
        """
        Measure latency of a function call.
        
        Args:
            func: Function to measure
            *args, **kwargs: Arguments to pass to the function
            
        Returns:
            Latency in milliseconds
        """
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        latency_ms = (end_time - start_time) * 1000
        return latency_ms
    
    def measure_memory(self) -> float:
        """
        Measure current memory usage.
        
        Returns:
            Memory usage in MB
        """
        memory_info = self.process.memory_info()
        return memory_info.rss / (1024 * 1024)  # Convert bytes to MB
    
    def calculate_throughput(self, operations: int, elapsed_time_sec: float) -> float:
        """
        Calculate throughput.
        
        Args:
            operations: Number of operations performed
            elapsed_time_sec: Elapsed time in seconds
            
        Returns:
            Operations per second
        """
        if elapsed_time_sec > 0:
            return operations / elapsed_time_sec
        return 0
    
    def get_results(self, benchmark_name: Optional[str] = None) -> Union[BenchmarkResult, Dict[str, BenchmarkResult]]:
        """
        Get benchmark results.
        
        Args:
            benchmark_name: Optional name of specific benchmark
            
        Returns:
            Benchmark result or dictionary of all results
        """
        if benchmark_name:
            return self.results.get(benchmark_name)
        return self.results
    
    def save_results(self, output_dir: str) -> List[str]:
        """
        Save all benchmark results.
        
        Args:
            output_dir: Directory to save results
            
        Returns:
            List of saved file paths
        """
        saved_files = []
        
        for benchmark_name, result in self.results.items():
            filepath = result.save(output_dir)
            saved_files.append(filepath)
            
        return saved_files 