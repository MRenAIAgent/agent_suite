"""
Benchmark Suite Configuration

This module provides a configuration class for benchmark parameters
and settings to ensure reproducible benchmarks.
"""

import os
import json
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Union


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark runs."""
    
    # General settings
    name: str = "hybrid_knowledge_benchmark"
    description: str = "Benchmark for hybrid knowledge retrieval system"
    seed: int = 42  # Random seed for reproducibility
    output_dir: str = "benchmark_suite/results"
    
    # Dataset settings
    dataset_size: str = "medium"  # small, medium, large
    dataset_complexity: str = "medium"  # simple, medium, complex
    num_entities: int = 1000
    num_relations: int = 5000
    num_documents: int = 100
    
    # Graph database settings
    graph_name: str = "benchmark_graph"
    graph_persist: bool = True
    graph_persistence_path: str = "benchmark_suite/data/graph"
    
    # Vector database settings
    vector_collection: str = "benchmark_vectors"
    vector_persist: bool = True
    vector_persistence_path: str = "benchmark_suite/data/vector"
    embedding_model: str = "all-MiniLM-L6-v2"
    
    # Benchmark settings
    iterations: int = 3  # Number of times to run each benchmark
    warmup_iterations: int = 1  # Warmup iterations before measuring
    timeout: int = 60  # Maximum time (seconds) for each benchmark
    
    # Query settings
    query_complexity: str = "medium"  # simple, medium, complex
    num_queries: int = 100
    
    # Metrics settings
    measure_memory: bool = True
    measure_storage: bool = True
    measure_accuracy: bool = True
    
    # Custom parameters
    custom_params: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Create output directory if it doesn't exist."""
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Adjust entity and relation counts based on dataset size
        if self.dataset_size == "small":
            self.num_entities = 100
            self.num_relations = 500
            self.num_documents = 10
        elif self.dataset_size == "large":
            self.num_entities = 10000
            self.num_relations = 50000
            self.num_documents = 1000
    
    def save(self, filepath: Optional[str] = None) -> str:
        """
        Save configuration to a JSON file.
        
        Args:
            filepath: Optional path to save the config
            
        Returns:
            Path to the saved config file
        """
        if filepath is None:
            filepath = os.path.join(self.output_dir, f"{self.name}_config.json")
            
        with open(filepath, 'w') as f:
            json.dump(asdict(self), f, indent=2)
            
        return filepath
    
    @classmethod
    def load(cls, filepath: str) -> "BenchmarkConfig":
        """
        Load configuration from a JSON file.
        
        Args:
            filepath: Path to the config file
            
        Returns:
            BenchmarkConfig instance
        """
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
            
        return cls(**config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self) 