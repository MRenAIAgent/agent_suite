"""
Benchmark Suite Configuration

This module provides a configuration class for benchmark parameters
and settings to ensure reproducible benchmarks.
"""

from dataclasses import dataclass, field, asdict
import json
from pathlib import Path
from typing import Dict, Any, List, Optional


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark runs."""
    
    # Basic information
    name: str = "hybrid_knowledge_benchmark"
    description: str = "Benchmark for hybrid knowledge retrieval system"
    
    output_dir: str = "benchmark/rag/results"
    
    # System under test configuration
    num_documents: int = 1000
    num_entities: int = 200
    num_relationships: int = 500
    
    # Graph database settings
    graph_database: str = "graphiti"  # Options: graphiti, neo4j
    graph_name: str = "benchmark_graph"
    # Where to store the graph data
    graph_persistence_path: str = "benchmark/rag/data/graph"
    
    # Vector database settings
    vector_database: str = "qdrant"  # Options: qdrant, chroma, faiss
    vector_collection: str = "benchmark_vectors"
    # Where to store the vector data
    vector_persistence_path: str = "benchmark/rag/data/vector"
    
    # Benchmark settings
    iterations: int = 3  # Number of times to run each benchmark
    warmup_iterations: int = 1  # Number of warmup iterations (not counted)
    timeout: int = 60  # Maximum time (seconds) for each benchmark
    
    # Query settings
    num_queries: int = 50
    query_complexity: str = "medium"  # Options: simple, medium, complex
    
    # Embedding settings
    embedding_model: str = "sentence-transformers"
    embedding_model_name: str = "all-MiniLM-L6-v2"
    vector_dimensions: int = 384
    
    # Additional parameters
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the configuration to a dictionary.
        
        Returns:
            Dictionary representation of the config
        """
        return asdict(self)
    
    def save(self, filepath: str) -> None:
        """
        Save the configuration to a JSON file.
        
        Args:
            filepath: Path to save the configuration
        """
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> "BenchmarkConfig":
        """
        Load configuration from a JSON file.
        
        Args:
            filepath: Path to load the configuration from
            
        Returns:
            BenchmarkConfig instance
        """
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        
        return cls(**config_dict) 