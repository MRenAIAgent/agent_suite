"""
Storage configuration for the math learning system.

This module provides configuration options to choose between different storage backends:
- NetworkX: Traditional in-memory graph with JSON persistence
- RAG: Uses the repository's RAG system for graph storage
"""

import os
from typing import Dict, Any, Optional, Literal
from dataclasses import dataclass
from enum import Enum

from .rag_config import RagConfig, get_memory_config, get_production_config


class StorageBackend(Enum):
    """Available storage backends for the math learning system."""
    NETWORKX = "networkx"
    RAG = "rag"


@dataclass
class MathLearningStorageConfig:
    """Configuration for math learning system storage."""
    
    # Primary storage backend
    backend: StorageBackend = StorageBackend.NETWORKX
    
    # NetworkX-specific settings
    networkx_persistence_file: str = "data/algebra_knowledge_graph.json"
    networkx_auto_save: bool = True
    
    # RAG-specific settings
    rag_config: Optional[RagConfig] = None
    rag_graph_name: str = "K-12 Algebra Knowledge Graph"
    rag_enable_vector_search: bool = True
    rag_enable_metadata_cache: bool = True
    
    # Common settings
    enable_caching: bool = True
    cache_size: int = 1000
    
    def __post_init__(self):
        """Initialize configuration from environment variables."""
        # Allow environment override of storage backend
        env_backend = os.getenv("MATH_LEARNING_STORAGE_BACKEND")
        if env_backend:
            try:
                self.backend = StorageBackend(env_backend.lower())
            except ValueError:
                pass  # Keep default if invalid value
        
        # Initialize RAG config if using RAG backend
        if self.backend == StorageBackend.RAG and self.rag_config is None:
            # Use memory config by default for development
            self.rag_config = get_memory_config()
        
        # Ensure persistence directory exists for NetworkX
        if self.backend == StorageBackend.NETWORKX:
            os.makedirs(os.path.dirname(self.networkx_persistence_file), exist_ok=True)


def get_networkx_config() -> MathLearningStorageConfig:
    """Get configuration for NetworkX storage backend."""
    return MathLearningStorageConfig(
        backend=StorageBackend.NETWORKX,
        networkx_persistence_file="data/algebra_knowledge_graph.json",
        networkx_auto_save=True,
        enable_caching=True
    )


def get_rag_memory_config() -> MathLearningStorageConfig:
    """Get configuration for RAG storage with memory backends."""
    return MathLearningStorageConfig(
        backend=StorageBackend.RAG,
        rag_config=get_memory_config(),
        rag_enable_vector_search=False,  # Disable to avoid external dependencies
        rag_enable_metadata_cache=True,
        enable_caching=True
    )


def get_rag_production_config() -> MathLearningStorageConfig:
    """Get configuration for RAG storage with production backends."""
    return MathLearningStorageConfig(
        backend=StorageBackend.RAG,
        rag_config=get_production_config(),
        rag_enable_vector_search=True,
        rag_enable_metadata_cache=True,
        enable_caching=True
    )


def get_config_from_env() -> MathLearningStorageConfig:
    """Get configuration based on environment variables."""
    backend_env = os.getenv("MATH_LEARNING_STORAGE_BACKEND", "networkx").lower()
    
    if backend_env == "rag":
        # Check if we should use production or memory RAG
        rag_mode = os.getenv("MATH_LEARNING_RAG_MODE", "memory").lower()
        if rag_mode == "production":
            return get_rag_production_config()
        else:
            return get_rag_memory_config()
    else:
        return get_networkx_config()


def get_default_config() -> MathLearningStorageConfig:
    """Get the default configuration for the math learning system."""
    # Default to NetworkX for backward compatibility
    return get_networkx_config() 