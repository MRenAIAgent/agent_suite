"""
Storage factory for the math learning knowledge graph.

This module provides a factory function to create the appropriate storage
backend based on configuration, abstracting the choice between NetworkX
and RAG storage from the rest of the system.
"""

from typing import Optional

from .storage_interface import KnowledgeGraphStorageInterface
from .networkx_storage import NetworkXStorage
from .rag_storage import RagStorage
from math_learning.config.storage_config import (
    MathLearningStorageConfig, 
    StorageBackend,
    get_default_config,
    get_config_from_env
)


def create_storage(config: Optional[MathLearningStorageConfig] = None, 
                  name: str = "") -> KnowledgeGraphStorageInterface:
    """
    Create a storage backend based on configuration.
    
    Args:
        config: Storage configuration. If None, uses default configuration.
        name: Optional name for the knowledge graph
        
    Returns:
        A storage backend implementing KnowledgeGraphStorageInterface
        
    Raises:
        ValueError: If an unsupported storage backend is specified
    """
    if config is None:
        config = get_default_config()
    
    if config.backend == StorageBackend.NETWORKX:
        return NetworkXStorage(config, name)
    elif config.backend == StorageBackend.RAG:
        return RagStorage(config, name)
    else:
        raise ValueError(f"Unsupported storage backend: {config.backend}")


def create_storage_from_env(name: str = "") -> KnowledgeGraphStorageInterface:
    """
    Create a storage backend based on environment variables.
    
    Args:
        name: Optional name for the knowledge graph
        
    Returns:
        A storage backend implementing KnowledgeGraphStorageInterface
    """
    config = get_config_from_env()
    return create_storage(config, name)


def create_networkx_storage(name: str = "", 
                           persistence_file: str = "data/algebra_knowledge_graph.json",
                           auto_save: bool = True) -> KnowledgeGraphStorageInterface:
    """
    Create a NetworkX storage backend with custom settings.
    
    Args:
        name: Optional name for the knowledge graph
        persistence_file: Path to the persistence file
        auto_save: Whether to auto-save changes
        
    Returns:
        A NetworkX storage backend
    """
    from math_learning.config.storage_config import get_networkx_config
    
    config = get_networkx_config()
    config.networkx_persistence_file = persistence_file
    config.networkx_auto_save = auto_save
    
    return NetworkXStorage(config, name)


def create_rag_storage(name: str = "", 
                      use_production: bool = False,
                      enable_vector_search: bool = True) -> KnowledgeGraphStorageInterface:
    """
    Create a RAG storage backend with custom settings.
    
    Args:
        name: Optional name for the knowledge graph
        use_production: Whether to use production RAG backends
        enable_vector_search: Whether to enable vector search
        
    Returns:
        A RAG storage backend
    """
    from math_learning.config.storage_config import get_rag_memory_config, get_rag_production_config
    
    if use_production:
        config = get_rag_production_config()
    else:
        config = get_rag_memory_config()
    
    config.rag_enable_vector_search = enable_vector_search
    if name:
        config.rag_graph_name = name
    
    return RagStorage(config, name) 