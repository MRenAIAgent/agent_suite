"""
Configuration for RAG-Enhanced Math Learning System.

This module provides configuration settings for integrating the repository's
RAG components with the math learning knowledge graphs and learning graphs.
"""

import os
from typing import Dict, Any, Optional
from dataclasses import dataclass

from agents.rag.api.rag_service import RagService
from agents.rag.factory import create_rag_service


@dataclass
class RagConfig:
    """Configuration for RAG integration."""
    
    # Primary storage type to use initially
    primary_storage_type: str = "graph"  # "graph", "vector", or "key_value"
    
    # Storage backend configurations
    vector_storage_type: str = "memory"  # "qdrant" or "memory"
    graph_storage_type: str = "memory"   # "neo4j" or "memory"
    kv_storage_type: str = "memory"      # "redis" or "memory"
    
    # Embedding configuration
    embedding_provider: str = "sentence-transformer"  # "openai", "sentence-transformer", "dummy"
    embedding_model: str = "all-MiniLM-L6-v2"
    
    # Vector storage settings
    vector_collection_name: str = "math_concepts"
    vector_size: int = 384
    
    # External service settings
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_username: str = "neo4j"
    neo4j_password: str = "password"
    redis_host: str = "localhost"
    redis_port: int = 6379
    
    # OpenAI settings
    openai_api_key: Optional[str] = None
    
    # New fields for graph-first configurations
    chunk_size: int = 512
    chunk_overlap: int = 50
    max_results: int = 10
    similarity_threshold: float = 0.7
    
    def __post_init__(self):
        """Load configuration from environment variables."""
        self.openai_api_key = self.openai_api_key or os.getenv("OPENAI_API_KEY")
        self.qdrant_host = os.getenv("QDRANT_HOST", self.qdrant_host)
        self.qdrant_port = int(os.getenv("QDRANT_PORT", str(self.qdrant_port)))
        self.neo4j_uri = os.getenv("NEO4J_URI", self.neo4j_uri)
        self.neo4j_username = os.getenv("NEO4J_USERNAME", self.neo4j_username)
        self.neo4j_password = os.getenv("NEO4J_PASSWORD", self.neo4j_password)
        self.redis_host = os.getenv("REDIS_HOST", self.redis_host)
        self.redis_port = int(os.getenv("REDIS_PORT", str(self.redis_port)))


def create_rag_config_dict(config: RagConfig) -> Dict[str, Any]:
    """
    Create a configuration dictionary for RAG service creation.
    
    Args:
        config: RagConfig instance
        
    Returns:
        Configuration dictionary for create_rag_service
    """
    rag_config = {}
    
    # Always configure graph storage since it's our primary storage
    rag_config["graph"] = {
        "type": config.graph_storage_type
    }
    
    if config.graph_storage_type == "neo4j":
        rag_config["graph"].update({
            "uri": config.neo4j_uri,
            "username": config.neo4j_username,
            "password": config.neo4j_password
        })
    
    # Configure vector storage for semantic search capabilities
    rag_config["vector"] = {
        "type": config.vector_storage_type,
        "collection_name": config.vector_collection_name,
        "vector_size": config.vector_size,
        "embedding": {
            "type": config.embedding_provider,
            "model": config.embedding_model
        }
    }
    
    if config.vector_storage_type == "qdrant":
        rag_config["vector"].update({
            "host": config.qdrant_host,
            "port": config.qdrant_port
        })
    
    if config.embedding_provider == "openai" and config.openai_api_key:
        rag_config["vector"]["embedding"]["api_key"] = config.openai_api_key
    
    # Configure key-value storage for metadata and caching
    rag_config["key_value"] = {
        "type": config.kv_storage_type
    }
    
    if config.kv_storage_type == "redis":
        rag_config["key_value"].update({
            "host": config.redis_host,
            "port": config.redis_port
        })
    
    return rag_config


async def create_configured_rag_service(config: RagConfig = None) -> RagService:
    """
    Create a RAG service with the specified configuration.
    
    Args:
        config: RagConfig instance, defaults to memory-based configuration
        
    Returns:
        Configured RagService instance
    """
    if config is None:
        config = RagConfig()
    
    rag_config_dict = create_rag_config_dict(config)
    return await create_rag_service(rag_config_dict)


async def create_simple_rag_service() -> RagService:
    """
    Create a simple RAG service with only graph and key-value storage for testing.
    
    Returns:
        Configured RagService instance with memory backends
    """
    simple_config = {
        "graph": {
            "type": "memory"
        },
        "key_value": {
            "type": "memory"
        }
        # No vector storage to avoid external dependencies
    }
    
    return await create_rag_service(simple_config)


# Default configurations for different environments
def get_memory_config() -> RagConfig:
    """Get configuration for memory-based storage (testing/development)."""
    return RagConfig(
        primary_storage_type="graph",
        vector_storage_type="qdrant",
        graph_storage_type="memory",
        kv_storage_type="memory",
        embedding_provider="dummy"
    )


def get_simple_memory_config() -> RagConfig:
    """Get configuration for simple memory-based storage without vector storage."""
    return RagConfig(
        primary_storage_type="graph",
        vector_storage_type="qdrant",
        graph_storage_type="memory",
        kv_storage_type="memory",
        embedding_provider="dummy"
    )


def get_production_config() -> RagConfig:
    """Get configuration for production environment."""
    return RagConfig(
        primary_storage_type="graph",
        vector_storage_type="qdrant",
        graph_storage_type="neo4j",
        kv_storage_type="redis",
        embedding_provider="openai"
    )


def get_local_config() -> RagConfig:
    """Get configuration for local development with external services."""
    return RagConfig(
        primary_storage_type="graph",
        vector_storage_type="qdrant",
        graph_storage_type="memory",
        kv_storage_type="memory",
        embedding_provider="sentence-transformer"
    )


def get_storage_configs(config: RagConfig) -> Dict[str, Dict[str, Any]]:
    """
    Get storage configurations for RAG services.
    
    Args:
        config: RAG configuration
        
    Returns:
        Dictionary of storage configurations
    """
    storage_configs = {
        "vector_storage": {
            "type": config.vector_storage_type,
            "config": {}
        },
        "graph_storage": {
            "type": config.graph_storage_type,
            "config": {}
        },
        "kv_storage": {
            "type": config.kv_storage_type,
            "config": {}
        }
    }
    
    # Configure vector storage
    if config.vector_storage_type == "qdrant":
        storage_configs["vector_storage"]["config"] = {
            "host": config.qdrant_host,
            "port": config.qdrant_port,
            "collection_name": config.vector_collection_name,
            "vector_size": config.vector_size
        }
    
    # Configure embedding provider
    embedding_config = {
        "provider": config.embedding_provider,
        "model": config.embedding_model,
        "dimension": config.vector_size
    }
    
    if config.embedding_provider == "openai":
        embedding_config["api_key"] = config.openai_api_key
    
    storage_configs["embedding"] = embedding_config
    
    return storage_configs


def get_rag_service_config(config: RagConfig) -> Dict[str, Any]:
    """
    Get RAG service configuration.
    
    Args:
        config: RAG configuration
        
    Returns:
        RAG service configuration dictionary
    """
    return {
        "max_context_length": 4000,
        "top_k_retrieval": 5,
        "rerank_enabled": True,
        "query_expansion_enabled": True,
        "concept_similarity_threshold": 0.7,
        "learning_path_max_depth": 5,
        "personalization_enabled": True
    }


# Default configuration instance
DEFAULT_RAG_CONFIG = get_memory_config()
DEFAULT_STORAGE_CONFIGS = get_storage_configs(DEFAULT_RAG_CONFIG)
DEFAULT_SERVICE_CONFIG = get_rag_service_config(DEFAULT_RAG_CONFIG)


def get_graph_first_config() -> RagConfig:
    """
    Get a graph-first configuration that uses RAG graph storage as primary.
    
    This configuration demonstrates the approach where the RAG system's
    graph storage is used as the foundation, with other storage types
    providing additional capabilities.
    
    Returns:
        RagConfig configured for graph-first approach
    """
    return RagConfig(
        primary_storage_type="graph",
        graph_storage_type="memory",
        vector_storage_type="memory",
        kv_storage_type="memory",
        embedding_provider="sentence-transformer",
        embedding_model="all-MiniLM-L6-v2",
        chunk_size=512,
        chunk_overlap=50,
        max_results=10,
        similarity_threshold=0.7
    )


def get_graph_only_config() -> RagConfig:
    """
    Get a minimal graph-only configuration.
    
    This configuration uses only the RAG system's graph storage,
    with minimal vector capabilities for basic functionality.
    
    Returns:
        RagConfig configured for graph-only approach
    """
    return RagConfig(
        primary_storage_type="graph",
        graph_storage_type="memory",
        vector_storage_type="memory",
        kv_storage_type="memory",
        embedding_provider="dummy",  # Minimal embedding
        embedding_model="dummy",
        chunk_size=512,
        chunk_overlap=50,
        max_results=10,
        similarity_threshold=0.7
    )


def get_production_graph_config() -> RagConfig:
    """
    Get a production-ready graph-first configuration.
    
    This configuration uses production-grade storage backends
    with the RAG graph storage as the primary mechanism.
    
    Returns:
        RagConfig configured for production graph-first approach
    """
    return RagConfig(
        primary_storage_type="graph",
        graph_storage_type=os.getenv("GRAPH_STORAGE_TYPE", "neo4j"),
        vector_storage_type=os.getenv("VECTOR_STORAGE_TYPE", "qdrant"),
        kv_storage_type=os.getenv("KV_STORAGE_TYPE", "redis"),
        embedding_provider=os.getenv("EMBEDDING_PROVIDER", "openai"),
        embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        neo4j_uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        neo4j_username=os.getenv("NEO4J_USERNAME", "neo4j"),
        neo4j_password=os.getenv("NEO4J_PASSWORD"),
        chunk_size=1024,
        chunk_overlap=100,
        max_results=20,
        similarity_threshold=0.8
    )


def create_config_by_flags(
    primary_storage: str = "graph",
    enable_vector: bool = True,
    enable_kv: bool = True,
    production_mode: bool = False
) -> RagConfig:
    """
    Create a RAG configuration using flags to control storage types.
    
    Args:
        primary_storage: Primary storage type ("graph", "vector", or "key_value")
        enable_vector: Whether to enable vector storage
        enable_kv: Whether to enable key-value storage
        production_mode: Whether to use production-grade backends
        
    Returns:
        RagConfig based on the specified flags
    """
    if production_mode:
        base_config = get_production_graph_config()
    else:
        base_config = get_memory_config()
    
    # Override primary storage type
    base_config.primary_storage_type = primary_storage
    
    # Disable storage types based on flags
    if not enable_vector:
        base_config.vector_storage_type = "disabled"
        base_config.embedding_provider = "dummy"
    
    if not enable_kv:
        base_config.kv_storage_type = "disabled"
    
    return base_config


def get_real_backends_config() -> RagConfig:
    """Get configuration for real backends (Neo4j + Redis) testing."""
    return RagConfig(
        primary_storage_type="graph",
        vector_storage_type="memory",  # Keep vector as memory for simplicity
        graph_storage_type="neo4j",
        kv_storage_type="redis",
        embedding_provider="dummy",
        # Neo4j settings
        neo4j_uri="bolt://localhost:7687",
        neo4j_username="neo4j",
        neo4j_password="password",
        # Redis settings
        redis_host="localhost",
        redis_port=6379
    ) 