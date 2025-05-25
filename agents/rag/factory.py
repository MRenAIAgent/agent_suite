"""Factory functions for creating RAG services.

This module provides factory functions for creating and configuring RAG services
with appropriate storage backends.
"""

import logging
from typing import Dict, Any, Optional

from agents.rag.api.rag_service import RagService
from agents.rag.middleware.storage_router import StorageType
from agents.rag.storage.vector.qdrant_storage import QdrantStorageAdaptor, QDRANT_AVAILABLE
from agents.rag.utils.embedding_provider import create_embedding_provider

logger = logging.getLogger(__name__)


async def create_rag_service(config: Dict[str, Any] = None) -> RagService:
    """
    Create and configure a RAG service with appropriate storage backends.
    
    Args:
        config: Configuration dictionary with settings for each storage backend
            Example:
            {
                "vector": {
                    "type": "qdrant",
                    "collection_name": "documents",
                    "vector_size": 384,
                    "host": "localhost",
                    "port": 6333,
                    "embedding": {
                        "type": "sentence-transformer",
                        "model": "all-MiniLM-L6-v2"
                    }
                },
                "graph": {
                    "type": "neo4j",
                    "uri": "bolt://localhost:7687",
                    "username": "neo4j",
                    "password": "password"
                },
                "key_value": {
                    "type": "redis",
                    "host": "localhost",
                    "port": 6379
                }
            }
            
    Returns:
        Configured RAG service instance
    """
    config = config or {}
    
    # Create RAG service
    service = RagService()
    
    # Configure vector storage (if specified in config)
    if "vector" in config:
        await _configure_vector_storage(service, config["vector"])
    
    # Configure graph storage (if specified in config)
    if "graph" in config:
        await _configure_graph_storage(service, config["graph"])
    
    # Configure key-value storage (if specified in config)
    if "key_value" in config:
        await _configure_key_value_storage(service, config["key_value"])
    
    return service


async def _configure_vector_storage(service: RagService, config: Dict[str, Any]) -> None:
    """
    Configure vector storage for a RAG service.
    
    Args:
        service: RAG service to configure
        config: Vector storage configuration
    """
    vector_type = config.get("type", "").lower()
    
    if vector_type == "qdrant":
        if not QDRANT_AVAILABLE:
            logger.warning("Qdrant is not available. Install with 'pip install qdrant-client'")
            return
        
        # Set up embedding provider
        embedding_config = config.get("embedding", {"type": "dummy"})
        embedding_provider = create_embedding_provider(embedding_config)
        
        # Create Qdrant storage adaptor
        adaptor = QdrantStorageAdaptor(
            collection_name=config.get("collection_name", "documents"),
            vector_size=config.get("vector_size", 384),
            embedding_provider=embedding_provider,
            host=config.get("host", "localhost"),
            port=config.get("port", 6333)
        )
        
        # Register with service
        service.register_storage_adaptor(StorageType.VECTOR, adaptor)
    
    else:
        logger.warning(f"Unsupported vector storage type: {vector_type}")


async def _configure_graph_storage(service: RagService, config: Dict[str, Any]) -> None:
    """
    Configure graph storage for a RAG service.
    
    Args:
        service: RAG service to configure
        config: Graph storage configuration
    """
    # This would be implemented similar to vector storage
    # Placeholder for future implementation
    logger.info("Graph storage configuration is not implemented yet")


async def _configure_key_value_storage(service: RagService, config: Dict[str, Any]) -> None:
    """
    Configure key-value storage for a RAG service.
    
    Args:
        service: RAG service to configure
        config: Key-value storage configuration
    """
    # This would be implemented similar to vector storage
    # Placeholder for future implementation
    logger.info("Key-value storage configuration is not implemented yet") 