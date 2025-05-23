"""Storage service module for agent memory management.

This module provides a unified storage service with support for
multiple backends including vector, graph, and document storage.
"""

from agents.storage.base_storage import BaseStorage
from agents.storage.vector_storage import VectorStorage
from agents.storage.graph_storage import GraphStorage
from agents.storage.document_storage import DocumentStorage
from agents.storage.service.storage_service import StorageService

# Export all backends
from agents.storage.backends import (
    # Vector storage
    QdrantStorage,
    QDRANT_AVAILABLE,
    
    # Graph storage
    Neo4jStorage,
    NEO4J_AVAILABLE,
    
    # Document storage
    MongoDBStorage,
    MONGODB_AVAILABLE,
    ElasticsearchStorage,
    ELASTICSEARCH_AVAILABLE,
)

# Export factory functions
from agents.storage.factory import (
    create_vector_storage,
    create_graph_storage,
    create_document_storage,
    create_storage_service,
    StorageFactory
)

# Export public interfaces
__all__ = [
    # Base interfaces
    "BaseStorage",
    "VectorStorage",
    "GraphStorage",
    "DocumentStorage",
    
    # Service
    "StorageService",
    
    # Vector storage backends
    "QdrantStorage",
    "QDRANT_AVAILABLE",
    
    # Graph storage backends
    "Neo4jStorage",
    "NEO4J_AVAILABLE",
    
    # Document storage backends
    "MongoDBStorage",
    "MONGODB_AVAILABLE",
    "ElasticsearchStorage",
    "ELASTICSEARCH_AVAILABLE",
    
    # Factory functions
    "create_vector_storage",
    "create_graph_storage",
    "create_document_storage",
    "create_storage_service",
    "StorageFactory",
] 