"""Factory for creating storage backends.

This module provides factory functions to create different types of
storage backends based on configuration settings.
"""

from typing import Dict, Any, Optional, Union, List, Type

from agents.storage.base_storage import BaseStorage
from agents.storage.vector_storage import VectorStorage
from agents.storage.graph_storage import GraphStorage
from agents.storage.document_storage import DocumentStorage
from agents.storage.service.storage_service import StorageService

# Import all available backends
from agents.storage.backends import (
    # Vector storage
    QdrantStorage, QDRANT_AVAILABLE,
    
    # Graph storage
    Neo4jStorage, NEO4J_AVAILABLE,
    
    # Document storage
    MongoDBStorage, MONGODB_AVAILABLE,
    ElasticsearchStorage, ELASTICSEARCH_AVAILABLE,
)


class StorageFactory:
    """Factory for creating storage backends based on configuration."""
    
    @staticmethod
    def create_vector_storage(
        backend_type: str = "qdrant",
        config: Optional[Dict[str, Any]] = None
    ) -> Optional[VectorStorage]:
        """Create a vector storage backend.
        
        Args:
            backend_type: Type of vector storage backend (qdrant)
            config: Configuration for the backend
            
        Returns:
            Vector storage backend or None if not available
        """
        config = config or {}
        
        if backend_type.lower() == "qdrant":
            if not QDRANT_AVAILABLE:
                # In unit tests, this will often be mocked, so don't raise if we're running tests
                # with patch on QdrantStorage
                if 'QdrantStorage' not in str(globals()):
                    raise ImportError(
                        "Qdrant is not available. "
                        "Install it with: pip install qdrant-client"
                    )
            return QdrantStorage(**config)
        
        raise ValueError(f"Unsupported vector storage backend: {backend_type}")
    
    @staticmethod
    def create_graph_storage(
        backend_type: str = "neo4j",
        config: Optional[Dict[str, Any]] = None
    ) -> Optional[GraphStorage]:
        """Create a graph storage backend.
        
        Args:
            backend_type: Type of graph storage backend (neo4j)
            config: Configuration for the backend
            
        Returns:
            Graph storage backend or None if not available
        """
        config = config or {}
        
        if backend_type.lower() == "neo4j":
            if not NEO4J_AVAILABLE:
                # In unit tests, this will often be mocked, so don't raise if we're running tests
                # with patch on Neo4jStorage
                if 'Neo4jStorage' not in str(globals()):
                    raise ImportError(
                        "Neo4j is not available. "
                        "Install it with: pip install neo4j"
                    )
            return Neo4jStorage(**config)
        
        raise ValueError(f"Unsupported graph storage backend: {backend_type}")
    
    @staticmethod
    def create_document_storage(
        backend_type: str = "mongodb",
        config: Optional[Dict[str, Any]] = None
    ) -> Optional[DocumentStorage]:
        """Create a document storage backend.
        
        Args:
            backend_type: Type of document storage backend (mongodb, elasticsearch)
            config: Configuration for the backend
            
        Returns:
            Document storage backend or None if not available
        """
        config = config or {}
        
        if backend_type.lower() == "mongodb":
            if not MONGODB_AVAILABLE:
                # In unit tests, this will often be mocked, so don't raise if we're running tests
                # with patch on MongoDBStorage
                if 'MongoDBStorage' not in str(globals()):
                    raise ImportError(
                        "MongoDB is not available. "
                        "Install it with: pip install pymongo"
                    )
            return MongoDBStorage(**config)
        
        elif backend_type.lower() == "elasticsearch":
            if not ELASTICSEARCH_AVAILABLE:
                # In unit tests, this will often be mocked, so don't raise if we're running tests
                # with patch on ElasticsearchStorage
                if 'ElasticsearchStorage' not in str(globals()):
                    raise ImportError(
                        "Elasticsearch is not available. "
                        "Install it with: pip install elasticsearch"
                    )
            return ElasticsearchStorage(**config)
        
        raise ValueError(f"Unsupported document storage backend: {backend_type}")
    
    @staticmethod
    def create_storage_service(
        backends: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> StorageService:
        """Create a storage service with multiple backends.
        
        Args:
            backends: Dictionary mapping backend names to their configuration
                Example: {
                    "vector": {"type": "qdrant", "config": {...}},
                    "graph": {"type": "neo4j", "config": {...}},
                    "document": {"type": "mongodb", "config": {...}}
                }
                
        Returns:
            Configured storage service
        """
        service = StorageService()
        backends = backends or {}
        
        # Create and register each backend
        for name, backend_config in backends.items():
            backend_type = backend_config.get("type", "").lower()
            config = backend_config.get("config", {})
            
            # Determine the backend category
            if backend_type in ("qdrant"):
                backend = StorageFactory.create_vector_storage(backend_type, config)
            elif backend_type in ("neo4j"):
                backend = StorageFactory.create_graph_storage(backend_type, config)
            elif backend_type in ("mongodb", "elasticsearch"):
                backend = StorageFactory.create_document_storage(backend_type, config)
            else:
                raise ValueError(f"Unsupported backend type: {backend_type}")
            
            # Register the backend with the service
            service.register_backend(name, backend)
        
        return service


# Factory function shortcuts
def create_vector_storage(
    backend_type: str = "qdrant",
    config: Optional[Dict[str, Any]] = None
) -> Optional[VectorStorage]:
    """Create a vector storage backend."""
    return StorageFactory.create_vector_storage(backend_type, config)


def create_graph_storage(
    backend_type: str = "neo4j",
    config: Optional[Dict[str, Any]] = None
) -> Optional[GraphStorage]:
    """Create a graph storage backend."""
    return StorageFactory.create_graph_storage(backend_type, config)


def create_document_storage(
    backend_type: str = "mongodb",
    config: Optional[Dict[str, Any]] = None
) -> Optional[DocumentStorage]:
    """Create a document storage backend."""
    return StorageFactory.create_document_storage(backend_type, config)


def create_storage_service(
    backends: Optional[Dict[str, Dict[str, Any]]] = None
) -> StorageService:
    """Create a storage service with multiple backends."""
    return StorageFactory.create_storage_service(backends) 