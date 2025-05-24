"""Unit tests for the storage factory functions."""

import pytest
from unittest.mock import MagicMock, patch

from agents.storage.factory import (
    StorageFactory,
    create_vector_storage,
    create_graph_storage,
    create_document_storage,
    create_storage_service
)
from agents.storage.vector_storage import VectorStorage
from agents.storage.graph_storage import GraphStorage
from agents.storage.document_storage import DocumentStorage
from agents.storage.service.storage_service import StorageService


@pytest.mark.parametrize("backend_type", ["qdrant"])
def test_create_vector_storage(backend_type):
    """Test the create_vector_storage factory function."""
    with patch("agents.storage.factory.QdrantStorage") as mock_qdrant:
        mock_qdrant.return_value = MagicMock(spec=VectorStorage)
        
        # Default configuration
        storage = create_vector_storage(backend_type=backend_type)
        assert storage is not None
        assert isinstance(storage, MagicMock)
        
        if backend_type == "qdrant":
            mock_qdrant.assert_called_once()
        
        # Custom configuration
        config = {"collection_name": "test", "vector_size": 768}
        storage = create_vector_storage(backend_type=backend_type, config=config)
        assert storage is not None
        assert isinstance(storage, MagicMock)
        
        if backend_type == "qdrant":
            mock_qdrant.assert_called_with(**config)


@pytest.mark.parametrize("backend_type", ["neo4j"])
def test_create_graph_storage(backend_type):
    """Test the create_graph_storage factory function."""
    with patch("agents.storage.factory.Neo4jStorage") as mock_neo4j:
        mock_neo4j.return_value = MagicMock(spec=GraphStorage)
        
        # Default configuration
        storage = create_graph_storage(backend_type=backend_type)
        assert storage is not None
        assert isinstance(storage, MagicMock)
        
        if backend_type == "neo4j":
            mock_neo4j.assert_called_once()
        
        # Custom configuration
        config = {"uri": "bolt://localhost:7687", "username": "neo4j", "password": "password"}
        storage = create_graph_storage(backend_type=backend_type, config=config)
        assert storage is not None
        assert isinstance(storage, MagicMock)
        
        if backend_type == "neo4j":
            mock_neo4j.assert_called_with(**config)


@pytest.mark.parametrize("backend_type", ["mongodb", "elasticsearch"])
def test_create_document_storage(backend_type):
    """Test the create_document_storage factory function."""
    with patch("agents.storage.factory.MongoDBStorage") as mock_mongodb, \
         patch("agents.storage.factory.ElasticsearchStorage") as mock_elasticsearch:
        
        if backend_type == "mongodb":
            mock_mongodb.return_value = MagicMock(spec=DocumentStorage)
        else:
            mock_elasticsearch.return_value = MagicMock(spec=DocumentStorage)
        
        # Default configuration
        storage = create_document_storage(backend_type=backend_type)
        assert storage is not None
        assert isinstance(storage, MagicMock)
        
        if backend_type == "mongodb":
            mock_mongodb.assert_called_once()
        elif backend_type == "elasticsearch":
            mock_elasticsearch.assert_called_once()
        
        # Custom configuration
        if backend_type == "mongodb":
            config = {"uri": "mongodb://localhost:27017", "database": "test_db"}
            storage = create_document_storage(backend_type=backend_type, config=config)
            assert storage is not None
            assert isinstance(storage, MagicMock)
            mock_mongodb.assert_called_with(**config)
        elif backend_type == "elasticsearch":
            config = {"hosts": "http://localhost:9200", "namespace": "test"}
            storage = create_document_storage(backend_type=backend_type, config=config)
            assert storage is not None
            assert isinstance(storage, MagicMock)
            mock_elasticsearch.assert_called_with(**config)


def test_create_storage_service():
    """Test the create_storage_service factory function."""
    with patch("agents.storage.factory.StorageService") as mock_service, \
         patch("agents.storage.factory.StorageFactory.create_vector_storage") as mock_create_vector, \
         patch("agents.storage.factory.StorageFactory.create_graph_storage") as mock_create_graph, \
         patch("agents.storage.factory.StorageFactory.create_document_storage") as mock_create_document:
        
        mock_service.return_value = MagicMock(spec=StorageService)
        mock_service.return_value.register_backend = MagicMock()
        
        mock_create_vector.return_value = MagicMock(spec=VectorStorage)
        mock_create_graph.return_value = MagicMock(spec=GraphStorage)
        mock_create_document.return_value = MagicMock(spec=DocumentStorage)
        
        # Create service with multiple backends
        backends = {
            "vector": {
                "type": "qdrant",
                "config": {"collection_name": "test"}
            },
            "graph": {
                "type": "neo4j",
                "config": {"uri": "bolt://localhost:7687"}
            },
            "document": {
                "type": "mongodb",
                "config": {"database": "test"}
            }
        }
        
        service = create_storage_service(backends)
        assert service is not None
        assert isinstance(service, MagicMock)
        
        mock_service.assert_called_once()
        assert mock_create_vector.call_count == 1
        assert mock_create_graph.call_count == 1
        assert mock_create_document.call_count == 1
        
        # Register backend should be called for each backend
        assert mock_service.return_value.register_backend.call_count == 3


def test_storage_factory_methods():
    """Test the storage factory class methods."""
    with patch("agents.storage.factory.QdrantStorage") as mock_qdrant, \
         patch("agents.storage.factory.Neo4jStorage") as mock_neo4j, \
         patch("agents.storage.factory.MongoDBStorage") as mock_mongodb, \
         patch("agents.storage.factory.ElasticsearchStorage") as mock_elasticsearch, \
         patch("agents.storage.factory.StorageService") as mock_service:
        
        mock_qdrant.return_value = MagicMock(spec=VectorStorage)
        mock_neo4j.return_value = MagicMock(spec=GraphStorage)
        mock_mongodb.return_value = MagicMock(spec=DocumentStorage)
        mock_elasticsearch.return_value = MagicMock(spec=DocumentStorage)
        mock_service.return_value = MagicMock(spec=StorageService)
        
        # Test vector storage creation
        storage = StorageFactory.create_vector_storage("qdrant", {"collection_name": "test"})
        assert storage is not None
        mock_qdrant.assert_called_once_with(collection_name="test")
        
        # Test graph storage creation
        storage = StorageFactory.create_graph_storage("neo4j", {"uri": "bolt://localhost:7687"})
        assert storage is not None
        mock_neo4j.assert_called_once_with(uri="bolt://localhost:7687")
        
        # Test document storage creation
        storage = StorageFactory.create_document_storage("mongodb", {"database": "test"})
        assert storage is not None
        mock_mongodb.assert_called_once_with(database="test")
        
        storage = StorageFactory.create_document_storage("elasticsearch", {"hosts": "localhost"})
        assert storage is not None
        mock_elasticsearch.assert_called_once_with(hosts="localhost")


def test_factory_invalid_backend():
    """Test factory behavior with invalid backend types."""
    # Test with invalid vector storage backend
    with pytest.raises(ValueError, match="Unsupported vector storage backend"):
        StorageFactory.create_vector_storage("invalid")
    
    # Test with invalid graph storage backend
    with pytest.raises(ValueError, match="Unsupported graph storage backend"):
        StorageFactory.create_graph_storage("invalid")
    
    # Test with invalid document storage backend
    with pytest.raises(ValueError, match="Unsupported document storage backend"):
        StorageFactory.create_document_storage("invalid") 