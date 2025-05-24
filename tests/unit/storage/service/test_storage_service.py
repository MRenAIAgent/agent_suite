"""Unit tests for the storage service."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from agents.storage.service.storage_service import StorageService
from agents.storage.vector_storage import VectorStorage
from agents.storage.graph_storage import GraphStorage
from agents.storage.document_storage import DocumentStorage


@pytest.mark.asyncio
async def test_storage_service_initialization(storage_service):
    """Test that the storage service initializes correctly."""
    assert storage_service is not None
    assert isinstance(storage_service.backends, dict)
    assert len(storage_service.backends) == 0


@pytest.mark.asyncio
async def test_register_backend(storage_service, mock_vector_storage, 
                               mock_graph_storage, mock_document_storage):
    """Test registering backends with the service."""
    # Register a vector storage backend
    await storage_service.register_backend("vector", mock_vector_storage)
    assert "vector" in storage_service.backends
    assert storage_service.backends["vector"] == mock_vector_storage
    
    # Register a graph storage backend
    await storage_service.register_backend("graph", mock_graph_storage)
    assert "graph" in storage_service.backends
    assert storage_service.backends["graph"] == mock_graph_storage
    
    # Register a document storage backend
    await storage_service.register_backend("document", mock_document_storage)
    assert "document" in storage_service.backends
    assert storage_service.backends["document"] == mock_document_storage


@pytest.mark.asyncio
async def test_get_backend(storage_service, mock_vector_storage):
    """Test retrieving a backend by name."""
    # Register a backend
    await storage_service.register_backend("vector", mock_vector_storage)
    
    # Get the backend by name
    backend = await storage_service.get_backend("vector")
    assert backend == mock_vector_storage
    
    # Try to get a non-existent backend
    backend = await storage_service.get_backend("nonexistent")
    assert backend is None


@pytest.mark.asyncio
async def test_get_backend_by_type(storage_service, mock_vector_storage, 
                                  mock_graph_storage, mock_document_storage):
    """Test retrieving backends by type."""
    # Register multiple backends
    await storage_service.register_backend("vector1", mock_vector_storage)
    await storage_service.register_backend("vector2", mock_vector_storage)
    await storage_service.register_backend("graph", mock_graph_storage)
    await storage_service.register_backend("document", mock_document_storage)
    
    # Get backends by type
    vector_backends = await storage_service.get_backend_by_type(VectorStorage)
    assert len(vector_backends) == 2
    
    graph_backends = await storage_service.get_backend_by_type(GraphStorage)
    assert len(graph_backends) == 1
    
    document_backends = await storage_service.get_backend_by_type(DocumentStorage)
    assert len(document_backends) == 1
    
    # Check convenience methods
    vector_backends = await storage_service.get_vector_backends()
    assert len(vector_backends) == 2
    
    graph_backends = await storage_service.get_graph_backends()
    assert len(graph_backends) == 1
    
    document_backends = await storage_service.get_document_backends()
    assert len(document_backends) == 1


@pytest.mark.asyncio
async def test_remove_backend(storage_service, mock_vector_storage):
    """Test removing a backend."""
    # Register a backend
    await storage_service.register_backend("vector", mock_vector_storage)
    assert "vector" in storage_service.backends
    
    # Remove the backend
    result = await storage_service.remove_backend("vector")
    assert result is True
    assert "vector" not in storage_service.backends
    
    # Try to remove a non-existent backend
    result = await storage_service.remove_backend("nonexistent")
    assert result is False


@pytest.mark.asyncio
async def test_storage_service_with_multiple_backends(storage_service, 
                                                     mock_vector_storage,
                                                     mock_document_storage):
    """Test using multiple backends through the service."""
    # Register multiple backends
    await storage_service.register_backend("vector", mock_vector_storage)
    await storage_service.register_backend("document", mock_document_storage)
    
    # Use vector storage for similarity search
    vector_backend = await storage_service.get_backend("vector")
    query_vector = [0.1, 0.2, 0.3, 0.4]
    await vector_backend.search_by_vector(query_vector, limit=5)
    mock_vector_storage.search_by_vector_mock.assert_called_once()
    
    # Use document storage for text search
    doc_backend = await storage_service.get_backend("document")
    await doc_backend.text_search("articles", "test query")
    mock_document_storage.text_search_mock.assert_called_once()


@pytest.mark.asyncio
async def test_storage_service_close():
    """Test closing the storage service."""
    service = StorageService()
    
    # Create mock backends with close methods
    vector_backend = MagicMock()
    vector_backend.close = AsyncMock()
    
    graph_backend = MagicMock()
    graph_backend.close = AsyncMock()
    
    document_backend = MagicMock()
    document_backend.close = AsyncMock()
    
    # Register backends
    await service.register_backend("vector", vector_backend)
    await service.register_backend("graph", graph_backend)
    await service.register_backend("document", document_backend)
    
    # Close the service
    await service.close()
    
    # Check that close was called on each backend
    vector_backend.close.assert_called_once()
    graph_backend.close.assert_called_once()
    document_backend.close.assert_called_once()
    
    # Test with backend that doesn't have close method
    service = StorageService()
    backend_without_close = MagicMock()
    
    # Should not raise an error
    await service.register_backend("test", backend_without_close)
    await service.close() 