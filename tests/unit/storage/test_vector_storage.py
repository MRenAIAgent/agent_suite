"""Unit tests for the VectorStorage interface."""

import pytest
import numpy as np
from unittest.mock import AsyncMock, patch

from agents.storage.vector_storage import VectorStorage


@pytest.mark.asyncio
async def test_vector_storage_base_methods(mock_vector_storage):
    """Test that the vector storage implements BaseStorage methods."""
    # Test inherited BaseStorage methods
    key = "test-key"
    data = {"vector": [0.1, 0.2], "metadata": {"text": "test"}}
    
    # Test store method
    result = await mock_vector_storage.store(key, data)
    assert result == "mock-id"
    mock_vector_storage.store_mock.assert_called_once_with(key, data)
    
    # Test retrieve method
    result = await mock_vector_storage.retrieve(key)
    assert result["vector"] == [0.1, 0.2]
    assert result["metadata"]["test"] == "data"
    mock_vector_storage.retrieve_mock.assert_called_once_with(key)
    
    # Test delete method
    result = await mock_vector_storage.delete(key)
    assert result is True
    mock_vector_storage.delete_mock.assert_called_once_with(key)
    
    # Test update method
    new_data = {"vector": [0.3, 0.4], "metadata": {"updated": "value"}}
    result = await mock_vector_storage.update(key, new_data)
    assert result is True
    mock_vector_storage.update_mock.assert_called_once_with(key, new_data)
    
    # Test list method
    pattern = "test-*"
    result = await mock_vector_storage.list(pattern)
    assert result == ["id1", "id2"]
    mock_vector_storage.list_mock.assert_called_once_with(pattern)
    
    # Test clear method
    result = await mock_vector_storage.clear()
    assert result is True
    mock_vector_storage.clear_mock.assert_called_once()


@pytest.mark.asyncio
async def test_vector_storage_methods(mock_vector_storage):
    """Test vector-specific methods."""
    # Test store_vector method
    key = "test-vector"
    vector = [0.1, 0.2, 0.3, 0.4]
    metadata = {"text": "test document"}
    result = await mock_vector_storage.store_vector(key, vector, metadata)
    assert result == "mock-vector-id"
    mock_vector_storage.store_vector_mock.assert_called_once_with(key, vector, metadata)
    
    # Test store_batch method
    items = [
        ("item1", [0.1, 0.2], {"text": "doc1"}),
        ("item2", [0.3, 0.4], {"text": "doc2"})
    ]
    result = await mock_vector_storage.store_batch(items)
    assert result == ["id1", "id2"]
    mock_vector_storage.store_batch_mock.assert_called_once_with(items)


@pytest.mark.asyncio
async def test_vector_search_methods(mock_vector_storage):
    """Test vector search methods."""
    # Test search_by_vector method
    query_vector = [0.1, 0.2, 0.3, 0.4]
    limit = 5
    score_threshold = 0.8
    filter_metadata = {"tag": "test"}
    
    result = await mock_vector_storage.search_by_vector(
        query_vector, 
        limit=limit,
        score_threshold=score_threshold,
        filter_metadata=filter_metadata
    )
    
    assert len(result) == 2
    assert result[0]["id"] == "id1"
    assert result[0]["score"] == 0.95
    mock_vector_storage.search_by_vector_mock.assert_called_once_with(
        query_vector, limit, score_threshold, filter_metadata
    )
    
    # Test search_by_metadata method
    result = await mock_vector_storage.search_by_metadata(filter_metadata, limit=limit)
    assert len(result) == 2
    assert result[0]["metadata"]["tag"] == "test"
    mock_vector_storage.search_by_metadata_mock.assert_called_once_with(filter_metadata, limit)


@pytest.mark.asyncio
async def test_vector_storage_numpy_compatibility(mock_vector_storage):
    """Test that vector storage works with numpy arrays."""
    # Test store_vector method with numpy array
    key = "test-numpy"
    vector = np.array([0.1, 0.2, 0.3, 0.4])
    metadata = {"text": "numpy test"}
    
    await mock_vector_storage.store_vector(key, vector, metadata)
    # Vector should be converted to list in the storage call
    mock_vector_storage.store_vector_mock.assert_called_with(key, vector, metadata)
    
    # Test search_by_vector with numpy array
    query_vector = np.array([0.5, 0.6, 0.7, 0.8])
    await mock_vector_storage.search_by_vector(query_vector, limit=5)
    mock_vector_storage.search_by_vector_mock.assert_called_with(query_vector, 5, None, None) 