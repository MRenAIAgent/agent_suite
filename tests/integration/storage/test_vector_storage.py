"""Integration tests for vector storage backends."""

import pytest
import numpy as np

from agents.storage import (
    VectorStorage,
    QDRANT_AVAILABLE
)


@pytest.mark.asyncio
@pytest.mark.skipif(not QDRANT_AVAILABLE, reason="Qdrant is not available")
async def test_qdrant_vector_storage(vector_storage):
    """Test the Qdrant vector storage implementation."""
    # Store vectors with metadata
    vector1 = [0.1, 0.2, 0.3, 0.4]
    vector2 = [0.5, 0.6, 0.7, 0.8]
    metadata1 = {"text": "This is test document 1", "tags": ["test", "document"]}
    metadata2 = {"text": "This is test document 2", "tags": ["test", "other"]}
    
    # Test store_vector method
    key1 = await vector_storage.store_vector("test1", vector1, metadata1)
    key2 = await vector_storage.store_vector("test2", vector2, metadata2)
    
    assert key1 == "test1"
    assert key2 == "test2"
    
    # Test retrieve method
    result1 = await vector_storage.retrieve(key1)
    result2 = await vector_storage.retrieve(key2)
    
    assert "vector" in result1
    assert "metadata" in result1
    assert result1["metadata"]["text"] == metadata1["text"]
    assert result1["metadata"]["tags"] == metadata1["tags"]
    
    assert "vector" in result2
    assert "metadata" in result2
    assert result2["metadata"]["text"] == metadata2["text"]
    assert result2["metadata"]["tags"] == metadata2["tags"]
    
    # Test store_batch method
    batch_items = [
        ("batch1", [0.2, 0.3, 0.4, 0.5], {"text": "Batch item 1", "batch": True}),
        ("batch2", [0.3, 0.4, 0.5, 0.6], {"text": "Batch item 2", "batch": True})
    ]
    
    keys = await vector_storage.store_batch(batch_items)
    assert len(keys) == 2
    assert "batch1" in keys
    assert "batch2" in keys
    
    # Test search_by_vector method
    results = await vector_storage.search_by_vector(
        vector1,
        limit=5,
        score_threshold=0.5
    )
    
    assert len(results) > 0
    # The first result should be the original vector (highest similarity)
    assert results[0]["id"] == "test1"
    assert "score" in results[0]
    
    # Test search by vector with numpy array
    np_vector = np.array([0.1, 0.2, 0.3, 0.4])
    results = await vector_storage.search_by_vector(np_vector, limit=5)
    assert len(results) > 0
    
    # Test search_by_metadata method
    results = await vector_storage.search_by_metadata(
        {"tags": ["test"]},
        limit=5
    )
    
    assert len(results) > 0
    assert "test" in results[0]["metadata"]["tags"]
    
    # Test list method
    keys = await vector_storage.list()
    assert len(keys) >= 4  # The 4 items we added
    assert "test1" in keys
    assert "test2" in keys
    assert "batch1" in keys
    assert "batch2" in keys
    
    # Test delete method
    result = await vector_storage.delete("test1")
    assert result is True
    
    # Verify deletion
    result = await vector_storage.retrieve("test1")
    assert result is None
    
    # Test update method
    updated_metadata = {"text": "Updated document", "tags": ["updated", "test"]}
    result = await vector_storage.update(
        "test2", 
        {"vector": vector2, "metadata": updated_metadata}
    )
    assert result is True
    
    # Verify update
    result = await vector_storage.retrieve("test2")
    assert result["metadata"]["text"] == updated_metadata["text"]
    assert result["metadata"]["tags"] == updated_metadata["tags"]
    
    # Test clear method
    result = await vector_storage.clear()
    assert result is True
    
    # Verify clear
    keys = await vector_storage.list()
    assert len(keys) == 0 