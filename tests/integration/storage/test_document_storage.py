"""Integration tests for document storage backends."""

import pytest
import uuid

from agents.storage import (
    DocumentStorage,
    MONGODB_AVAILABLE,
    ELASTICSEARCH_AVAILABLE
)


@pytest.mark.asyncio
@pytest.mark.skipif(not MONGODB_AVAILABLE, reason="MongoDB is not available")
async def test_mongodb_document_storage(mongodb_storage):
    """Test the MongoDB document storage implementation."""
    # Create test collection name with unique identifier
    collection = f"test_collection_{uuid.uuid4().hex[:8]}"
    
    # Test creating an index
    index_name = await mongodb_storage.create_index(
        collection,
        {
            "title": "text",
            "content": "text",
            "author": "keyword",
            "tags": "keyword"
        }
    )
    assert index_name is not None
    
    # Test insert_document method
    doc1 = {
        "title": "Test Document 1",
        "content": "This is a test document with some sample content for testing.",
        "author": "Test Author",
        "tags": ["test", "document", "sample"],
        "published": True,
        "views": 100
    }
    
    doc1_id = await mongodb_storage.insert_document(collection, doc1)
    assert doc1_id is not None
    
    # Test store method (which internally uses insert_document)
    doc2 = {
        "title": "Test Document 2",
        "content": "Another test document with different content.",
        "author": "Another Author",
        "tags": ["test", "different"],
        "published": True,
        "views": 50
    }
    
    doc2_id = await mongodb_storage.store("test_doc2", doc2, collection=collection)
    assert doc2_id is not None
    
    # Test retrieve method
    result = await mongodb_storage.retrieve(doc1_id, collection=collection)
    assert result is not None
    assert result["title"] == doc1["title"]
    assert result["content"] == doc1["content"]
    
    # Test batch insertion
    docs = [
        {
            "title": "Batch Doc 1",
            "content": "First batch document content.",
            "tags": ["batch", "first"]
        },
        {
            "title": "Batch Doc 2",
            "content": "Second batch document content.",
            "tags": ["batch", "second"]
        }
    ]
    
    batch_ids = await mongodb_storage.insert_documents(collection, docs)
    assert len(batch_ids) == 2
    
    # Test find_documents method
    results = await mongodb_storage.find_documents(collection, {"tags": "test"})
    assert len(results) == 2  # doc1 and doc2 have 'test' tag
    
    # Test find_documents with projection
    results = await mongodb_storage.find_documents(
        collection,
        {"tags": "test"},
        projection={"title": 1, "author": 1}
    )
    assert len(results) == 2
    assert "title" in results[0]
    assert "author" in results[0]
    assert "content" not in results[0]
    
    # Test find_documents with sort
    results = await mongodb_storage.find_documents(
        collection,
        {"tags": "test"},
        sort={"views": -1}  # Sort by views descending
    )
    assert len(results) == 2
    assert results[0]["views"] > results[1]["views"]
    
    # Test text_search method
    results = await mongodb_storage.text_search(
        collection,
        "sample content",
        fields=["title", "content"]
    )
    assert len(results) > 0
    assert "score" in results[0]  # Text search should include relevance score
    
    # Test update_document method
    update = {"$set": {"views": 150, "updated": True}}
    result = await mongodb_storage.update_document(
        collection,
        doc1_id,
        update
    )
    assert result is True
    
    # Verify update
    result = await mongodb_storage.retrieve(doc1_id, collection=collection)
    assert result["views"] == 150
    assert result["updated"] is True
    
    # Test delete_documents method
    result = await mongodb_storage.delete_documents(collection, {"tags": "batch"})
    assert result == 2  # Should delete the 2 batch documents
    
    # Test delete method
    result = await mongodb_storage.delete(doc1_id, collection=collection)
    assert result is True
    
    # Verify deletion
    result = await mongodb_storage.retrieve(doc1_id, collection=collection)
    assert result is None
    
    # Test aggregate method
    pipeline = [
        {"$match": {"published": True}},
        {"$group": {"_id": "$author", "total_views": {"$sum": "$views"}}}
    ]
    
    results = await mongodb_storage.aggregate(collection, pipeline)
    assert len(results) == 1  # Only one author left (doc2)
    
    # Test list method
    ids = await mongodb_storage.list(collection=collection)
    assert len(ids) == 1  # Only doc2 should remain
    assert doc2_id in ids
    
    # Test clear method (drops the collection)
    result = await mongodb_storage.clear(collection=collection)
    assert result is True
    
    # Verify clear
    results = await mongodb_storage.find_documents(collection, {})
    assert len(results) == 0


@pytest.mark.asyncio
@pytest.mark.skipif(not ELASTICSEARCH_AVAILABLE, reason="Elasticsearch is not available")
async def test_elasticsearch_document_storage(elasticsearch_storage):
    """Test the Elasticsearch document storage implementation."""
    # Create test index name with unique identifier
    index = f"test_index_{uuid.uuid4().hex[:8]}"
    
    # Wait for Elasticsearch to be ready
    await elasticsearch_storage.wait_for_ready()
    
    # Create mappings/index
    mappings = {
        "properties": {
            "title": {"type": "text"},
            "content": {"type": "text"},
            "author": {"type": "keyword"},
            "tags": {"type": "keyword"},
            "published": {"type": "boolean"},
            "views": {"type": "integer"}
        }
    }
    
    index_name = await elasticsearch_storage.create_index(index, mappings)
    assert index_name is not None
    
    # Test insert_document method
    doc1 = {
        "title": "Elasticsearch Test Document 1",
        "content": "This is a test document for Elasticsearch with some sample content.",
        "author": "ES Author",
        "tags": ["elasticsearch", "test", "document"],
        "published": True,
        "views": 200
    }
    
    doc1_id = await elasticsearch_storage.insert_document(index, doc1)
    assert doc1_id is not None
    
    # Force refresh to make documents immediately available for search
    await elasticsearch_storage._client.indices.refresh(index=index)
    
    # Test store method (might be similar to insert_document depending on implementation)
    doc2 = {
        "title": "Elasticsearch Test Document 2",
        "content": "Another test document for searching in Elasticsearch.",
        "author": "Another ES Author",
        "tags": ["elasticsearch", "search"],
        "published": True,
        "views": 100
    }
    
    doc2_id = await elasticsearch_storage.store("es_doc2", doc2, index=index)
    assert doc2_id is not None
    
    # Force refresh
    await elasticsearch_storage._client.indices.refresh(index=index)
    
    # Test retrieve method
    result = await elasticsearch_storage.retrieve(doc1_id, index=index)
    assert result is not None
    assert result["title"] == doc1["title"]
    assert result["content"] == doc1["content"]
    
    # Test batch insertion
    docs = [
        {
            "title": "ES Batch Doc 1",
            "content": "First batch document for Elasticsearch.",
            "tags": ["batch", "first", "elasticsearch"]
        },
        {
            "title": "ES Batch Doc 2",
            "content": "Second batch document for Elasticsearch.",
            "tags": ["batch", "second", "elasticsearch"]
        }
    ]
    
    batch_ids = await elasticsearch_storage.insert_documents(index, docs)
    assert len(batch_ids) == 2
    
    # Force refresh
    await elasticsearch_storage._client.indices.refresh(index=index)
    
    # Test find_documents method (term query in Elasticsearch)
    results = await elasticsearch_storage.find_documents(
        index, {"tags": "elasticsearch"}
    )
    assert len(results) == 4  # All docs have 'elasticsearch' tag
    
    # Test text_search method - specialized for Elasticsearch
    results = await elasticsearch_storage.text_search(
        index,
        "sample content",
        fields=["title", "content"]
    )
    assert len(results) > 0
    assert "score" in results[0]  # Text search includes score
    
    # Test update_document method
    update = {"doc": {"views": 250, "updated": True}}
    result = await elasticsearch_storage.update_document(
        index,
        doc1_id,
        update
    )
    assert result is True
    
    # Force refresh
    await elasticsearch_storage._client.indices.refresh(index=index)
    
    # Verify update
    result = await elasticsearch_storage.retrieve(doc1_id, index=index)
    assert result["views"] == 250
    assert result["updated"] is True
    
    # Test delete_documents method (with query)
    result = await elasticsearch_storage.delete_documents(
        index, {"terms": {"tags": ["batch"]}}
    )
    assert result >= 2  # Should delete the 2 batch documents
    
    # Force refresh
    await elasticsearch_storage._client.indices.refresh(index=index)
    
    # Test delete method
    result = await elasticsearch_storage.delete(doc1_id, index=index)
    assert result is True
    
    # Force refresh
    await elasticsearch_storage._client.indices.refresh(index=index)
    
    # Verify deletion
    result = await elasticsearch_storage.retrieve(doc1_id, index=index)
    assert result is None
    
    # Test list method (might return document IDs)
    ids = await elasticsearch_storage.list(index=index)
    assert len(ids) == 1  # Only doc2 should remain
    assert doc2_id in ids
    
    # Test clear method (deletes the index)
    result = await elasticsearch_storage.clear(index=index)
    assert result is True
    
    # Verify clear (index should no longer exist)
    with pytest.raises(Exception):
        await elasticsearch_storage.find_documents(index, {}) 