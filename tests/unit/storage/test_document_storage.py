"""Unit tests for the DocumentStorage interface."""

import pytest
from unittest.mock import AsyncMock, patch

from agents.storage.document_storage import DocumentStorage


@pytest.mark.asyncio
async def test_document_storage_base_methods(mock_document_storage):
    """Test that the document storage implements BaseStorage methods."""
    # Test inherited BaseStorage methods
    key = "test-key"
    data = {"id": "doc1", "title": "Test Document", "content": "This is a test"}
    
    # Test store method
    result = await mock_document_storage.store(key, data)
    assert result == "mock-id"
    mock_document_storage.store_mock.assert_called_once_with(key, data)
    
    # Test retrieve method
    result = await mock_document_storage.retrieve(key)
    assert result["id"] == "doc1"
    assert result["title"] == "Test"
    mock_document_storage.retrieve_mock.assert_called_once_with(key)
    
    # Test delete method
    result = await mock_document_storage.delete(key)
    assert result is True
    mock_document_storage.delete_mock.assert_called_once_with(key)
    
    # Test update method
    new_data = {"title": "Updated Title"}
    result = await mock_document_storage.update(key, new_data)
    assert result is True
    mock_document_storage.update_mock.assert_called_once_with(key, new_data)
    
    # Test list method
    pattern = "test-*"
    result = await mock_document_storage.list(pattern)
    assert result == ["id1", "id2"]
    mock_document_storage.list_mock.assert_called_once_with(pattern)
    
    # Test clear method
    result = await mock_document_storage.clear()
    assert result is True
    mock_document_storage.clear_mock.assert_called_once()


@pytest.mark.asyncio
async def test_document_storage_index_methods(mock_document_storage):
    """Test document index methods."""
    # Test create_index method
    collection = "articles"
    fields = {
        "title": "text",
        "content": "text",
        "author": "keyword",
        "tags": "keyword"
    }
    index_name = "articles_text_index"
    
    result = await mock_document_storage.create_index(collection, fields, index_name)
    assert result == "index-name"
    mock_document_storage.create_index_mock.assert_called_once_with(
        collection, fields, index_name
    )


@pytest.mark.asyncio
async def test_document_storage_document_methods(mock_document_storage):
    """Test document-specific methods."""
    # Test insert_document method
    collection = "articles"
    document = {
        "id": "article1",
        "title": "Test Article",
        "content": "This is test content",
        "author": "Test Author"
    }
    
    result = await mock_document_storage.insert_document(collection, document)
    assert result == "doc-id"
    mock_document_storage.insert_document_mock.assert_called_once_with(collection, document)
    
    # Test insert_documents method
    documents = [
        {"id": "article1", "title": "Article 1"},
        {"id": "article2", "title": "Article 2"}
    ]
    
    result = await mock_document_storage.insert_documents(collection, documents)
    assert result == ["doc1", "doc2"]
    mock_document_storage.insert_documents_mock.assert_called_once_with(collection, documents)
    
    # Test find_documents method
    query = {"author": "Test Author"}
    projection = {"_id": 0, "title": 1, "author": 1}
    sort = {"publishedAt": -1}
    limit = 10
    skip = 0
    
    result = await mock_document_storage.find_documents(
        collection, query, projection, sort, limit, skip
    )
    assert len(result) == 2
    assert result[0]["id"] == "doc1"
    assert result[0]["title"] == "Test 1"
    mock_document_storage.find_documents_mock.assert_called_once_with(
        collection, query, projection, sort, limit, skip
    )
    
    # Test update_document method
    document_id = "doc1"
    update = {"$set": {"title": "Updated Title"}}
    upsert = True
    
    result = await mock_document_storage.update_document(
        collection, document_id, update, upsert
    )
    assert result is True
    mock_document_storage.update_document_mock.assert_called_once_with(
        collection, document_id, update, upsert
    )
    
    # Test delete_documents method
    query = {"status": "archived"}
    
    result = await mock_document_storage.delete_documents(collection, query)
    assert result == 2
    mock_document_storage.delete_documents_mock.assert_called_once_with(collection, query)


@pytest.mark.asyncio
async def test_document_storage_query_methods(mock_document_storage):
    """Test document query methods."""
    collection = "articles"
    
    # Test text_search method
    query = "test content"
    fields = ["title", "content"]
    limit = 5
    
    result = await mock_document_storage.text_search(collection, query, fields, limit)
    assert len(result) == 2
    assert result[0]["id"] == "doc1"
    assert result[0]["score"] == 0.9
    mock_document_storage.text_search_mock.assert_called_once_with(
        collection, query, fields, limit
    )
    
    # Test aggregate method
    pipeline = [
        {"$match": {"status": "published"}},
        {"$group": {"_id": "$author", "count": {"$sum": 1}}}
    ]
    
    result = await mock_document_storage.aggregate(collection, pipeline)
    assert len(result) == 1
    assert result[0]["_id"] == "group1"
    assert result[0]["count"] == 2
    mock_document_storage.aggregate_mock.assert_called_once_with(collection, pipeline)


@pytest.mark.asyncio
async def test_document_storage_with_kwargs(mock_document_storage):
    """Test that document storage methods properly handle kwargs."""
    # Test with extra parameters
    await mock_document_storage.find_documents(
        "articles", 
        {"status": "active"}, 
        explain=True,
        read_preference="secondaryPreferred"
    )
    mock_document_storage.find_documents_mock.assert_called_with(
        "articles", 
        {"status": "active"}, 
        None, None, 100, 0,
        explain=True,
        read_preference="secondaryPreferred"
    )
    
    await mock_document_storage.text_search(
        "articles", 
        "test", 
        highlight=True,
        min_score=0.5
    )
    mock_document_storage.text_search_mock.assert_called_with(
        "articles", 
        "test", 
        None, 10,
        highlight=True,
        min_score=0.5
    ) 