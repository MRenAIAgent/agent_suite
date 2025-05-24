"""Unit tests for the storage backend implementations."""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch, call

from agents.storage.backends.qdrant_storage import QdrantStorage
from agents.storage.backends.neo4j_storage import Neo4jStorage
from agents.storage.backends.mongodb_storage import MongoDBStorage
from agents.storage.backends.elasticsearch_storage import ElasticsearchStorage


@pytest.mark.asyncio
async def test_qdrant_storage_init():
    """Test initializing QdrantStorage with different parameters."""
    # Test with default parameters
    with patch("agents.storage.backends.qdrant_storage.QdrantClient") as mock_client:
        storage = QdrantStorage()
        mock_client.assert_called_once()
        assert storage.collection_name == "vectors"
        assert storage.vector_size == 384
    
    # Test with custom parameters
    with patch("agents.storage.backends.qdrant_storage.QdrantClient") as mock_client:
        storage = QdrantStorage(
            collection_name="test_collection",
            vector_size=768,
            url="http://localhost:6333",
            prefer_grpc=True
        )
        mock_client.assert_called_once_with(url="http://localhost:6333", prefer_grpc=True)
        assert storage.collection_name == "test_collection"
        assert storage.vector_size == 768
        assert storage.client is mock_client.return_value


@pytest.mark.asyncio
async def test_neo4j_storage_init():
    """Test initializing Neo4jStorage with different parameters."""
    # Test with default parameters
    with patch("agents.storage.backends.neo4j_storage.GraphDatabase") as mock_db:
        mock_driver = AsyncMock()
        mock_db.driver.return_value = mock_driver
        
        storage = Neo4jStorage()
        mock_db.driver.assert_called_once()
        assert storage.driver is mock_driver
    
    # Test with custom parameters
    with patch("agents.storage.backends.neo4j_storage.GraphDatabase") as mock_db:
        mock_driver = AsyncMock()
        mock_db.driver.return_value = mock_driver
        
        storage = Neo4jStorage(
            uri="bolt://localhost:7687",
            username="neo4j",
            password="password",
            database="testdb"
        )
        mock_db.driver.assert_called_once_with(
            "bolt://localhost:7687",
            auth=("neo4j", "password")
        )
        assert storage.driver is mock_driver
        assert storage.database == "testdb"


@pytest.mark.asyncio
async def test_mongodb_storage_init():
    """Test initializing MongoDBStorage with different parameters."""
    # Test with default parameters
    with patch("agents.storage.backends.mongodb_storage.MongoClient") as mock_client:
        mock_db = AsyncMock()
        mock_client.return_value.__getitem__.return_value = mock_db
        
        storage = MongoDBStorage()
        mock_client.assert_called_once()
        assert storage.client is mock_client.return_value
        assert storage.db is mock_db
    
    # Test with custom parameters
    with patch("agents.storage.backends.mongodb_storage.MongoClient") as mock_client:
        mock_db = AsyncMock()
        mock_client.return_value.__getitem__.return_value = mock_db
        
        storage = MongoDBStorage(
            uri="mongodb://localhost:27017",
            database="testdb",
            namespace="test"
        )
        mock_client.assert_called_once_with("mongodb://localhost:27017")
        assert storage.client is mock_client.return_value
        mock_client.return_value.__getitem__.assert_called_once_with("testdb")
        assert storage.db is mock_db
        assert storage.namespace == "test"


@pytest.mark.asyncio
async def test_elasticsearch_storage_init():
    """Test initializing ElasticsearchStorage with different parameters."""
    # Test with default parameters
    with patch("agents.storage.backends.elasticsearch_storage.Elasticsearch") as mock_client:
        storage = ElasticsearchStorage()
        mock_client.assert_called_once()
        assert storage.client is mock_client.return_value
        assert storage.namespace == "default"
    
    # Test with custom parameters
    with patch("agents.storage.backends.elasticsearch_storage.Elasticsearch") as mock_client:
        storage = ElasticsearchStorage(
            hosts="http://localhost:9200",
            namespace="test",
            api_key="test_key",
            basic_auth=("user", "pass")
        )
        mock_client.assert_called_once_with(
            hosts="http://localhost:9200",
            api_key="test_key",
            basic_auth=("user", "pass")
        )
        assert storage.client is mock_client.return_value
        assert storage.namespace == "test"


@pytest.mark.asyncio
async def test_qdrant_storage_methods():
    """Test QdrantStorage methods with mocks."""
    with patch("agents.storage.backends.qdrant_storage.QdrantClient") as mock_client_cls:
        mock_client = mock_client_cls.return_value
        mock_client.get_collection.return_value = None
        mock_client.create_collection = AsyncMock()
        mock_client.upsert = AsyncMock()
        mock_client.search = AsyncMock()
        mock_client.retrieve = AsyncMock()
        mock_client.delete = AsyncMock()
        mock_client.scroll = AsyncMock()
        
        # Initialize storage
        storage = QdrantStorage(collection_name="test", vector_size=4)
        
        # Test create_collection
        await storage._ensure_collection()
        mock_client.create_collection.assert_called_once()
        
        # Test store_vector
        vector = [0.1, 0.2, 0.3, 0.4]
        metadata = {"text": "test"}
        await storage.store_vector("test-id", vector, metadata)
        mock_client.upsert.assert_called_once()
        
        # Test search_by_vector
        mock_client.search.return_value = [
            MagicMock(id="test-id", score=0.95, payload={"text": "test"})
        ]
        
        results = await storage.search_by_vector(vector, limit=5)
        assert len(results) == 1
        assert results[0]["id"] == "test-id"
        mock_client.search.assert_called_once()
        
        # Test retrieve
        mock_client.retrieve.return_value = [
            MagicMock(id="test-id", payload={"text": "test"}, vector=[0.1, 0.2, 0.3, 0.4])
        ]
        
        result = await storage.retrieve("test-id")
        assert result is not None
        assert result["metadata"]["text"] == "test"
        mock_client.retrieve.assert_called_once()
        
        # Test delete
        mock_client.delete.return_value = None
        
        result = await storage.delete("test-id")
        assert result is True
        mock_client.delete.assert_called_once()


@pytest.mark.asyncio
async def test_neo4j_storage_methods():
    """Test Neo4jStorage methods with mocks."""
    with patch("agents.storage.backends.neo4j_storage.GraphDatabase") as mock_db:
        mock_driver = AsyncMock()
        mock_db.driver.return_value = mock_driver
        
        mock_session = AsyncMock()
        mock_driver.session.return_value.__aenter__.return_value = mock_session
        
        # Mock exec_query results
        mock_session.run.return_value.__aenter__.return_value.data.return_value = [
            {"n": {"id": "node-id", "properties": {"name": "test"}}}
        ]
        
        # Initialize storage
        storage = Neo4jStorage()
        
        # Test create_node
        node_type = "TestNode"
        properties = {"name": "test"}
        
        node_id = await storage.create_node(node_type, properties)
        assert node_id is not None
        mock_session.run.assert_called_once()
        
        # Reset mock
        mock_session.run.reset_mock()
        
        # Test get_nodes
        mock_session.run.return_value.__aenter__.return_value.data.return_value = [
            {"n": {"id": "node-id", "labels": ["TestNode"], "properties": {"name": "test"}}}
        ]
        
        nodes = await storage.get_nodes("TestNode", {"name": "test"})
        assert len(nodes) == 1
        assert nodes[0]["properties"]["name"] == "test"
        mock_session.run.assert_called_once()
        
        # Reset mock
        mock_session.run.reset_mock()
        
        # Test create_relationship
        source_id = "source-id"
        target_id = "target-id"
        rel_type = "RELATES_TO"
        
        mock_session.run.return_value.__aenter__.return_value.data.return_value = [
            {"r": {"id": "rel-id", "type": rel_type, "properties": {}}}
        ]
        
        rel_id = await storage.create_relationship(source_id, target_id, rel_type)
        assert rel_id is not None
        mock_session.run.assert_called_once()


@pytest.mark.asyncio
async def test_mongodb_storage_methods():
    """Test MongoDBStorage methods with mocks."""
    with patch("agents.storage.backends.mongodb_storage.MongoClient") as mock_client_cls:
        mock_client = mock_client_cls.return_value
        mock_db = AsyncMock()
        mock_collection = AsyncMock()
        
        mock_client.__getitem__.return_value = mock_db
        mock_db.__getitem__.return_value = mock_collection
        
        # Set up mock return values
        mock_collection.insert_one.return_value = AsyncMock(inserted_id="doc-id")
        mock_collection.find_one.return_value = {"_id": "doc-id", "title": "test"}
        mock_collection.find.return_value.to_list.return_value = [
            {"_id": "doc-id", "title": "test"}
        ]
        mock_collection.create_index.return_value = "index-name"
        
        # Initialize storage
        storage = MongoDBStorage()
        
        # Test insert_document
        collection = "test_collection"
        document = {"title": "test", "content": "test content"}
        
        doc_id = await storage.insert_document(collection, document)
        assert doc_id is not None
        mock_db.__getitem__.assert_called_with(collection)
        mock_collection.insert_one.assert_called_once()
        
        # Test retrieve
        result = await storage.retrieve("doc-id", collection=collection)
        assert result is not None
        assert result["title"] == "test"
        mock_collection.find_one.assert_called_once()
        
        # Test find_documents
        query = {"title": "test"}
        
        results = await storage.find_documents(collection, query)
        assert len(results) == 1
        assert results[0]["title"] == "test"
        mock_collection.find.assert_called_once()
        
        # Test create_index
        fields = {"title": "text", "content": "text"}
        
        index_name = await storage.create_index(collection, fields)
        assert index_name == "index-name"
        mock_collection.create_index.assert_called_once()


@pytest.mark.asyncio
async def test_elasticsearch_storage_methods():
    """Test ElasticsearchStorage methods with mocks."""
    with patch("agents.storage.backends.elasticsearch_storage.AsyncElasticsearch") as mock_client_cls:
        mock_client = mock_client_cls.return_value
        
        # Set up mock return values
        mock_client.indices.exists.return_value = False
        mock_client.indices.create = AsyncMock()
        mock_client.index.return_value = {"_id": "doc-id"}
        mock_client.get.return_value = {"_id": "doc-id", "_source": {"title": "test"}}
        mock_client.search.return_value = {
            "hits": {
                "total": {"value": 1},
                "hits": [
                    {
                        "_id": "doc-id",
                        "_score": 0.95,
                        "_source": {"title": "test"}
                    }
                ]
            }
        }
        
        # Initialize storage
        storage = ElasticsearchStorage()
        
        # Test create_index
        index = "test_index"
        mappings = {"properties": {"title": {"type": "text"}}}
        
        await storage.create_index(index, mappings)
        mock_client.indices.exists.assert_called_once()
        mock_client.indices.create.assert_called_once()
        
        # Test insert_document
        document = {"title": "test", "content": "test content"}
        
        doc_id = await storage.insert_document(index, document)
        assert doc_id is not None
        mock_client.index.assert_called_once()
        
        # Test retrieve
        result = await storage.retrieve("doc-id", index=index)
        assert result is not None
        assert result["title"] == "test"
        mock_client.get.assert_called_once()
        
        # Test text_search
        query = "test"
        fields = ["title", "content"]
        
        results = await storage.text_search(index, query, fields)
        assert len(results) == 1
        assert results[0]["id"] == "doc-id"
        assert results[0]["score"] == 0.95
        mock_client.search.assert_called_once() 