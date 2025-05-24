"""Test fixtures for storage unit tests."""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import numpy as np

from agents.storage.base_storage import BaseStorage
from agents.storage.vector_storage import VectorStorage
from agents.storage.graph_storage import GraphStorage
from agents.storage.document_storage import DocumentStorage
from agents.storage.service.storage_service import StorageService


class MockBaseStorage(BaseStorage):
    """Mock implementation of BaseStorage for testing."""
    
    def __init__(self):
        self.store_mock = AsyncMock(return_value="mock-id")
        self.retrieve_mock = AsyncMock(return_value={"test": "data"})
        self.delete_mock = AsyncMock(return_value=True)
        self.update_mock = AsyncMock(return_value=True)
        self.list_mock = AsyncMock(return_value=["id1", "id2"])
        self.clear_mock = AsyncMock(return_value=True)
    
    async def store(self, key, data, **kwargs):
        return await self.store_mock(key, data, **kwargs)
    
    async def retrieve(self, key, **kwargs):
        return await self.retrieve_mock(key, **kwargs)
    
    async def delete(self, key, **kwargs):
        return await self.delete_mock(key, **kwargs)
    
    async def update(self, key, data, **kwargs):
        return await self.update_mock(key, data, **kwargs)
    
    async def list(self, pattern="*", **kwargs):
        return await self.list_mock(pattern, **kwargs)
    
    async def clear(self, **kwargs):
        return await self.clear_mock(**kwargs)


class MockVectorStorage(VectorStorage):
    """Mock implementation of VectorStorage for testing."""
    
    def __init__(self):
        self.store_mock = AsyncMock(return_value="mock-id")
        self.retrieve_mock = AsyncMock(return_value={"vector": [0.1, 0.2], "metadata": {"test": "data"}})
        self.delete_mock = AsyncMock(return_value=True)
        self.update_mock = AsyncMock(return_value=True)
        self.list_mock = AsyncMock(return_value=["id1", "id2"])
        self.clear_mock = AsyncMock(return_value=True)
        self.store_vector_mock = AsyncMock(return_value="mock-vector-id")
        self.store_batch_mock = AsyncMock(return_value=["id1", "id2"])
        self.search_by_vector_mock = AsyncMock(return_value=[
            {"id": "id1", "vector": [0.1, 0.2], "metadata": {"text": "test"}, "score": 0.95},
            {"id": "id2", "vector": [0.3, 0.4], "metadata": {"text": "test2"}, "score": 0.85}
        ])
        self.search_by_metadata_mock = AsyncMock(return_value=[
            {"id": "id1", "vector": [0.1, 0.2], "metadata": {"tag": "test"}},
            {"id": "id2", "vector": [0.3, 0.4], "metadata": {"tag": "test"}}
        ])
    
    async def store(self, key, data, **kwargs):
        return await self.store_mock(key, data, **kwargs)
    
    async def retrieve(self, key, **kwargs):
        return await self.retrieve_mock(key, **kwargs)
    
    async def delete(self, key, **kwargs):
        return await self.delete_mock(key, **kwargs)
    
    async def update(self, key, data, **kwargs):
        return await self.update_mock(key, data, **kwargs)
    
    async def list(self, pattern="*", **kwargs):
        return await self.list_mock(pattern, **kwargs)
    
    async def clear(self, **kwargs):
        return await self.clear_mock(**kwargs)
    
    async def store_vector(self, key, vector, metadata=None, **kwargs):
        return await self.store_vector_mock(key, vector, metadata, **kwargs)
    
    async def store_batch(self, items, **kwargs):
        return await self.store_batch_mock(items, **kwargs)
    
    async def search_by_vector(self, query_vector, limit=10, score_threshold=None,
                               filter_metadata=None, **kwargs):
        return await self.search_by_vector_mock(query_vector, limit, score_threshold, 
                                         filter_metadata, **kwargs)
    
    async def search_by_metadata(self, filter_metadata, limit=10, **kwargs):
        return await self.search_by_metadata_mock(filter_metadata, limit, **kwargs)


class MockGraphStorage(GraphStorage):
    """Mock implementation of GraphStorage for testing."""
    
    def __init__(self):
        self.store_mock = AsyncMock(return_value="mock-id")
        self.retrieve_mock = AsyncMock(return_value={"type": "node", "properties": {"name": "test"}})
        self.delete_mock = AsyncMock(return_value=True)
        self.update_mock = AsyncMock(return_value=True)
        self.list_mock = AsyncMock(return_value=["id1", "id2"])
        self.clear_mock = AsyncMock(return_value=True)
        self.create_node_mock = AsyncMock(return_value="node-id")
        self.create_relationship_mock = AsyncMock(return_value="rel-id")
        self.get_nodes_mock = AsyncMock(return_value=[
            {"id": "node1", "type": "Person", "properties": {"name": "Alice"}},
            {"id": "node2", "type": "Person", "properties": {"name": "Bob"}}
        ])
        self.get_relationships_mock = AsyncMock(return_value=[
            {
                "relationship": {"id": "rel1", "type": "KNOWS", "properties": {"since": "2020"}},
                "source": {"id": "node1"},
                "target": {"id": "node2"}
            }
        ])
        self.query_mock = AsyncMock(return_value=[{"n": {"name": "test"}}])
        self.traverse_mock = AsyncMock(return_value={
            "start_node": {"id": "node1"},
            "paths": [{"segments": [{"start": {"id": "node1"}, "end": {"id": "node2"}}]}]
        })
    
    async def store(self, key, data, **kwargs):
        return await self.store_mock(key, data, **kwargs)
    
    async def retrieve(self, key, **kwargs):
        return await self.retrieve_mock(key, **kwargs)
    
    async def delete(self, key, **kwargs):
        return await self.delete_mock(key, **kwargs)
    
    async def update(self, key, data, **kwargs):
        return await self.update_mock(key, data, **kwargs)
    
    async def list(self, pattern="*", **kwargs):
        return await self.list_mock(pattern, **kwargs)
    
    async def clear(self, **kwargs):
        return await self.clear_mock(**kwargs)
    
    async def create_node(self, node_type, properties, **kwargs):
        return await self.create_node_mock(node_type, properties, **kwargs)
    
    async def create_relationship(self, source_id, target_id, relationship_type, 
                                  properties=None, **kwargs):
        return await self.create_relationship_mock(source_id, target_id, relationship_type, 
                                            properties, **kwargs)
    
    async def get_nodes(self, node_type=None, properties=None, limit=100, **kwargs):
        return await self.get_nodes_mock(node_type, properties, limit, **kwargs)
    
    async def get_relationships(self, source_id=None, target_id=None, relationship_type=None,
                               properties=None, limit=100, **kwargs):
        return await self.get_relationships_mock(source_id, target_id, relationship_type,
                                          properties, limit, **kwargs)
    
    async def query(self, query, parameters=None, **kwargs):
        return await self.query_mock(query, parameters, **kwargs)
    
    async def traverse(self, start_node_id, relationship_types=None, direction="outgoing",
                      max_depth=1, **kwargs):
        return await self.traverse_mock(start_node_id, relationship_types, direction,
                                 max_depth, **kwargs)


class MockDocumentStorage(DocumentStorage):
    """Mock implementation of DocumentStorage for testing."""
    
    def __init__(self):
        self.store_mock = AsyncMock(return_value="mock-id")
        self.retrieve_mock = AsyncMock(return_value={"id": "doc1", "title": "Test"})
        self.delete_mock = AsyncMock(return_value=True)
        self.update_mock = AsyncMock(return_value=True)
        self.list_mock = AsyncMock(return_value=["id1", "id2"])
        self.clear_mock = AsyncMock(return_value=True)
        self.create_index_mock = AsyncMock(return_value="index-name")
        self.insert_document_mock = AsyncMock(return_value="doc-id")
        self.insert_documents_mock = AsyncMock(return_value=["doc1", "doc2"])
        self.find_documents_mock = AsyncMock(return_value=[
            {"id": "doc1", "title": "Test 1"},
            {"id": "doc2", "title": "Test 2"}
        ])
        self.update_document_mock = AsyncMock(return_value=True)
        self.delete_documents_mock = AsyncMock(return_value=2)
        self.aggregate_mock = AsyncMock(return_value=[{"_id": "group1", "count": 2}])
        self.text_search_mock = AsyncMock(return_value=[
            {"id": "doc1", "title": "Test 1", "score": 0.9},
            {"id": "doc2", "title": "Test 2", "score": 0.8}
        ])
    
    async def store(self, key, data, **kwargs):
        return await self.store_mock(key, data, **kwargs)
    
    async def retrieve(self, key, **kwargs):
        return await self.retrieve_mock(key, **kwargs)
    
    async def delete(self, key, **kwargs):
        return await self.delete_mock(key, **kwargs)
    
    async def update(self, key, data, **kwargs):
        return await self.update_mock(key, data, **kwargs)
    
    async def list(self, pattern="*", **kwargs):
        return await self.list_mock(pattern, **kwargs)
    
    async def clear(self, **kwargs):
        return await self.clear_mock(**kwargs)
    
    async def create_index(self, collection, fields, index_name=None, **kwargs):
        return await self.create_index_mock(collection, fields, index_name, **kwargs)
    
    async def insert_document(self, collection, document, **kwargs):
        return await self.insert_document_mock(collection, document, **kwargs)
    
    async def insert_documents(self, collection, documents, **kwargs):
        return await self.insert_documents_mock(collection, documents, **kwargs)
    
    async def find_documents(self, collection, query, projection=None, sort=None, 
                            limit=100, skip=0, **kwargs):
        return await self.find_documents_mock(collection, query, projection, sort,
                                       limit, skip, **kwargs)
    
    async def update_document(self, collection, document_id, update, upsert=False, **kwargs):
        return await self.update_document_mock(collection, document_id, update, upsert, **kwargs)
    
    async def delete_documents(self, collection, query, **kwargs):
        return await self.delete_documents_mock(collection, query, **kwargs)
    
    async def aggregate(self, collection, pipeline, **kwargs):
        return await self.aggregate_mock(collection, pipeline, **kwargs)
    
    async def text_search(self, collection, query, fields=None, limit=10, **kwargs):
        return await self.text_search_mock(collection, query, fields, limit, **kwargs)


@pytest.fixture
def mock_base_storage():
    """Fixture that returns a mock BaseStorage implementation."""
    return MockBaseStorage()


@pytest.fixture
def mock_vector_storage():
    """Fixture that returns a mock VectorStorage implementation."""
    return MockVectorStorage()


@pytest.fixture
def mock_graph_storage():
    """Fixture that returns a mock GraphStorage implementation."""
    return MockGraphStorage()


@pytest.fixture
def mock_document_storage():
    """Fixture that returns a mock DocumentStorage implementation."""
    return MockDocumentStorage()


@pytest.fixture
def storage_service():
    """Fixture that returns a fresh StorageService instance."""
    return StorageService()


@pytest.fixture
def event_loop():
    """Create an instance of the default event loop for each test case."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close() 