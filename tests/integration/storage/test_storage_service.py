"""Integration tests for the storage service."""

import pytest
import uuid
import numpy as np

from agents.storage import (
    StorageService,
    create_vector_storage,
    create_graph_storage,
    create_document_storage,
    QDRANT_AVAILABLE,
    NEO4J_AVAILABLE,
    MONGODB_AVAILABLE
)


@pytest.mark.asyncio
@pytest.mark.skipif(not (QDRANT_AVAILABLE and NEO4J_AVAILABLE and MONGODB_AVAILABLE),
                   reason="Not all required backends are available")
async def test_storage_service_with_multiple_backends(vector_storage, graph_storage, mongodb_storage):
    """Test the storage service with multiple backends."""
    # Create a storage service
    service = StorageService()
    
    # Register different backends
    await service.register_backend("vector", vector_storage)
    await service.register_backend("graph", graph_storage)
    await service.register_backend("document", mongodb_storage)
    
    # Create test collection for document storage
    collection = f"service_test_{uuid.uuid4().hex[:8]}"
    
    # Test that all backends are registered
    assert await service.get_backend("vector") == vector_storage
    assert await service.get_backend("graph") == graph_storage
    assert await service.get_backend("document") == mongodb_storage
    
    # Test vector storage through service
    vector_backend = await service.get_backend("vector")
    vector = [0.1, 0.2, 0.3, 0.4]
    metadata = {"text": "This is a test", "tags": ["test", "vector"]}
    
    vec_id = await vector_backend.store_vector("test_vector", vector, metadata)
    assert vec_id is not None
    
    # Test graph storage through service
    graph_backend = await service.get_backend("graph")
    
    # Create a node
    node_props = {"name": "Test Node", "type": "test"}
    node_id = await graph_backend.create_node("TestNode", node_props)
    assert node_id is not None
    
    # Test document storage through service
    doc_backend = await service.get_backend("document")
    
    # Create a document
    document = {
        "title": "Test Document",
        "content": "This is a test document for the service integration test.",
        "tags": ["test", "service", "integration"]
    }
    
    doc_id = await doc_backend.insert_document(collection, document)
    assert doc_id is not None
    
    # Test complex use case with multiple backends
    # 1. Search for a vector
    vec_results = await vector_backend.search_by_vector(vector, limit=1)
    assert len(vec_results) == 1
    assert vec_results[0]["id"] == "test_vector"
    
    # 2. Create a node representing the document
    doc_node_id = await graph_backend.create_node(
        "Document",
        {
            "title": document["title"],
            "vector_id": "test_vector",
            "document_id": doc_id
        }
    )
    assert doc_node_id is not None
    
    # 3. Create relationship between document node and test node
    rel_id = await graph_backend.create_relationship(
        doc_node_id,
        node_id,
        "RELATED_TO",
        {"strength": 0.95}
    )
    assert rel_id is not None
    
    # 4. Get relationships from graph
    relationships = await graph_backend.get_relationships(source_id=doc_node_id)
    assert len(relationships) == 1
    assert relationships[0]["target"]["id"] == node_id
    
    # 5. Query the document for its content
    doc_result = await doc_backend.retrieve(doc_id, collection=collection)
    assert doc_result is not None
    assert doc_result["title"] == document["title"]
    
    # Test getting backends by type
    vector_backends = await service.get_vector_backends()
    assert len(vector_backends) == 1
    assert vector_backends[0] == vector_storage
    
    graph_backends = await service.get_graph_backends()
    assert len(graph_backends) == 1
    assert graph_backends[0] == graph_storage
    
    document_backends = await service.get_document_backends()
    assert len(document_backends) == 1
    assert document_backends[0] == mongodb_storage
    
    # Test removing a backend
    result = await service.remove_backend("vector")
    assert result is True
    
    # Verify backend is removed
    assert await service.get_backend("vector") is None
    
    # Clean up
    await graph_backend.clear()
    await doc_backend.clear(collection=collection)
    
    # Close the service (which should close all backends)
    await service.close()


@pytest.mark.asyncio
async def test_storage_service_factory():
    """Test creating a storage service using the factory pattern."""
    # Skip if not all backends available
    if not (QDRANT_AVAILABLE and NEO4J_AVAILABLE and MONGODB_AVAILABLE):
        pytest.skip("Not all required backends are available")
    
    # Import the factory function
    from agents.storage import create_storage_service
    
    # Configure backends for the service
    backends_config = {
        "vector": {
            "type": "qdrant",
            "config": {
                "collection_name": f"test_vectors_{uuid.uuid4().hex[:8]}",
                "vector_size": 4
            }
        },
        "graph": {
            "type": "neo4j",
            "config": {}  # Use default configuration
        },
        "document": {
            "type": "mongodb",
            "config": {
                "database": "test_db"
            }
        }
    }
    
    # Create the service
    service = await create_storage_service(backends_config)
    assert service is not None
    
    # Test that all backends are registered
    vector_backend = await service.get_backend("vector")
    assert vector_backend is not None
    
    graph_backend = await service.get_backend("graph")
    assert graph_backend is not None
    
    document_backend = await service.get_backend("document")
    assert document_backend is not None
    
    # Test basic operations with each backend
    # Vector storage
    vector = [0.1, 0.2, 0.3, 0.4]
    vec_id = await vector_backend.store_vector("factory_test", vector, {"test": True})
    assert vec_id is not None
    
    # Graph storage
    node_id = await graph_backend.create_node("TestFactoryNode", {"name": "Factory Test"})
    assert node_id is not None
    
    # Document storage
    collection = f"factory_test_{uuid.uuid4().hex[:8]}"
    doc_id = await document_backend.insert_document(
        collection,
        {"title": "Factory Test", "content": "Testing the factory pattern"}
    )
    assert doc_id is not None
    
    # Clean up
    await vector_backend.clear()
    await graph_backend.clear()
    await document_backend.clear(collection=collection)
    
    # Close the service
    await service.close() 