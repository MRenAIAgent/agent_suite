"""Test fixtures for storage integration tests."""

import os
import asyncio
import pytest
import docker
import time
import uuid

from agents.storage import (
    # Factory
    create_storage_service,
    create_vector_storage,
    create_graph_storage,
    create_document_storage,
    
    # Availability flags
    QDRANT_AVAILABLE,
    NEO4J_AVAILABLE,
    MONGODB_AVAILABLE,
    ELASTICSEARCH_AVAILABLE
)


# Define constants for test services
QDRANT_PORT = 6333
NEO4J_PORT = 7687
NEO4J_HTTP_PORT = 7474
MONGODB_PORT = 27017
ELASTICSEARCH_PORT = 9200

# Skip flags for specific backends
skip_qdrant = not QDRANT_AVAILABLE
skip_neo4j = not NEO4J_AVAILABLE
skip_mongodb = not MONGODB_AVAILABLE
skip_elasticsearch = not ELASTICSEARCH_AVAILABLE


# Function to check if a port is accessible
def is_port_accessible(port, host="localhost"):
    """Check if a port is accessible on the given host."""
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex((host, port))
    sock.close()
    return result == 0


# Skip decorators for specific backends
skip_if_qdrant_unavailable = pytest.mark.skipif(
    skip_qdrant or not is_port_accessible(QDRANT_PORT),
    reason="Qdrant is not available"
)

skip_if_neo4j_unavailable = pytest.mark.skipif(
    skip_neo4j or not is_port_accessible(NEO4J_PORT),
    reason="Neo4j is not available"
)

skip_if_mongodb_unavailable = pytest.mark.skipif(
    skip_mongodb or not is_port_accessible(MONGODB_PORT),
    reason="MongoDB is not available"
)

skip_if_elasticsearch_unavailable = pytest.mark.skipif(
    skip_elasticsearch or not is_port_accessible(ELASTICSEARCH_PORT),
    reason="Elasticsearch is not available"
)


@pytest.fixture
def test_namespace():
    """Generate a unique namespace for testing."""
    return f"test_{uuid.uuid4().hex[:8]}"


@pytest.fixture
async def vector_storage(test_namespace):
    """Fixture to create a vector storage instance for testing."""
    if not QDRANT_AVAILABLE or not is_port_accessible(QDRANT_PORT):
        pytest.skip("Qdrant is not available")
    
    # Create storage with a unique collection name
    storage = create_vector_storage(
        backend_type="qdrant",
        config={
            "collection_name": test_namespace,
            "vector_size": 4,  # Small vectors for testing
            "host": "localhost",
            "port": QDRANT_PORT
        }
    )
    
    yield storage
    
    # Cleanup
    await storage.clear(keep_collection=False)


@pytest.fixture
async def graph_storage(test_namespace):
    """Fixture to create a graph storage instance for testing."""
    if not NEO4J_AVAILABLE or not is_port_accessible(NEO4J_PORT):
        pytest.skip("Neo4j is not available")
    
    # Create storage with a unique namespace
    storage = create_graph_storage(
        backend_type="neo4j",
        config={
            "uri": f"bolt://localhost:{NEO4J_PORT}",
            "username": "neo4j",
            "password": "password",  # This should match your Neo4j password
            "namespace": test_namespace
        }
    )
    
    yield storage
    
    # Cleanup
    await storage.clear(only_namespace=True)
    storage.close()


@pytest.fixture
async def document_storage_mongodb(test_namespace):
    """Fixture to create a MongoDB document storage instance for testing."""
    if not MONGODB_AVAILABLE or not is_port_accessible(MONGODB_PORT):
        pytest.skip("MongoDB is not available")
    
    # Create storage with a unique namespace
    storage = create_document_storage(
        backend_type="mongodb",
        config={
            "uri": f"mongodb://localhost:{MONGODB_PORT}",
            "database": "test_db",
            "namespace": test_namespace
        }
    )
    
    yield storage
    
    # Cleanup
    await storage.clear()
    storage.close()


@pytest.fixture
async def document_storage_elasticsearch(test_namespace):
    """Fixture to create an Elasticsearch document storage instance for testing."""
    if not ELASTICSEARCH_AVAILABLE or not is_port_accessible(ELASTICSEARCH_PORT):
        pytest.skip("Elasticsearch is not available")
    
    # Create storage with a unique namespace
    storage = create_document_storage(
        backend_type="elasticsearch",
        config={
            "hosts": f"http://localhost:{ELASTICSEARCH_PORT}",
            "namespace": test_namespace
        }
    )
    
    yield storage
    
    # Cleanup
    await storage.clear()
    await storage.close()


@pytest.fixture
async def storage_service(test_namespace):
    """Fixture to create a storage service with all available backends."""
    backends = {}
    
    # Add available backends
    if QDRANT_AVAILABLE and is_port_accessible(QDRANT_PORT):
        backends["vector"] = {
            "type": "qdrant",
            "config": {
                "collection_name": f"{test_namespace}_vector",
                "vector_size": 4,
                "host": "localhost",
                "port": QDRANT_PORT
            }
        }
    
    if NEO4J_AVAILABLE and is_port_accessible(NEO4J_PORT):
        backends["graph"] = {
            "type": "neo4j",
            "config": {
                "uri": f"bolt://localhost:{NEO4J_PORT}",
                "username": "neo4j",
                "password": "password",  # This should match your Neo4j password
                "namespace": f"{test_namespace}_graph"
            }
        }
    
    if MONGODB_AVAILABLE and is_port_accessible(MONGODB_PORT):
        backends["document"] = {
            "type": "mongodb",
            "config": {
                "uri": f"mongodb://localhost:{MONGODB_PORT}",
                "database": "test_db",
                "namespace": f"{test_namespace}_document"
            }
        }
    elif ELASTICSEARCH_AVAILABLE and is_port_accessible(ELASTICSEARCH_PORT):
        backends["document"] = {
            "type": "elasticsearch",
            "config": {
                "hosts": f"http://localhost:{ELASTICSEARCH_PORT}",
                "namespace": f"{test_namespace}_document"
            }
        }
    
    if not backends:
        pytest.skip("No storage backends are available")
    
    # Create the service
    service = create_storage_service(backends)
    
    yield service
    
    # Cleanup
    for name, backend in backends.items():
        backend_instance = await service.get_backend(name)
        if backend_instance:
            await backend_instance.clear()


# Event loop fixture for running async tests
@pytest.fixture
def event_loop():
    """Create an instance of the default event loop for each test case."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close() 