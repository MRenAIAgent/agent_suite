# Storage Service

This module provides a unified storage service with support for different backend technologies, organized by storage type:

- **Vector Storage**: Optimized for similarity search (Qdrant)
- **Graph Storage**: For modeling relationships between entities (Neo4j)
- **Document Storage**: For storing and querying structured documents (MongoDB, Elasticsearch)

## Architecture

The storage service follows a modular design with clear interfaces:

- **Base Interfaces**: Abstract classes defining storage operations
- **Backend Implementations**: Concrete implementations for specific technologies
- **Storage Service**: Central service class for managing multiple backends
- **Factory**: Functions for creating and configuring storage backends

## Available Backends

### Vector Storage
- **Qdrant**: High-performance vector database for similarity search

### Graph Storage
- **Neo4j**: Popular graph database for storing and querying graph data

### Document Storage
- **MongoDB**: Flexible document database
- **Elasticsearch**: Search and analytics engine with powerful text search

## Usage

### Basic Usage

```python
from agents.storage import create_storage_service

# Configure backends
backends = {
    "vector": {
        "type": "qdrant",
        "config": {
            "collection_name": "my_vectors",
            "vector_size": 1536,
            "host": "localhost",
            "port": 6333
        }
    },
    "graph": {
        "type": "neo4j",
        "config": {
            "uri": "bolt://localhost:7687",
            "username": "neo4j",
            "password": "password",
            "namespace": "my_app"
        }
    },
    "document": {
        "type": "mongodb",
        "config": {
            "uri": "mongodb://localhost:27017",
            "database": "my_app",
            "namespace": "my_app"
        }
    }
}

# Create storage service with multiple backends
service = create_storage_service(backends)

# Get a specific backend
vector_store = service.get_backend("vector")
graph_store = service.get_backend("graph")
doc_store = service.get_backend("document")

# Use the service
async def example():
    # Store data
    key = await vector_store.store("item1", {
        "vector": [0.1, 0.2, 0.3, ...],
        "metadata": {"text": "Example document"}
    })
    
    # Search by vector
    results = await vector_store.search_by_vector(
        query_vector=[0.1, 0.2, 0.3, ...],
        limit=5
    )
    
    # Close backends when done
    await service.close()
```

### Direct Backend Usage

If you prefer to use a specific backend directly:

```python
from agents.storage import create_vector_storage, create_graph_storage, create_document_storage

# Create vector storage
vector_store = create_vector_storage(
    backend_type="qdrant",
    config={
        "collection_name": "my_vectors",
        "vector_size": 1536
    }
)

# Create graph storage
graph_store = create_graph_storage(
    backend_type="neo4j",
    config={
        "uri": "bolt://localhost:7687",
        "username": "neo4j",
        "password": "password"
    }
)

# Create document storage
doc_store = create_document_storage(
    backend_type="mongodb",
    config={
        "uri": "mongodb://localhost:27017",
        "database": "my_app"
    }
)
```

## Dependencies

Each backend requires its corresponding client library:

- Qdrant: `pip install qdrant-client`
- Neo4j: `pip install neo4j`
- MongoDB: `pip install pymongo`
- Elasticsearch: `pip install elasticsearch`

The storage service will work with available backends, so you only need to install the dependencies for the backends you plan to use.

## Examples

See the `examples/storage_example.py` file for a complete example of using the storage service with different backends. 