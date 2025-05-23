"""Storage backend implementations.

This module provides implementations for different storage backends
including vector, graph, and document storage services.
"""

# Check which backends are available
try:
    from agents.storage.backends.qdrant_storage import QdrantStorage, QDRANT_AVAILABLE
except ImportError:
    QdrantStorage = None
    QDRANT_AVAILABLE = False

try:
    from agents.storage.backends.neo4j_storage import Neo4jStorage, NEO4J_AVAILABLE
except ImportError:
    Neo4jStorage = None
    NEO4J_AVAILABLE = False

try:
    from agents.storage.backends.mongodb_storage import MongoDBStorage, MONGODB_AVAILABLE
except ImportError:
    MongoDBStorage = None
    MONGODB_AVAILABLE = False

try:
    from agents.storage.backends.elasticsearch_storage import (
        ElasticsearchStorage, ELASTICSEARCH_AVAILABLE
    )
except ImportError:
    ElasticsearchStorage = None
    ELASTICSEARCH_AVAILABLE = False


# Export available backends
__all__ = [
    # Vector storage
    "QdrantStorage",
    "QDRANT_AVAILABLE",
    
    # Graph storage
    "Neo4jStorage",
    "NEO4J_AVAILABLE",
    
    # Document storage
    "MongoDBStorage",
    "MONGODB_AVAILABLE",
    "ElasticsearchStorage",
    "ELASTICSEARCH_AVAILABLE",
] 