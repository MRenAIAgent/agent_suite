# Knowledge Creation Module

This module provides tools for creating and managing knowledge using knowledge graphs and vector databases.

## Overview

The Knowledge Creation module offers a unified interface for working with structured and unstructured data:

- **Knowledge Graphs**: Store entities and relationships in a graph structure for sophisticated relationship-based querying
- **Vector Database**: Store and retrieve unstructured text using semantic similarity search with Cohere embeddings
- **Combined**: Use both approaches together for maximum flexibility

## Features

- Create, update, and delete entities and relationships
- Store unstructured text with metadata
- Query entities and relationships with filtering
- Perform semantic searches over stored knowledge using state-of-the-art Cohere embeddings
- Centralized vector DB and embedding services shared across the Agent Suite
- Automatic persistence of data
- Extensible architecture

## Components

- `KnowledgeBase`: Abstract base class defining the interface
- `GraphitiKnowledgeGraph`: Implementation using Graphiti for knowledge graphs
- `QdrantVectorDatabase`: Implementation using Qdrant and Cohere for vector storage
- `CombinedKnowledgeBase`: Implementation using both technologies
- `KnowledgeBaseFactory`: Factory class for creating instances

## Usage

```python
from knowledge.factory import KnowledgeBaseFactory

# Create a knowledge base with Cohere embeddings
kb = KnowledgeBaseFactory.create_knowledge_base("combined", {
    "kg_graph_name": "my_knowledge",
    "vector_collection_name": "my_vectors",
    "embedding_model": "embed-english-v3.0",  # Cohere model
    "persist": True,
    "cohere_api_key": "your_api_key"  # or set in .env file
})

# Add an entity
entity_id = kb.add_entity(
    entity_id="person-1",
    entity_type="person",
    properties={
        "name": "John Doe",
        "age": 35,
        "profession": "Software Engineer"
    }
)

# Add a text document
text_id = kb.add_text(
    "Python is a high-level programming language known for its readability and simplicity.",
    {"topic": "Programming", "source": "Documentation"}
)

# Query entities
people = kb.query_entities(entity_type="person")

# Semantic search
results = kb.semantic_search("Python programming language")
```

## Examples

- `knowledge/example.py`: Comprehensive examples of using the different knowledge base implementations
- `knowledge/cohere_example.py`: Examples specifically showcasing the Cohere embedding capabilities
- `knowledge/agent_integration.py`: Demo of integrating knowledge with agents

## Centralized Vector Infrastructure

The module now includes a centralized vector database and embedding infrastructure that can be shared by other modules in the Agent Suite:

- `agent_suite.vector.qdrant_client`: Centralized client for Qdrant operations
- `agent_suite.vector.embedding_service`: Embedding generation service using Cohere

This architecture allows both the knowledge module and long-term storage implementations to leverage the same vector infrastructure.

## API Keys

API keys for Cohere and Qdrant (if using cloud version) can be provided in two ways:

1. Directly in the configuration when creating the knowledge base:
   ```python
   kb = KnowledgeBaseFactory.create_knowledge_base("vector", {
       "cohere_api_key": "your_api_key",
       "qdrant_api_key": "your_api_key"
   })
   ```

2. Through environment variables (recommended):
   - Set `COHERE_API_KEY` and `QDRANT_API_KEY` in your environment
   - Or create a `.env` file in your project root containing these variables

## Requirements

Install dependencies:

```
pip install -r knowledge/requirements.txt
```

## Dependencies

- `graphiti`: For knowledge graph storage and querying
- `qdrant-client`: For vector database capabilities
- `cohere`: For generating high-quality embeddings
- `sentence-transformers`: For backward compatibility (optional)
- `numpy`: For numerical operations
- `python-dotenv`: For loading environment variables
- `torch`: (Optional) For better performance with embeddings 