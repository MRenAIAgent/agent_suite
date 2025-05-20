# Vector Embedding Examples

This directory contains examples demonstrating how to use the Cohere embeddings with the vector database infrastructure.

## Prerequisites

1. **API Keys**: You need a Cohere API key to run these examples.
   - Get a Cohere API key from [dashboard.cohere.com/api-keys](https://dashboard.cohere.com/api-keys)

2. **Environment Setup**:
   - Create a `.env` file in the project root with your API keys:
     ```
     COHERE_API_KEY=your_cohere_api_key_here
     QDRANT_API_KEY=your_qdrant_cloud_api_key_here  # Only if using Qdrant Cloud
     ```

3. **Install Dependencies**:
   - Ensure you have the required packages installed:
     ```
     pip install -r knowledge/requirements.txt
     ```

## Examples

### 1. Low-Level Vector Operations (example_cohere_vector.py)

This example demonstrates direct use of the centralized vector infrastructure:
- Using `EmbeddingService` from `agent_suite.vector.embedding_service`
- Using `QdrantManager` from `agent_suite.vector.qdrant_client`
- Creating embeddings, storing them, and retrieving via semantic search

Run the example:
```
python example_cohere_vector.py
```

### 2. High-Level Knowledge Module API (example_knowledge_vector.py)

This example demonstrates using the knowledge module's high-level API:
- Using `KnowledgeBaseFactory` to create a vector knowledge base
- Adding text documents with metadata
- Performing semantic search queries

Run the example:
```
python example_knowledge_vector.py
```

## Key Features

The Cohere embedding implementation provides:
- State-of-the-art `embed-english-v3.0` model (1024 dimensions)
- Optimized embedding generation for queries vs. documents
- Efficient caching mechanism for better performance
- Compatibility with both local and cloud Qdrant instances

## Architecture

The vector infrastructure is implemented in two layers:
1. **Core infrastructure** (`agent_suite/vector/`):
   - `QdrantManager`: A centralized client for Qdrant operations
   - `EmbeddingService`: Service for generating embeddings with Cohere

2. **Knowledge module integration** (`knowledge/vector/`):
   - `vector_db_cohere.py`: Knowledge base implementation using Cohere embeddings

This design allows the same infrastructure to be used by both the knowledge module and long-term storage features. 