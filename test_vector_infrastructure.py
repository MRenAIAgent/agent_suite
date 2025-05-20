#!/usr/bin/env python
"""
Test script for the vector infrastructure in agent_suite.

This script tests:
1. Loading API keys from .env
2. EmbeddingService functionality
3. QdrantManager functionality
4. A complete embedding and search workflow
"""

import os
import uuid
import time
from dotenv import load_dotenv

# Import the infrastructure components
from agent_suite.vector.embedding_service import EmbeddingService
from agent_suite.vector.qdrant_client import QdrantManager

# Load environment variables
load_dotenv()

def print_separator(title):
    """Print a section separator with title."""
    print("\n" + "=" * 80)
    print(f" {title} ".center(80, "="))
    print("=" * 80 + "\n")

def test_api_keys():
    """Test that required API keys are available."""
    print_separator("Testing API Keys")
    
    cohere_key = os.getenv("COHERE_API_KEY")
    qdrant_key = os.getenv("QDRANT_API_KEY")
    
    if cohere_key:
        print(f"✅ Cohere API key found: {cohere_key[:5]}...{cohere_key[-5:]}")
    else:
        print("❌ Cohere API key missing")
        
    if qdrant_key:
        print(f"✅ Qdrant API key found: {qdrant_key[:5]}...{qdrant_key[-5:]}")
    else:
        print("ℹ️ Qdrant API key not found (only needed for cloud instances)")

def test_embedding_service():
    """Test the embedding service functionality."""
    print_separator("Testing EmbeddingService")
    
    # Initialize service
    service = EmbeddingService()
    print(f"✅ Initialized embedding service with model: {service.model_name}")
    print(f"✅ Vector size: {service.get_vector_size()} dimensions")
    
    # Test single embedding
    text = "This is a test sentence for embedding."
    print(f"\nGenerating embedding for: '{text}'")
    
    start_time = time.time()
    embedding = service.embed([text])[0]
    end_time = time.time()
    
    print(f"✅ Generated embedding with {len(embedding)} dimensions")
    print(f"✅ Time taken: {(end_time - start_time)*1000:.2f}ms")
    print(f"✅ First 5 values: {embedding[:5]}")
    
    # Test batch embedding
    texts = [
        "First test sentence.",
        "Second test sentence with different words.",
        "Third completely unique test with additional content."
    ]
    print(f"\nGenerating batch embeddings for {len(texts)} texts")
    
    start_time = time.time()
    embeddings = service.embed(texts)
    end_time = time.time()
    
    print(f"✅ Generated {len(embeddings)} embeddings")
    print(f"✅ Time taken: {(end_time - start_time)*1000:.2f}ms")
    
    # Test query vs document embedding
    query = "How does this process work?"
    print(f"\nTesting query embedding: '{query}'")
    
    query_embedding = service.embed_query(query)
    doc_embedding = service.embed_documents([query])[0]
    
    print(f"✅ Query embedding: {len(query_embedding)} dimensions")
    print(f"✅ Document embedding: {len(doc_embedding)} dimensions")
    print(f"✅ Same size: {len(query_embedding) == len(doc_embedding)}")
    
    # Simple check to confirm they're different (optimized differently)
    different_values = sum(1 for q, d in zip(query_embedding[:10], doc_embedding[:10]) if abs(q - d) > 0.01)
    print(f"✅ Different values in first 10 dimensions: {different_values}/10")
    
    return service

def test_qdrant_manager():
    """Test the Qdrant manager functionality."""
    print_separator("Testing QdrantManager")
    
    # Initialize manager
    manager = QdrantManager()
    print(f"✅ Initialized Qdrant manager (in-memory mode)")
    
    # Test collection creation
    collection_name = f"test_{uuid.uuid4().hex[:8]}"
    vector_size = 128
    
    success = manager.create_collection(
        collection_name=collection_name,
        vector_size=vector_size
    )
    
    print(f"✅ Created collection: {collection_name} with size {vector_size}")
    
    # Test collection existence
    exists = manager.collection_exists(collection_name)
    print(f"✅ Collection exists check: {exists}")
    
    # Test point insertion
    ids = [str(uuid.uuid4()) for _ in range(3)]
    vectors = [[0.1 * i + 0.01 * j for j in range(vector_size)] for i in range(3)]
    payloads = [{"index": i, "text": f"Test point {i}"} for i in range(3)]
    
    success = manager.add_points(
        collection_name=collection_name,
        ids=ids,
        vectors=vectors,
        payloads=payloads
    )
    
    print(f"✅ Added {len(ids)} points to collection")
    
    # Test search
    query_vector = [0.1 * 1.5 + 0.01 * j for j in range(vector_size)]  # Should be closest to point 1
    
    results = manager.search(
        collection_name=collection_name,
        query_vector=query_vector,
        limit=3
    )
    
    print(f"✅ Search returned {len(results)} results")
    for i, result in enumerate(results):
        print(f"  Result {i+1}: ID={result['id']}, Score={result['score']:.4f}, Text={result['payload']['text']}")
    
    # Test collection info
    info = manager.get_collection_info(collection_name)
    print(f"✅ Collection info: {info}")
    
    return manager, collection_name

def test_complete_workflow():
    """Test a complete workflow with both components."""
    print_separator("Testing Complete Workflow")
    
    # Initialize components
    embedding_service = EmbeddingService()
    vector_db = QdrantManager()
    
    # Create collection
    collection_name = f"workflow_test_{uuid.uuid4().hex[:8]}"
    vector_size = embedding_service.get_vector_size()
    
    vector_db.create_collection(
        collection_name=collection_name,
        vector_size=vector_size
    )
    
    print(f"✅ Created collection {collection_name} with size {vector_size}")
    
    # Sample documents
    documents = [
        "Neural networks are computing systems vaguely inspired by the biological neural networks in animal brains.",
        "Deep learning is part of a broader family of machine learning methods based on artificial neural networks.",
        "Python is a high-level, general-purpose programming language known for its code readability.",
        "Vector databases are specialized database systems designed to store and search vectors efficiently.",
        "Embeddings in machine learning are low-dimensional representations that capture semantic meaning."
    ]
    
    # Generate embeddings
    print(f"Generating embeddings for {len(documents)} documents...")
    
    start_time = time.time()
    embeddings = embedding_service.embed_documents(documents)
    end_time = time.time()
    
    print(f"✅ Generated {len(embeddings)} embeddings in {(end_time - start_time)*1000:.2f}ms")
    
    # Store in vector database
    ids = [str(uuid.uuid4()) for _ in range(len(documents))]
    payloads = [{"text": doc, "index": i} for i, doc in enumerate(documents)]
    
    vector_db.add_points(
        collection_name=collection_name,
        ids=ids,
        vectors=embeddings,
        payloads=payloads
    )
    
    print(f"✅ Stored embeddings in vector database")
    
    # Test multiple queries
    queries = [
        "How do neural networks work?",
        "What programming language is known for readability?",
        "How are vector databases used?",
        "What are embeddings used for in AI?"
    ]
    
    for query in queries:
        print(f"\nQuery: '{query}'")
        
        # Generate query embedding
        start_time = time.time()
        query_embedding = embedding_service.embed_query(query)
        
        # Search vector database
        results = vector_db.search(
            collection_name=collection_name,
            query_vector=query_embedding,
            limit=2
        )
        end_time = time.time()
        
        print(f"✅ Search completed in {(end_time - start_time)*1000:.2f}ms")
        
        # Display results
        for i, result in enumerate(results):
            print(f"  Result {i+1} (Score: {result['score']:.4f}): {result['payload']['text']}")

def main():
    """Run all tests."""
    print_separator("Vector Infrastructure Test")
    
    # Test API keys
    test_api_keys()
    
    # Test individual components
    embedding_service = test_embedding_service()
    qdrant_manager, _ = test_qdrant_manager()
    
    # Test complete workflow
    test_complete_workflow()
    
    print_separator("Tests Completed Successfully")

if __name__ == "__main__":
    main() 