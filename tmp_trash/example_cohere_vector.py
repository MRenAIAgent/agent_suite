#!/usr/bin/env python
"""
Simple example demonstrating how to:
1. Generate embeddings for sample text
2. Store them in a vector database
3. Retrieve vectors using a semantic query

Usage:
    python example_cohere_vector.py --api_key your_cohere_api_key
    # Or use an API key from the environment:
    # export COHERE_API_KEY=your_cohere_api_key
    # python example_cohere_vector.py
"""

import os
import argparse
import uuid
from dotenv import load_dotenv

# Load environment variables (COHERE_API_KEY and QDRANT_API_KEY if using cloud)
load_dotenv()

# Import the necessary components
from agent_suite.vector.embedding_service import EmbeddingService
from agent_suite.vector.qdrant_client import QdrantManager


def main(api_key=None):
    # Initialize the embedding service
    embedding_service = EmbeddingService(
        model_name="embed-english-v3.0",  # Cohere's state-of-the-art embedding model
        input_type="search_document",  # Optimized for document indexing
        api_key=api_key  # Use provided key or from environment
    )
    
    # Initialize the vector database
    # For this example, we'll use in-memory storage
    vector_db = QdrantManager()
    
    # Create a collection to store our vectors
    collection_name = "example_collection"
    vector_size = embedding_service.get_vector_size()  # 1024 for embed-english-v3.0
    
    vector_db.create_collection(
        collection_name=collection_name,
        vector_size=vector_size,
        distance="cosine",  # Now handled by the QdrantManager
        recreate=True  # Start with a fresh collection
    )
    
    print(f"Created collection '{collection_name}' with vector size {vector_size}")
    
    # Sample texts to embed
    sample_texts = [
        "Python is a high-level programming language known for its readability.",
        "Machine learning is a field of study that gives computers the ability to learn without being explicitly programmed.",
        "Natural language processing (NLP) is a subfield of linguistics, computer science, and AI concerned with interactions between computers and human language.",
        "Vector databases are specialized database systems designed for storing and searching high-dimensional vectors efficiently.",
        "Cohere provides state-of-the-art embedding models for text representation."
    ]
    
    # Generate embeddings for all texts at once
    print("Generating embeddings for sample texts...")
    embeddings = embedding_service.embed_documents(sample_texts)
    
    # Create IDs and metadata for each text - using UUIDs
    ids = [str(uuid.uuid4()) for _ in range(len(sample_texts))]
    payloads = [{"text": text, "category": "sample"} for text in sample_texts]
    
    # Store the embeddings in the vector database
    print("Storing embeddings in vector database...")
    vector_db.add_points(
        collection_name=collection_name,
        ids=ids,
        vectors=embeddings,
        payloads=payloads
    )
    
    # Now let's perform some semantic searches
    queries = [
        "programming languages that are easy to read",
        "AI systems that learn from data",
        "how computers understand human language",
        "database for storing embeddings",
        "best embedding models for text"
    ]
    
    print("\nPerforming semantic searches:")
    for query in queries:
        print(f"\nQuery: '{query}'")
        
        # Generate embedding for the query
        # Note: We use embed_query which is optimized for search queries
        query_embedding = embedding_service.embed_query(query)
        
        # Search for similar vectors
        results = vector_db.search(
            collection_name=collection_name,
            query_vector=query_embedding,
            limit=3  # Get top 3 results
        )
        
        # Display results
        print("Results:")
        for i, result in enumerate(results):
            print(f"  {i+1}. Score: {result['score']:.4f} - {result['payload']['text']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vector embedding example")
    parser.add_argument("--api_key", help="Cohere API key")
    args = parser.parse_args()
    
    main(api_key=args.api_key) 