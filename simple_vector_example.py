#!/usr/bin/env python
"""
Simple example demonstrating vector embeddings with proper API key handling.

This example will:
1. Create/read .env file for API keys
2. Generate embeddings using Cohere
3. Store them in Qdrant
4. Perform a semantic search

Usage:
    python simple_vector_example.py [--cohere_key YOUR_COHERE_KEY]
"""

import os
import uuid
import time
import argparse
from pathlib import Path
from dotenv import load_dotenv

# Import components
from agent_suite.vector.embedding_service import EmbeddingService
from agent_suite.vector.qdrant_client import QdrantManager

def setup_env_file(cohere_key=None):
    """Create or update .env file with API keys."""
    # Check if .env exists
    env_path = Path('.env')
    
    if env_path.exists():
        print(f"Found existing .env file at {env_path.absolute()}")
        # Load existing values first
        load_dotenv()
    else:
        print(f"Creating new .env file at {env_path.absolute()}")
        
    # If key provided as argument, use it
    if cohere_key:
        os.environ["COHERE_API_KEY"] = cohere_key
        # Also write to .env file
        with open(env_path, 'w') as f:
            f.write(f"COHERE_API_KEY={cohere_key}\n")
            # Add a commented line for Qdrant
            f.write("# QDRANT_API_KEY=your_qdrant_api_key_here\n")
        print(f"Updated .env file with provided Cohere API key")
    
    # Reload env vars
    load_dotenv()
    
def check_api_keys():
    """Check if required API keys are available."""
    cohere_key = os.getenv("COHERE_API_KEY")
    
    if not cohere_key:
        print("WARNING: No Cohere API key found in .env file")
        print("Please create a .env file in the current directory with:")
        print("COHERE_API_KEY=your_api_key_here")
        print("You can get a key at: https://dashboard.cohere.com/api-keys")
        return False
        
    if cohere_key == "demo" or cohere_key == "your_cohere_api_key_here":
        print("WARNING: Using placeholder Cohere API key. Please update with a real key.")
        return False
        
    return True

def main(cohere_key=None):
    """Run the vector embedding example."""
    # Set up environment
    setup_env_file(cohere_key)
    
    # Check for API keys
    api_keys_valid = check_api_keys()
    if not api_keys_valid:
        print("\nContinuing with example, but embeddings will be dummy vectors...")
    
    # Get API keys from environment
    cohere_api_key = os.getenv("COHERE_API_KEY")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    
    print("Initializing embedding service...")
    embedding_service = EmbeddingService(
        model_name="embed-english-v3.0",
        api_key=cohere_api_key
    )
    
    print("Initializing vector database (in-memory)...")
    vector_db = QdrantManager(api_key=qdrant_api_key)
    
    # Create a collection
    collection_name = "quick_example"
    vector_size = embedding_service.get_vector_size()
    print(f"Creating collection '{collection_name}' with vector size {vector_size}...")
    
    vector_db.create_collection(
        collection_name=collection_name,
        vector_size=vector_size,
        distance="cosine",
        recreate=True
    )
    
    # Sample documents
    documents = [
        "Python is a programming language that lets you work quickly and integrate systems effectively.",
        "Embeddings are vector representations of text that capture semantic meaning.",
        "Vector databases store high-dimensional vectors for efficient similarity search.",
        "Semantic search uses meaning rather than keywords to find relevant information.",
        "Artificial intelligence is transforming how we interact with information."
    ]
    
    # Create embeddings
    print("Generating embeddings...")
    start_time = time.time()
    embeddings = embedding_service.embed_documents(documents)
    end_time = time.time()
    print(f"Generated {len(embeddings)} embeddings in {end_time - start_time:.2f} seconds")
    
    # Store in vector database
    print("Storing in vector database...")
    ids = [str(uuid.uuid4()) for _ in range(len(documents))]
    payloads = [{"text": doc, "index": i} for i, doc in enumerate(documents)]
    
    vector_db.add_points(
        collection_name=collection_name,
        ids=ids,
        vectors=embeddings,
        payloads=payloads
    )
    
    # Perform a search
    print("\nPerforming semantic search...")
    query = "How do computers understand text meaning?"
    
    print(f"Query: '{query}'")
    start_time = time.time()
    query_embedding = embedding_service.embed_query(query)
    
    results = vector_db.search(
        collection_name=collection_name,
        query_vector=query_embedding,
        limit=3
    )
    end_time = time.time()
    
    # Display results
    print(f"Search completed in {end_time - start_time:.2f} seconds")
    print("Results:")
    
    for i, result in enumerate(results):
        print(f"  {i+1}. Score: {result['score']:.4f} - {result['payload']['text']}")
    
    print("\nExample completed successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vector embedding example")
    parser.add_argument("--cohere_key", help="Your Cohere API key")
    args = parser.parse_args()
    
    main(cohere_key=args.cohere_key) 