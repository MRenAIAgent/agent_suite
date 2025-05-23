#!/usr/bin/env python
"""Example of using the storage service with different backends.

This example demonstrates how to create and use the storage service
with vector, graph, and document backends.
"""

import asyncio
import json
from typing import Dict, Any

from agents.storage import (
    # Factory
    create_storage_service,
    
    # Available backend checks
    QDRANT_AVAILABLE,
    NEO4J_AVAILABLE,
    MONGODB_AVAILABLE,
    ELASTICSEARCH_AVAILABLE,
)


async def vector_storage_example(service):
    """Example of using vector storage."""
    print("\n=== Vector Storage Example ===")
    
    # Get the vector storage backend
    vector_store = service.get_backend("vector")
    if not vector_store:
        print("Vector storage backend not available")
        return
    
    # Store vectors with metadata
    vector1 = [0.1, 0.2, 0.3, 0.4]
    vector2 = [0.5, 0.6, 0.7, 0.8]
    
    key1 = await vector_store.store("item1", {
        "vector": vector1,
        "metadata": {
            "text": "This is a test document about AI",
            "tags": ["ai", "test"]
        }
    })
    
    key2 = await vector_store.store("item2", {
        "vector": vector2,
        "metadata": {
            "text": "This is another document about machine learning",
            "tags": ["ml", "test"]
        }
    })
    
    print(f"Stored vectors with keys: {key1}, {key2}")
    
    # Retrieve a stored item
    item = await vector_store.retrieve(key1)
    print(f"Retrieved item: {item}")
    
    # Search by vector
    results = await vector_store.search_by_vector(
        vector1, 
        limit=5, 
        score_threshold=0.7
    )
    print(f"Found {len(results)} similar vectors")
    for i, result in enumerate(results):
        print(f"  Result {i+1}: score={result.get('score')}, metadata={result.get('metadata')}")
    
    # Search by metadata
    results = await vector_store.search_by_metadata(
        {"tags": ["test"]},
        limit=5
    )
    print(f"Found {len(results)} items with matching metadata")


async def graph_storage_example(service):
    """Example of using graph storage."""
    print("\n=== Graph Storage Example ===")
    
    # Get the graph storage backend
    graph_store = service.get_backend("graph")
    if not graph_store:
        print("Graph storage backend not available")
        return
    
    # Create nodes
    person1_id = await graph_store.create_node(
        "Person",
        {
            "name": "Alice",
            "age": 30,
            "interests": ["AI", "Programming"]
        }
    )
    
    person2_id = await graph_store.create_node(
        "Person",
        {
            "name": "Bob",
            "age": 25,
            "interests": ["Machine Learning", "Data Science"]
        }
    )
    
    topic_id = await graph_store.create_node(
        "Topic",
        {
            "name": "Artificial Intelligence",
            "level": "Advanced"
        }
    )
    
    print(f"Created nodes: {person1_id}, {person2_id}, {topic_id}")
    
    # Create relationships
    rel1_id = await graph_store.create_relationship(
        person1_id,
        topic_id,
        "INTERESTED_IN",
        {
            "since": "2020",
            "level": "Expert"
        }
    )
    
    rel2_id = await graph_store.create_relationship(
        person2_id,
        topic_id,
        "LEARNING",
        {
            "since": "2022",
            "level": "Beginner"
        }
    )
    
    rel3_id = await graph_store.create_relationship(
        person1_id,
        person2_id,
        "KNOWS",
        {
            "since": "2021",
            "relationship": "Colleague"
        }
    )
    
    print(f"Created relationships: {rel1_id}, {rel2_id}, {rel3_id}")
    
    # Get nodes
    nodes = await graph_store.get_nodes("Person")
    print(f"Person nodes: {len(nodes)}")
    
    # Get relationships
    rels = await graph_store.get_relationships(source_id=person1_id)
    print(f"Relationships from Alice: {len(rels)}")
    
    # Traverse the graph
    traversal = await graph_store.traverse(
        person1_id, 
        relationship_types=["INTERESTED_IN", "KNOWS"], 
        max_depth=2
    )
    print(f"Traversal result paths: {len(traversal.get('paths', []))}")


async def document_storage_example(service):
    """Example of using document storage."""
    print("\n=== Document Storage Example ===")
    
    # Get the document storage backend
    doc_store = service.get_backend("document")
    if not doc_store:
        print("Document storage backend not available")
        return
    
    # Create a collection index
    collection = "articles"
    await doc_store.create_index(
        collection,
        {
            "title": "text",
            "content": "text",
            "author": "keyword",
            "publishedAt": "date",
            "tags": "keyword"
        }
    )
    
    # Insert documents
    docs = [
        {
            "title": "Introduction to AI",
            "content": "Artificial intelligence is the simulation of human intelligence by machines.",
            "author": "Alice Smith",
            "publishedAt": "2023-01-15",
            "tags": ["AI", "technology", "beginner"]
        },
        {
            "title": "Machine Learning Basics",
            "content": "Machine learning is a subset of AI that enables systems to learn from data.",
            "author": "Bob Johnson",
            "publishedAt": "2023-02-10",
            "tags": ["ML", "AI", "data science"]
        },
        {
            "title": "Neural Networks Explained",
            "content": "Neural networks are computing systems inspired by biological neural networks.",
            "author": "Alice Smith",
            "publishedAt": "2023-03-05",
            "tags": ["neural networks", "deep learning", "AI"]
        }
    ]
    
    doc_ids = await doc_store.insert_documents(collection, docs)
    print(f"Inserted documents with IDs: {doc_ids}")
    
    # Find documents
    results = await doc_store.find_documents(
        collection,
        {"author": "Alice Smith"},
        sort={"publishedAt": -1}
    )
    print(f"Found {len(results)} documents by Alice Smith")
    
    # Text search
    search_results = await doc_store.text_search(
        collection,
        "neural networks learning",
        fields=["title", "content"]
    )
    print(f"Found {len(search_results)} documents matching text search")
    for i, doc in enumerate(search_results):
        print(f"  Result {i+1}: title={doc.get('title')}, score={doc.get('score', 0)}")
    
    # Aggregation
    if hasattr(doc_store, "aggregate"):
        pipeline = [
            {"$group": {"_id": "$author", "count": {"$sum": 1}}}
        ]
        agg_results = await doc_store.aggregate(collection, pipeline)
        print(f"Aggregation results: {agg_results}")


async def main():
    """Run the storage example."""
    print("Storage Service Example")
    print("======================")
    
    # Check which backends are available
    print("\nAvailable backends:")
    print(f"  Qdrant: {QDRANT_AVAILABLE}")
    print(f"  Neo4j: {NEO4J_AVAILABLE}")
    print(f"  MongoDB: {MONGODB_AVAILABLE}")
    print(f"  Elasticsearch: {ELASTICSEARCH_AVAILABLE}")
    
    # Configure backends
    backends = {}
    
    if QDRANT_AVAILABLE:
        backends["vector"] = {
            "type": "qdrant",
            "config": {
                "collection_name": "example",
                "vector_size": 4,  # For this example we're using tiny vectors
                "host": "localhost",
                "port": 6333
            }
        }
    
    if NEO4J_AVAILABLE:
        backends["graph"] = {
            "type": "neo4j",
            "config": {
                "uri": "bolt://localhost:7687",
                "username": "neo4j",
                "password": "password",  # Replace with your password
                "namespace": "example"
            }
        }
    
    if MONGODB_AVAILABLE:
        backends["document"] = {
            "type": "mongodb",
            "config": {
                "uri": "mongodb://localhost:27017",
                "database": "example",
                "namespace": "example"
            }
        }
    elif ELASTICSEARCH_AVAILABLE:
        backends["document"] = {
            "type": "elasticsearch",
            "config": {
                "hosts": "http://localhost:9200",
                "namespace": "example"
            }
        }
    
    # Create the storage service
    service = create_storage_service(backends)
    
    # Run examples for available backends
    if "vector" in backends:
        try:
            await vector_storage_example(service)
        except Exception as e:
            print(f"Error in vector storage example: {e}")
    
    if "graph" in backends:
        try:
            await graph_storage_example(service)
        except Exception as e:
            print(f"Error in graph storage example: {e}")
    
    if "document" in backends:
        try:
            await document_storage_example(service)
        except Exception as e:
            print(f"Error in document storage example: {e}")
    
    # Close all backends
    await service.close()
    print("\nStorage service closed")


if __name__ == "__main__":
    asyncio.run(main()) 