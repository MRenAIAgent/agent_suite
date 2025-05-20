#!/usr/bin/env python
"""
Example script demonstrating the use of Cohere embeddings with the knowledge base.

This script shows how to:
1. Create a knowledge base with Cohere embeddings
2. Add entities and text
3. Perform semantic search
4. Compare results with the previous embedding model
"""

import os
import json
import time
from typing import Dict, Any, List

from knowledge.factory import KnowledgeBaseFactory
from knowledge.base import KnowledgeBase


def print_separator(title: str = None):
    """Print a separator line with optional title."""
    width = 80
    if title:
        print("\n" + "=" * 10 + f" {title} " + "=" * (width - 12 - len(title)) + "\n")
    else:
        print("\n" + "=" * width + "\n")


def pretty_print(data: Any):
    """Pretty print data structures."""
    if isinstance(data, (dict, list)):
        print(json.dumps(data, indent=2))
    else:
        print(data)


def demo_cohere_vector_database():
    """Demonstrate the vector database with Cohere embeddings."""
    print_separator("Cohere Vector Database Demo")
    
    # Create a knowledge base instance with Cohere embeddings
    knowledge_dir = os.path.join(os.getcwd(), "knowledge_data")
    os.makedirs(knowledge_dir, exist_ok=True)
    
    kb = KnowledgeBaseFactory.create_knowledge_base("vector", {
        "collection_name": "demo_cohere_vector",
        "embedding_model": "embed-english-v3.0",  # Cohere's latest English embedding model
        "persist": True,
        "persistence_path": knowledge_dir
    })
    
    # Add text documents
    print("Adding text documents...")
    
    python_doc = """Python is a high-level, interpreted programming language known for its readability and simplicity.
    It supports multiple programming paradigms including procedural, object-oriented, and functional programming.
    Python's design philosophy emphasizes code readability with its use of significant whitespace.
    It provides constructs that enable clear programming on both small and large scales."""
    
    kb.add_text(python_doc, {
        "title": "Python Overview",
        "category": "Programming Languages"
    })
    
    ai_doc = """Artificial Intelligence (AI) is intelligence demonstrated by machines, as opposed to the natural 
    intelligence displayed by humans or animals. Leading AI textbooks define the field as the study of "intelligent 
    agents": any system that perceives its environment and takes actions that maximize its chance of achieving its goals.
    Machine learning is a subset of AI that involves training algorithms on data to make predictions or decisions."""
    
    kb.add_text(ai_doc, {
        "title": "AI Overview",
        "category": "Artificial Intelligence"
    })
    
    ml_doc = """Machine learning is a field of study in artificial intelligence concerned with the development of 
    algorithms and statistical models that computer systems use to perform tasks without explicit instructions, 
    relying instead on patterns and inference. It is seen as a subset of artificial intelligence.
    Machine learning approaches include supervised learning, unsupervised learning, and reinforcement learning."""
    
    kb.add_text(ml_doc, {
        "title": "Machine Learning Overview",
        "category": "Artificial Intelligence"
    })
    
    nlp_doc = """Natural Language Processing (NLP) is a subfield of linguistics, computer science, and artificial 
    intelligence concerned with the interactions between computers and human language. It involves the ability of 
    a computer to understand, analyze, manipulate, and potentially generate human language. NLP tasks include 
    sentiment analysis, language translation, speech recognition, and text summarization."""
    
    kb.add_text(nlp_doc, {
        "title": "NLP Overview",
        "category": "Natural Language Processing"
    })
    
    llm_doc = """Large Language Models (LLMs) are advanced AI systems trained on vast amounts of text data to 
    understand and generate human-like text. These models utilize transformer-based neural network architectures 
    and have billions of parameters. LLMs like GPT-4, Claude, and LLaMA can perform a wide range of natural language 
    tasks including translation, summarization, question answering, and creative content generation."""
    
    kb.add_text(llm_doc, {
        "title": "Large Language Models Overview",
        "category": "Artificial Intelligence"
    })
    
    # Perform semantic search
    print("\nSearching for 'programming language with readable syntax':")
    start_time = time.time()
    results = kb.semantic_search("programming language with readable syntax")
    end_time = time.time()
    pretty_print(results)
    print(f"Search time: {end_time - start_time:.4f} seconds")
    
    print("\nSearching for 'AI systems that can understand and generate language':")
    start_time = time.time()
    results = kb.semantic_search("AI systems that can understand and generate language")
    end_time = time.time()
    pretty_print(results)
    print(f"Search time: {end_time - start_time:.4f} seconds")
    
    print("\nSearching for 'differences between AI and machine learning':")
    start_time = time.time()
    results = kb.semantic_search("differences between AI and machine learning")
    end_time = time.time()
    pretty_print(results)
    print(f"Search time: {end_time - start_time:.4f} seconds")
    
    # Get stats
    print("\nVector Database Statistics:")
    stats = kb.get_stats()
    pretty_print(stats)


def demo_cohere_vs_sentence_transformer():
    """Compare Cohere embeddings with SentenceTransformers."""
    print_separator("Embedding Model Comparison")
    
    knowledge_dir = os.path.join(os.getcwd(), "knowledge_data")
    
    # Create two knowledge bases: one with Cohere and one with sentence-transformers
    print("Creating two vector databases with different embedding models...")
    
    # We'll use the older vector_db.py implementation for this
    from knowledge.vector.vector_db import QdrantVectorDatabase as STVectorDB
    
    st_kb = STVectorDB(
        collection_name="compare_sentence_transformers",
        embedding_model="all-MiniLM-L6-v2",
        persist=True,
        persistence_path=knowledge_dir
    )
    
    cohere_kb = KnowledgeBaseFactory.create_knowledge_base("vector", {
        "collection_name": "compare_cohere",
        "embedding_model": "embed-english-v3.0",
        "persist": True,
        "persistence_path": knowledge_dir
    })
    
    # Add the same documents to both
    print("Adding identical texts to both databases...")
    
    test_documents = [
        {
            "text": """Knowledge graphs are a type of database that uses a graph-structured data model to store 
            interconnected descriptions of entities, with nodes representing objects and edges defining relationships 
            between them. They allow for semantic queries and inferencing over complex relationships.""",
            "metadata": {"title": "Knowledge Graph", "category": "Data Structures"}
        },
        {
            "text": """Vector databases are specialized database systems designed to store and search high-dimensional 
            vectors efficiently. They're optimized for similarity search operations like nearest neighbor search, which 
            is crucial for applications like image recognition and semantic search.""",
            "metadata": {"title": "Vector Database", "category": "Data Structures"}
        },
        {
            "text": """Embeddings in machine learning are vector representations of data in a continuous space where 
            similar items are closer together. They capture semantic meaning and allow machines to understand the 
            relationships between different pieces of data, enabling operations like semantic search.""",
            "metadata": {"title": "Embeddings", "category": "Machine Learning"}
        }
    ]
    
    for doc in test_documents:
        st_kb.add_text(doc["text"], doc["metadata"])
        cohere_kb.add_text(doc["text"], doc["metadata"])
    
    # Define test queries
    test_queries = [
        "How do databases store relationships between entities?",
        "What is the best way to find similar vectors?",
        "How do AI systems understand semantic meaning?",
        "Database for semantic search operations",
        "Representing objects and relationships in a database"
    ]
    
    # Test semantic search with each query
    for i, query in enumerate(test_queries):
        print(f"\nQuery {i+1}: {query}")
        
        print("\nSentence Transformers results:")
        st_start = time.time()
        st_results = st_kb.semantic_search(query, limit=2)
        st_end = time.time()
        for j, result in enumerate(st_results):
            print(f"Result {j+1}: {result['score']:.4f} - {result.get('content', '')[:100]}...")
        print(f"Search time: {st_end - st_start:.4f} seconds")
        
        print("\nCohere results:")
        cohere_start = time.time()
        cohere_results = cohere_kb.semantic_search(query, limit=2)
        cohere_end = time.time()
        for j, result in enumerate(cohere_results):
            print(f"Result {j+1}: {result['score']:.4f} - {result.get('content', '')[:100]}...")
        print(f"Search time: {cohere_end - cohere_start:.4f} seconds")


def main():
    """Run the demo functions."""
    print("Cohere Embedding Knowledge Base Demo")
    print("====================================")
    
    demo_cohere_vector_database()
    
    try:
        # This requires the older vector_db.py implementation to still be present
        demo_cohere_vs_sentence_transformer()
    except ImportError:
        print("\nSkipping comparison demo since the SentenceTransformers implementation is not available.")
    
    print("\nDemo complete!")


if __name__ == "__main__":
    main() 