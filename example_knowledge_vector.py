#!/usr/bin/env python
"""
Example demonstrating the high-level knowledge module API with Cohere embeddings:
1. Create a vector knowledge base
2. Add sample text data
3. Perform semantic search queries

Usage:
    python example_knowledge_vector.py --api_key your_cohere_api_key
    # Or use an API key from the environment:
    # export COHERE_API_KEY=your_cohere_api_key
    # python example_knowledge_vector.py
"""

import os
import json
import argparse
from dotenv import load_dotenv

# Load environment variables (COHERE_API_KEY and QDRANT_API_KEY if using cloud)
load_dotenv()

# Import the knowledge module factory
from knowledge.factory import KnowledgeBaseFactory


def pretty_print(obj):
    """Pretty print an object."""
    if isinstance(obj, (dict, list)):
        print(json.dumps(obj, indent=2, default=str))
    else:
        print(obj)


def main(api_key=None, debug=False):
    # Create a vector knowledge base with Cohere embeddings
    config = {
        "collection_name": "knowledge_example",
        "embedding_model": "embed-english-v3.0",  # Cohere's state-of-the-art model
        "persist": True,  # Save to disk
        "persistence_path": os.path.join(os.getcwd(), "knowledge_data")  # Save in knowledge_data directory
    }
    
    # Add API key if provided
    if api_key:
        config["cohere_api_key"] = api_key
    
    kb = KnowledgeBaseFactory.create_knowledge_base("vector", config)
    
    print(f"Created vector knowledge base with model: {kb.embedding_model_name}")
    
    # Sample text documents to add
    sample_texts = [
        {
            "text": "Python is a high-level programming language known for its readability. It features a clean syntax, dynamic typing, and garbage collection, making it easy to learn and use effectively.",
            "metadata": {
                "title": "Python Programming",
                "category": "Programming Languages"
            }
        },
        {
            "text": "Machine learning is a field of study that gives computers the ability to learn without being explicitly programmed. It focuses on algorithms that can learn from and make predictions based on data.",
            "metadata": {
                "title": "Machine Learning",
                "category": "Artificial Intelligence"
            }
        },
        {
            "text": "Natural language processing (NLP) combines linguistics, computer science, and AI to help computers understand, interpret, and generate human language in valuable ways.",
            "metadata": {
                "title": "Natural Language Processing",
                "category": "Artificial Intelligence"
            }
        },
        {
            "text": "Vector databases are specialized systems designed for storing and searching vector embeddings efficiently. They're optimized for similarity search operations and used in recommendation systems, semantic search, and AI applications.",
            "metadata": {
                "title": "Vector Databases",
                "category": "Databases"
            }
        },
        {
            "text": "Embeddings are numerical representations of data in a continuous vector space. In NLP, embeddings capture semantic meaning by placing similar words or concepts close together in the vector space.",
            "metadata": {
                "title": "Embeddings",
                "category": "Machine Learning"
            }
        }
    ]
    
    # Add texts to the knowledge base
    print("Adding sample texts to knowledge base...")
    text_ids = []
    for sample in sample_texts:
        text_id = kb.add_text(sample["text"], sample["metadata"])
        text_ids.append(text_id)
        print(f"Added text: {sample['metadata']['title']} (ID: {text_id})")
    
    # Get knowledge base stats
    stats = kb.get_stats()
    print("\nKnowledge base statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Number of texts we added
    print(f"Added text documents: {len(text_ids)}")
    
    # Perform semantic searches
    queries = [
        "programming languages with clean syntax",
        "how do computers learn from data",
        "systems that understand human language",
        "storing vector embeddings efficiently",
        "representing words as vectors"
    ]
    
    print("\nPerforming semantic searches:")
    for query in queries:
        print(f"\nQuery: '{query}'")
        
        # Perform semantic search
        results = kb.semantic_search(query, limit=2)
        
        # Show detailed structure of first result if debug mode
        if debug and results:
            print("\nFirst result structure:")
            pretty_print(results[0])
        
        # Display results
        print("Results:")
        for i, result in enumerate(results):
            # Extract the title and text from the result
            score = result.get('score', 0.0)
            
            # Extract title - directly from properties or from payload
            title = "Unknown"
            if 'properties' in result and 'title' in result['properties']:
                title = result['properties']['title']
            elif 'payload' in result and 'properties' in result['payload'] and 'title' in result['payload']['properties']:
                title = result['payload']['properties']['title']
            
            # Extract content - directly from result or from payload
            text = None
            if 'content' in result:
                text = result['content']
            elif 'payload' in result and 'content' in result['payload']:
                text = result['payload']['content']
            
            print(f"  {i+1}. Score: {score:.4f} - {title}")
            
            if text:
                snippet = text[:100] + "..." if len(text) > 100 else text
                print(f"     Snippet: {snippet}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Knowledge vector example")
    parser.add_argument("--api_key", help="Cohere API key")
    parser.add_argument("--debug", action="store_true", help="Show debug information")
    args = parser.parse_args()
    
    main(api_key=args.api_key, debug=args.debug) 