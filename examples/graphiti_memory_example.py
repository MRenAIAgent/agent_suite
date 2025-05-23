#!/usr/bin/env python3
"""
Example of using Graphiti-based memory for advanced relationship storage.

This example demonstrates how to create and use a memory manager 
that stores conversation history in a graph database, enabling
more advanced relationship queries and knowledge extraction.

Requirements:
- Graphiti package installed (pip install graphiti)
"""

import time
from agents.memory.factory import create_graphiti_memory_manager


def print_separator(title):
    """Print a section separator."""
    print("\n" + "=" * 50)
    print(f"  {title}")
    print("=" * 50)


def main():
    """Demonstrate Graphiti-backed memory usage."""
    print_separator("Graphiti Memory Example")
    
    # Create a memory manager with Graphiti backing for long-term memory
    memory = create_graphiti_memory_manager(
        max_history=10,  # Keep 10 messages in context
        db_path="./agent_memory.graph",  # Path to graph database
        namespace="example_agent",
        context_store_type="in_memory"  # Use in-memory store for context
    )
    
    # Check if we have existing history (persistent across runs)
    history = memory.get_history()
    if history:
        print_separator("Existing Messages")
        for msg in history:
            print(f"- {msg['role']}: {msg['content']}")
    else:
        print("\nNo existing messages found.")
    
    # Add messages to the conversation with relationships and topics
    print_separator("Adding New Messages")
    
    messages = [
        {"role": "user", "content": "Hello, I'm interested in learning about machine learning and neural networks."},
        {"role": "assistant", "content": "Great! Machine learning is a branch of artificial intelligence that uses data to train models."},
        {"role": "user", "content": "How do neural networks relate to deep learning?"},
        {"role": "assistant", "content": "Deep learning is a subset of machine learning that uses neural networks with multiple layers."},
        {"role": "user", "content": "What are transformers in deep learning?"},
        {"role": "assistant", "content": "Transformers are a type of neural network architecture designed for sequential data like text."},
        {"role": "user", "content": "Can transformers be used for image processing too?"},
        {"role": "assistant", "content": "Yes, Vision Transformers (ViT) apply the transformer architecture to image processing tasks."}
    ]
    
    for msg in messages:
        memory.add(msg)
        print(f"Added: {msg['role']}: {msg['content']}")
        time.sleep(0.1)  # Small delay to ensure different timestamps
    
    # Display current history
    print_separator("Current Conversation History")
    history = memory.get_history()
    for msg in history:
        print(f"- {msg['role']}: {msg['content']}")
    
    # Demonstrate search with semantic context
    print_separator("Semantic Search Examples")
    
    # Search by explicit topic
    query = "neural networks"
    print(f"\nSearching for: '{query}'")
    results = memory.search(query)
    for i, msg in enumerate(results, 1):
        print(f"{i}. [{msg.get('match_type', 'unknown')}] {msg['role']}: {msg['content']}")
    
    # Search by related topic
    query = "transformers"
    print(f"\nSearching for: '{query}'")
    results = memory.search(query)
    for i, msg in enumerate(results, 1):
        print(f"{i}. [{msg.get('match_type', 'unknown')}] {msg['role']}: {msg['content']}")
    
    # Search by broader domain
    query = "machine learning"
    print(f"\nSearching for: '{query}'")
    results = memory.search(query)
    for i, msg in enumerate(results, 1):
        print(f"{i}. [{msg.get('match_type', 'unknown')}] {msg['role']}: {msg['content']}")
    
    print("\nThe graph database allows more sophisticated searches and can extract relationships")
    print("between concepts, building a knowledge graph from the conversation.")
    print("\nThe graph database is stored at './agent_memory.graph' and persists between runs.")
    

if __name__ == "__main__":
    main() 