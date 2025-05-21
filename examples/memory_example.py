#!/usr/bin/env python3
"""
Example usage of the agent_suite memory system.

This script demonstrates how to use the refactored memory system,
including both context memory and long-term memory with different backends.
"""

import os
from datetime import datetime

from agent_suite.memory.factory import MemoryFactory
from agent_suite.vector.embedding_service import EmbeddingService


def main():
    """Run the memory example."""
    print("Agent Suite Memory System Example")
    print("---------------------------------\n")
    
    # Create an embedding service for vector memory
    embedding_service = EmbeddingService()
    
    # Example 1: Simple in-memory context with Redis vector memory
    print("\n=== Example 1: In-memory context + Redis vector long-term memory ===\n")
    
    # Configuration for memory components
    context_config = {
        "type": "in_memory"
    }
    
    long_term_config = {
        "type": "redis_vector",
        "embedding_service": embedding_service,
        "redis_url": os.environ.get("REDIS_URL", "redis://localhost:6379"),
        "index_name": "example_memory",
        "prefix": "example:"
    }
    
    # Create a memory manager with user-specific memory
    user_id = "user123"
    memory_manager = MemoryFactory.create_memory_manager(
        context_config=context_config,
        long_term_config=long_term_config,
        user_id=user_id
    )
    
    # Add items to context memory
    memory_manager.add_to_context(
        "message",
        "Hello, world!",
        {"timestamp": datetime.now().isoformat()}
    )
    
    memory_manager.add_to_context(
        "user_info",
        {"name": "Alice", "preferences": ["AI", "Python"]},
        {"timestamp": datetime.now().isoformat()}
    )
    
    # Get and display context memory
    print("Context Memory (default format):")
    print(memory_manager.get_formatted_context())
    
    print("\nContext Memory (compact format):")
    print(memory_manager.get_formatted_context("compact"))
    
    # Store some information in long-term memory
    memory_id1 = memory_manager.store(
        "Python is a high-level programming language known for its readability and simplicity.",
        {"topic": "programming", "language": "python"}
    )
    
    memory_id2 = memory_manager.store(
        "TensorFlow and PyTorch are popular deep learning frameworks for Python.",
        {"topic": "deep learning", "frameworks": ["tensorflow", "pytorch"]}
    )
    
    memory_id3 = memory_manager.store(
        "Natural Language Processing (NLP) is a field of AI focused on the interaction between computers and human language.",
        {"topic": "ai", "field": "nlp"}
    )
    
    print(f"\nStored memories with IDs: {memory_id1}, {memory_id2}, {memory_id3}")
    
    # Query long-term memory
    query_result = memory_manager.query("What is Python programming language?", limit=2)
    print("\nQuery Results for 'What is Python programming language?':")
    for i, item in enumerate(query_result):
        print(f"Result {i+1} (score: {item.get('score', 'N/A'):.4f}):")
        print(f"  Text: {item.get('text', 'No text')}")
        print(f"  Metadata: {item.get('metadata', {})}")
    
    # Use remember to query and add to context
    memory_manager.remember("AI and deep learning", context_key="ai_info")
    print("\nAfter remembering 'AI and deep learning':")
    print(memory_manager.get_formatted_context())
    
    # Example 2: Redis context memory
    if os.environ.get("REDIS_URL"):
        print("\n\n=== Example 2: Redis context memory ===\n")
        
        # Create Redis context memory config
        redis_context_config = {
            "type": "redis",
            "redis_url": os.environ.get("REDIS_URL", "redis://localhost:6379"),
            "namespace": "example_context:",
            "ttl": 3600  # 1 hour
        }
        
        # Create a new memory manager
        redis_memory_manager = MemoryFactory.create_memory_manager(
            context_config=redis_context_config,
            long_term_config=long_term_config
        )
        
        # Add items to Redis context memory
        redis_memory_manager.add_to_context(
            "system_message",
            "You are a helpful assistant.",
            {"role": "system"}
        )
        
        redis_memory_manager.add_to_context(
            "conversation",
            {"role": "user", "content": "What can you tell me about Python?"},
            {"timestamp": datetime.now().isoformat()}
        )
        
        # Get and display Redis context memory
        print("Redis Context Memory:")
        print(redis_memory_manager.get_formatted_context())
        
        # Clear Redis context to clean up
        redis_memory_manager.clear_context()
    
    # Clean up memories from the first example
    print("\n\nCleaning up example memories...")
    memory_manager.delete(memory_id1)
    memory_manager.delete(memory_id2)
    memory_manager.delete(memory_id3)
    
    print("\nMemory example completed successfully.")


if __name__ == "__main__":
    main() 