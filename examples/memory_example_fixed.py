#!/usr/bin/env python3
"""
Example usage of the memory system.

This script demonstrates how to use the refactored memory system,
including both context memory and long-term memory with different backends.
"""

import os
import sys
from datetime import datetime

# Add the project root to the path to ensure imports work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from agents.memory.factory import create_memory_manager
    from agents.memory.memory_manager import MemoryManager
    from agents.memory.context_memory import ContextMemory
    from agents.memory.stores.in_memory_store import InMemoryStore
except ImportError as e:
    print(f"ERROR: Unable to import memory modules: {e}")
    sys.exit(1)

# We'll use a dummy embedding service since the real one might be in a different location
class DummyEmbeddingService:
    """A dummy embedding service for demonstration purposes."""
    
    def __init__(self):
        pass
        
    def embed_query(self, text):
        """Return a dummy embedding for a query."""
        return [0.1] * 1024
        
    def embed_documents(self, documents):
        """Return dummy embeddings for documents."""
        return [[0.1] * 1024 for _ in documents]
        
    def get_vector_size(self):
        """Return the vector size."""
        return 1024


def main():
    """Run the memory example."""
    print("Memory System Example")
    print("---------------------------------\n")
    
    # Create an embedding service for vector memory
    embedding_service = DummyEmbeddingService()
    
    # Example 1: Simple in-memory memory manager
    print("\n=== Example 1: In-memory context memory ===\n")
    
    # Create a memory manager with default in-memory storage
    try:
        # Create memory manager using the factory function
        memory_manager = create_memory_manager(
            max_history=20,
            session_id="user123"
        )
        print("Successfully created memory manager using factory function")
    except Exception as e:
        print(f"Error creating memory manager with factory: {e}")
        # Fall back to direct creation
        try:
            context_memory = ContextMemory(store=InMemoryStore())
            memory_manager = MemoryManager(context_memory=context_memory)
            print("Successfully created memory manager directly")
        except Exception as e:
            print(f"ERROR: Could not create memory components: {e}")
            return
    
    # Add items to context
    memory_manager.set_context(
        "message",
        "Hello, world!"
    )
    
    memory_manager.set_context(
        "user_info",
        {"name": "Alice", "preferences": ["AI", "Python"]}
    )
    
    # Add a message to the conversation history
    message = {
        "role": "user",
        "content": "Tell me about memory systems",
        "timestamp": datetime.now().isoformat()
    }
    memory_manager.add(message)
    
    # Add assistant response
    response = {
        "role": "assistant", 
        "content": "Memory systems allow agents to store and retrieve information over time.",
        "timestamp": datetime.now().isoformat()
    }
    memory_manager.add(response)
    
    # Get and display context memory
    print("\nContext Values:")
    print(memory_manager.get_context())
    
    # Get and display conversation history
    print("\nConversation History:")
    print(memory_manager.get_formatted_history())
    
    print("\nConversation History (compact format):")
    print(memory_manager.get_formatted_history("compact"))
    
    print("\nMemory example completed successfully.")


if __name__ == "__main__":
    main() 