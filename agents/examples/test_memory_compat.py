#!/usr/bin/env python3
"""
Self-contained minimal example demonstrating compatibility between
React agent and new memory system using only the minimal_memory module.
"""

import os
import sys
from datetime import datetime

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Import our minimal memory components
from agent_suite.memory.minimal_memory import (
    InMemoryContextMemory,
    ContextMemoryAdapter
)

# Import the needed components from agent_suite
from agents.memory_manager import MemoryManager as OldMemoryManager
from agents.memory.stores.store import Store


class CustomMemoryManager(OldMemoryManager):
    """Custom memory manager that uses our provided storage."""
    
    def __init__(self, storage):
        """Initialize with the provided storage."""
        super().__init__()
        self.store = storage  # Override the default store with our adapted storage


def main():
    """Run the minimal memory compatibility test."""
    print("=== Minimal Memory Compatibility Test ===\n")
    
    # Create new memory system
    context_memory = InMemoryContextMemory()
    print("Created new InMemoryContextMemory instance")
    
    # Add some test data to the memory
    context_memory.add("system_message", 
                    "You are a helpful assistant using the new memory system.", 
                    {"timestamp": datetime.now().isoformat()})
    context_memory.add("note", 
                     "The user is interested in learning about Python programming.", 
                     {"timestamp": datetime.now().isoformat()})
    print("Added initial messages to new memory")
    
    # Display the formatted memory content
    print("\nNew Memory Content:")
    print(context_memory.get_formatted())
    
    # Create an adapter to convert the new memory to the old IntermediateStorage format
    adapted_storage = ContextMemoryAdapter.adapt_context_to_intermediate(context_memory)
    print("\nCreated adapter from new memory to IntermediateStorage")
    
    # Test intermediate storage operations
    adapted_storage.add_step("thought", "I should provide helpful information about Python")
    adapted_storage.add_tool_call(
        "search", 
        {"query": "Python for beginners"}, 
        {"result": "Python is a great language for beginners due to its readability."}
    )
    print("Added steps to the adapted storage")
    
    # Display memory through both interfaces
    print("\nAccessing through old interface:")
    print(adapted_storage.get_formatted_context())
    
    print("\nAccessing through new interface:")
    print(context_memory.get_formatted())
    
    # Test using the adapted storage with old memory manager
    memory_manager = CustomMemoryManager(adapted_storage)
    print("\nCreated custom memory manager with adapted storage")
    
    # Add a message through the old memory manager
    memory_manager.add({"role": "user", "content": "How do I learn Python?"})
    print("Added message through old memory manager")
    
    # Retrieve and display history
    history = memory_manager.get_history()
    print("\nRetrieved history through old memory manager:")
    for msg in history:
        print(f"- {msg['role']}: {msg['content']}")
    
    # Verify that changes appear in the new memory system
    print("\nVerifying new memory content contains all items:")
    print(context_memory.get_formatted())
    
    print("\nMemory compatibility test completed successfully.")


if __name__ == "__main__":
    main() 