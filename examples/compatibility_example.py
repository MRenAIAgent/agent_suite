#!/usr/bin/env python3
"""
Example demonstrating compatibility between existing IntermediateStorage and new memory system.

This example shows how to use the adapter pattern to allow existing agent code to work
with the new memory system and vice versa.
"""

from datetime import datetime

from agent_suite.agents.storage.intermediate_storage import InMemoryIntermediateStorage
from agent_suite.memory.context.in_memory import InMemoryContextMemory
from agent_suite.memory.adapter import ContextMemoryAdapter


def main():
    """Run the compatibility example."""
    print("Agent Suite Memory Compatibility Example")
    print("---------------------------------------\n")
    
    # Example 1: Using existing IntermediateStorage with new memory system
    print("\n=== Example 1: Using existing IntermediateStorage with new memory system ===\n")
    
    # Create an instance of the existing storage
    existing_storage = InMemoryIntermediateStorage()
    
    # Add some steps and tool calls to existing storage
    existing_storage.add_step("thought", "I should search for information about Python.")
    existing_storage.add_step("action", "search")
    existing_storage.add_step("action_input", "Python programming language")
    existing_storage.add_tool_call(
        "search", 
        {"query": "Python programming language"},
        {"result": "Python is a high-level programming language known for its readability."}
    )
    
    # Display original storage content
    print("Original IntermediateStorage content:")
    print(existing_storage.get_formatted_context())
    
    # Adapt the existing storage to the new ContextMemory interface
    context_adapter = ContextMemoryAdapter.adapt_intermediate_to_context(existing_storage)
    
    # Use the adapter with the new context memory interface
    print("\nUsing the same storage through ContextMemory interface:")
    print(context_adapter.get_formatted("default"))
    
    # Add more data through the adapter
    context_adapter.add("thought", "I should look up advanced Python concepts.", 
                       {"timestamp": datetime.now().isoformat()})
    
    # Show that both interfaces reflect the same data
    print("\nAfter adding data through the adapter:")
    print("Via ContextMemory interface:")
    print(context_adapter.get_formatted("default"))
    print("\nVia original IntermediateStorage interface:")
    print(existing_storage.get_formatted_context())
    
    # Example 2: Using new ContextMemory with existing agent code
    print("\n\n=== Example 2: Using new ContextMemory with existing agent code ===\n")
    
    # Create an instance of the new context memory
    new_memory = InMemoryContextMemory()
    
    # Add some items to the new memory
    new_memory.add("message", "Hello, I'm a helpful assistant.", 
                  {"role": "system", "timestamp": datetime.now().isoformat()})
    new_memory.add("query", "How can I learn Python?", 
                  {"role": "user", "timestamp": datetime.now().isoformat()})
    
    # Display original memory content
    print("Original ContextMemory content:")
    print(new_memory.get_formatted())
    
    # Adapt the new memory to the existing IntermediateStorage interface
    storage_adapter = ContextMemoryAdapter.adapt_context_to_intermediate(new_memory)
    
    # Use the adapter with the existing IntermediateStorage interface
    print("\nUsing the same memory through IntermediateStorage interface:")
    print(storage_adapter.get_formatted_context())
    
    # Add more data through the adapter
    storage_adapter.add_step("thought", "I should recommend Python resources.")
    storage_adapter.add_tool_call(
        "knowledge_base", 
        {"topic": "Python learning resources"},
        {"resources": ["docs.python.org", "Real Python", "Python Crash Course"]}
    )
    
    # Show that both interfaces reflect the same data
    print("\nAfter adding data through the adapter:")
    print("Via IntermediateStorage interface:")
    print(storage_adapter.get_formatted_context())
    print("\nVia original ContextMemory interface:")
    print(new_memory.get_formatted())
    
    print("\n\nCompatibility example completed successfully.")


if __name__ == "__main__":
    main() 