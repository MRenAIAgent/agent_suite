#!/usr/bin/env python3
"""
Simple example to test a basic agent.

This script creates a minimal agent to test the structure works correctly.
"""

import os
import sys

# Add the project root to the path to ensure imports work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from agents.memory.memory_manager import MemoryManager
    from agents.memory.context_memory import ContextMemory 
    from agents.memory.stores.in_memory_store import InMemoryStore
except ImportError as e:
    print(f"Error importing memory components: {e}")
    sys.exit(1)


class SimpleAgent:
    """A very simple agent with memory for testing."""
    
    def __init__(self, name="SimpleAgent"):
        """Initialize the agent with memory."""
        self.name = name
        
        # Create memory components
        context_store = InMemoryStore()
        
        # Create memory manager with in-memory store for context
        self.memory = MemoryManager(
            max_history=20,
            use_longterm=False,
            context_store=context_store
        )
        
        print(f"Initialized {self.name} with memory")
    
    def add_message(self, role, content):
        """Add a message to the agent's memory."""
        message = {
            "role": role,
            "content": content
        }
        self.memory.add(message)
        print(f"Added {role} message: {content[:50]}..." if len(content) > 50 else f"Added {role} message: {content}")
    
    def get_history(self):
        """Get the agent's conversation history."""
        return self.memory.get_formatted_history()
    
    def set_context(self, key, value):
        """Set a value in the agent's context."""
        self.memory.set_context(key, value)
        print(f"Set context {key}: {str(value)[:50]}..." if len(str(value)) > 50 else f"Set context {key}: {value}")
    
    def get_context(self):
        """Get the agent's context."""
        return self.memory.get_context()


def main():
    """Run a simple agent test."""
    print("Simple Agent Test")
    print("----------------\n")
    
    # Create a simple agent
    agent = SimpleAgent()
    
    # Set some context values
    agent.set_context("system", "You are a helpful assistant.")
    agent.set_context("preferences", {
        "verbose": True,
        "creativity": "high",
        "format": "markdown"
    })
    
    # Add some conversation history
    agent.add_message("user", "Hello, can you help me with a Python question?")
    agent.add_message("assistant", "Of course! I'd be happy to help with your Python question. What would you like to know?")
    agent.add_message("user", "How do I create a simple class in Python?")
    agent.add_message("assistant", """
    To create a simple class in Python, you can use the following syntax:

    ```python
    class MyClass:
        def __init__(self, param1, param2):
            self.param1 = param1
            self.param2 = param2
            
        def my_method(self):
            return f"Values: {self.param1}, {self.param2}"
    
    # Create an instance
    my_instance = MyClass("hello", 42)
    print(my_instance.my_method())
    ```
    
    This creates a class with:
    1. An initializer (`__init__`) that takes parameters
    2. Instance variables (self.param1, self.param2)
    3. A method that can access those variables
    
    Would you like me to explain any part of this in more detail?
    """)
    
    # Print the agent's state
    print("\nAgent Context:")
    print(agent.get_context())
    
    print("\nConversation History:")
    print(agent.get_history())
    
    print("\nSimple agent test completed successfully!")


if __name__ == "__main__":
    main() 