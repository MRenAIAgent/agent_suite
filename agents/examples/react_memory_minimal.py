#!/usr/bin/env python3
"""
Minimal example demonstrating compatibility between React agent and new memory system.

This example shows how to use the memory adapter to connect the existing React agent
with the new memory system, using only the in-memory components without external dependencies.
"""

import asyncio
import os
import sys
from datetime import datetime
from typing import List, Optional

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from agents.react_agent import ReActAgent
from llm.litellm.litellm import LiteLLM
from log.logging import LogManager
from tools.serper_api import SerperSearchTool
from agents.memory_manager import MemoryManager as OldMemoryManager

# Import minimal context memory - the only component we need for compatibility
from agent_suite.memory.context.in_memory import InMemoryContextMemory
from agent_suite.memory.adapter import ContextMemoryAdapter


class CustomMemoryManager(OldMemoryManager):
    """Custom memory manager that uses our provided storage."""
    
    def __init__(self, storage):
        """Initialize with the provided storage."""
        super().__init__()
        self.store = storage  # Override the default store with our adapted storage


def main():
    """Run the minimal memory compatibility example with React agent."""
    print("=== React Agent with Minimal Memory System Compatibility Example ===\n")
    
    # Create new memory system - just using the context memory without dependencies
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
    
    # Create a custom memory manager that uses our adapted storage
    memory_manager = CustomMemoryManager(adapted_storage)
    print("Created custom memory manager with adapted storage")
    
    # Display the memory through the old interface
    print("\nAccessing through old interface:")
    print(adapted_storage.get_formatted_context())
    
    # Initialize LLM
    llm = LiteLLM.create_llm()
    model = "anthropic/claude-3-7-sonnet-20250219"
    
    # Create a simple tool
    tools = [SerperSearchTool(query="")]
    
    # Initialize the React agent with the adapted memory
    agent = ReActAgent(
        llm=llm,
        role="You are a helpful assistant who provides concise answers.",
        task="Answer user questions directly and briefly.",
        guide="Keep answers short and to the point.",
        tools=tools,
        log_manager=LogManager(),
        memory_manager=memory_manager,  # Using the adapted memory
        max_iterations=2
    )
    print("\nCreated React agent with adapted memory")
    
    # Process a simple query
    query = "What's a good Python project for beginners?"
    print(f"\n=== Processing query: '{query}' ===")
    
    # Add the query to our new memory system
    context_memory.add("user_query", query, {"timestamp": datetime.now().isoformat()})
    print("Added user query to new memory")
    
    # Run the agent synchronously with a short timeout to avoid hanging
    try:
        response = asyncio.run(asyncio.wait_for(agent.arun(query, model), timeout=30))
        print("\nAgent response:")
        print(response)
    except asyncio.TimeoutError:
        print("\nAgent response timed out (this is expected in this minimal example without valid API keys)")
        response = "A good Python project for beginners is a simple to-do list application. It teaches basic concepts like lists, functions, and user input while creating something practical."
    
    # Manually add the response to our memory to simulate full execution
    context_memory.add("assistant_response", response, {"timestamp": datetime.now().isoformat()})
    
    # Display the updated memory content to verify everything worked
    print("\nNew Memory Content After Processing:")
    print(context_memory.get_formatted())
    
    # Verify memory content through the old interface
    print("\nAccessing through old interface after processing:")
    print(adapted_storage.get_formatted_context())
    
    print("\nMinimal memory compatibility example completed successfully.")


if __name__ == "__main__":
    main() 