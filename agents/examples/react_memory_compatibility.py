#!/usr/bin/env python3
"""
Example demonstrating compatibility between React agent and new memory system.

This example shows how to use the memory adapter to connect the existing React agent
with the new memory system.
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

# Import new memory system components
from agent_suite.memory.context.in_memory import InMemoryContextMemory
from agent_suite.memory.adapter import ContextMemoryAdapter


class CustomMemoryManager(OldMemoryManager):
    """Custom memory manager that uses our provided storage."""
    
    def __init__(self, storage):
        """Initialize with the provided storage."""
        super().__init__()
        self.store = storage  # Override the default store with our adapted storage


def main():
    """Run the memory compatibility example with React agent."""
    print("=== React Agent with New Memory System Compatibility Example ===\n")
    
    # Create new memory system
    context_memory = InMemoryContextMemory()
    print("Created new InMemoryContextMemory instance")
    
    # Add some test data to the memory
    context_memory.add("system_message", 
                    "You are a helpful assistant using the new memory system.", 
                    {"timestamp": datetime.now().isoformat()})
    print("Added system message to new memory")
    
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
        role="You are a helpful assistant.",
        task="Answer user questions concisely.",
        guide="Provide brief, helpful responses to user queries.",
        tools=tools,
        log_manager=LogManager(),
        memory_manager=memory_manager,  # Using the adapted memory
        max_iterations=3
    )
    print("\nCreated React agent with adapted memory")
    
    # Process a simple query
    query = "What is the capital of France?"
    print(f"\n=== Processing query: '{query}' ===")
    
    # Add the query to our new memory system
    context_memory.add("user_query", query, {"timestamp": datetime.now().isoformat()})
    print("Added user query to new memory")
    
    # Run the agent synchronously
    response = asyncio.run(agent.arun(query, model))
    print("\nAgent response:")
    print(response)
    
    # Display the updated memory content to verify everything worked
    print("\nNew Memory Content After Processing:")
    print(context_memory.get_formatted())
    
    # Verify memory content through the old interface
    print("\nAccessing through old interface after processing:")
    print(adapted_storage.get_formatted_context())
    
    print("\nMemory compatibility example completed successfully.")


if __name__ == "__main__":
    main() 