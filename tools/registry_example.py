"""
Tool Registry Example with React Agent.

This script demonstrates how to:
1. Register existing tools with the enhanced tool registry
2. Retrieve tools from the registry
3. Use them with a React agent
"""
import asyncio
import os
import sys
from typing import List, Optional

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.react_agent import ReActAgent
from llm.litellm.litellm import LiteLLM
from log.logging import LogManager

# Import original tools
from tools.serpapi_tool import SerpApiSearchTool
from tools.animal_type import AnimalType

# Import enhanced tool system
from tools.registry import registry
from tools.base import EnhancedTool


def main():
    """Run the tool registry example."""
    print("Starting Tool Registry Example")
    
    # Step 1: Create instances of original tools
    search_tool = SerpApiSearchTool()
    animal_tool = AnimalType()
    
    print(f"Created original tools: {search_tool.__class__.__name__}, {animal_tool.__class__.__name__}")
    
    # Step 2: Convert original tools to enhanced tools and register them
    enhanced_search_tool = search_tool.to_enhanced_tool()
    enhanced_animal_tool = animal_tool.to_enhanced_tool()
    
    # Clear registry before registering our tools
    registry.clear()
    
    # Register the enhanced tools
    registry.register_tool(enhanced_search_tool)
    registry.register_tool(enhanced_animal_tool)
    
    print(f"Registered {len(registry.get_all_tools())} tools with the registry")
    
    # Step 3: Retrieve tools from the registry
    retrieved_tools = registry.get_all_tools()
    for tool in retrieved_tools:
        print(f"Retrieved tool: {tool.__class__.__name__}")
        # Print some metadata to show it's an enhanced tool
        print(f"  - Categories: {[str(c) for c in tool.metadata.categories]}")
        print(f"  - Capabilities: {[str(c) for c in tool.metadata.capabilities]}")
    
    # Step 4: For the React agent, we need to use the original tools
    # In a real-world scenario, you might maintain a mapping between enhanced
    # and original tools, or modify the React agent to work with enhanced tools
    
    # Initialize the ReAct agent with original tools
    llm = LiteLLM.create_llm()
    agent = ReActAgent(
        llm=llm,
        role="You are a helpful assistant with knowledge about animals and the ability to search the web.",
        task="Answer questions by using your tools when needed.",
        guide="Leverage your tools to find accurate information. For animal questions, use the AnimalType tool.",
        tools=[search_tool, animal_tool],  # Using the original tools
        log_manager=LogManager(),
        max_iterations=5
    )
    
    print("\nInitialized ReAct agent with original tools")
    
    # Step 5: Run a simple query
    query = "What type of animal is a dog?"
    print(f"\nRunning query: {query}")
    
    try:
        response = asyncio.run(agent.arun(query, "anthropic/claude-3-sonnet-20240229"))
        print(f"\nResponse:\n{response}")
    except Exception as e:
        print(f"Error running agent: {e}")
    
    # Step 6: Demonstrate adapter usage (using auto conversion)
    print("\n=== Demonstrating Adapter Usage ===")
    
    # Clear the registry
    registry.clear()
    
    # Import and use the auto_convert_legacy_tools function
    from tools.adapters.integration import auto_convert_legacy_tools
    
    # Auto-convert and register any existing tools
    converted_tools = auto_convert_legacy_tools()
    print(f"Auto-converted and registered {len(converted_tools)} tools")
    
    # List the converted tools
    for tool in converted_tools:
        print(f"Converted: {tool.__class__.__name__}")


if __name__ == "__main__":
    main() 