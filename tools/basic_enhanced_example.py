"""
Basic Enhanced Tool Example with React Agent.

This script demonstrates how to use existing tools with the enhanced tool system:
1. Converting existing tools to enhanced tools
2. Registering them with the tool registry 
3. Using them with a React agent
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

# Import tool converters
from tools.base import Tool
from tools.tool import Tool as OriginalTool


def main():
    """Run the basic enhanced tool example."""
    print("Starting Basic Enhanced Tool Example")
    
    # Step 1: Create instances of original tools
    search_tool = SerpApiSearchTool()
    animal_tool = AnimalType()
    
    print(f"Created original tools: {search_tool.__class__.__name__}, {animal_tool.__class__.__name__}")
    
    # Step 2: Convert original tools to enhanced tools
    # This is just for demonstration - we don't actually need to use the enhanced tools
    # for this example, but this shows how the conversion works
    enhanced_search_tool = search_tool.to_enhanced_tool()
    enhanced_animal_tool = animal_tool.to_enhanced_tool()
    
    print(f"Converted to enhanced tools: {enhanced_search_tool.__class__.__name__}, {enhanced_animal_tool.__class__.__name__}")
    
    # Step 3: Initialize the ReAct agent with original tools
    # The React agent still expects original Tool instances, not EnhancedTool
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
    
    print("Initialized ReAct agent with original tools")
    
    # Step 4: Run a simple query
    query = "What type of animal is a dog?"
    print(f"\nRunning query: {query}")
    
    try:
        response = asyncio.run(agent.arun(query, "anthropic/claude-3-sonnet-20240229"))
        print(f"\nResponse:\n{response}")
    except Exception as e:
        print(f"Error running agent: {e}")


if __name__ == "__main__":
    main() 