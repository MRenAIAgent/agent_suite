#!/usr/bin/env python3
"""
Run the MCP React Agent Example.

This script demonstrates running the React agent with MCP tools
registered in the enhanced tool registry.
"""
import os
import sys

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from agents.examples.mcp_react_agent_example import MCPReactAgentExample


def main():
    """Run the MCP React agent example."""
    print("=== React Agent with MCP Tool System ===")
    
    # Create the MCP React agent
    agent = MCPReactAgentExample(
        model="openai/gpt-3.5-turbo",  # Use OpenAI model for better compatibility
        max_iterations=20
    )
    
    # Show registered tools
    tools = agent.get_registered_tools()
    print(f"\nRegistered {len(tools)} MCP tools in the registry:")
    for tool in tools:
        print(f"- {tool.metadata.display_name}: {tool.metadata.description}")
    
    # Run a test query
    test_query = "Find the most popular Python web frameworks on GitHub and compare their features."
    print(f"\n\n{'='*50}")
    print(f"RUNNING QUERY: {test_query}")
    print(f"{'='*50}")
    
    response = agent.run(test_query)
    print(f"\nRESPONSE:\n{response}")


if __name__ == "__main__":
    main() 