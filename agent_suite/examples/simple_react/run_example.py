#!/usr/bin/env python3
"""
Run the React Agent Example with Enhanced Tool System.

This script demonstrates running the React agent example with tools
registered in the enhanced tool registry.
"""
import os
import sys
import argparse

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from agent_suite.examples.simple_react.react_agent_example import ReactAgentExample


def main():
    """Run the React agent example with enhanced tool system."""
    print("=== React Agent with Enhanced Tool System ===")
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run the React agent example")
    parser.add_argument("--query", type=str, 
                        default="What is the current stock price of Tesla and what are analysts predicting for 2025?",
                        help="The query to run")
    args = parser.parse_args()
    
    # Create the React agent
    agent = ReactAgentExample(max_iterations=15)
    
    # Show registered tools
    tools = agent.get_registered_tools()
    print(f"\nRegistered {len(tools)} enhanced tools in the registry:")
    for tool in tools:
        print(f"- {tool.metadata.display_name}: {tool.metadata.description}")
    
    # Run the query
    print(f"\n\n{'='*50}")
    print(f"RUNNING QUERY: {args.query}")
    print(f"{'='*50}")
    
    response = agent.run(args.query)
    print(f"\nRESPONSE:\n{response}")


if __name__ == "__main__":
    main() 