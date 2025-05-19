#!/usr/bin/env python3
"""
Run MCP React Agent Example

This script demonstrates how to run the MCP React Agent
with a mock MCP server for testing purposes.
"""
import asyncio
import os
import sys
import logging
import argparse

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from agent_suite.agent_suite.integrations.mcp.mock_server import MockMCPServer
from agent_suite.examples.mcp_integration.mcp_react_agent import MCPEnhancedReactAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def run_agent(query: str, model: str = "anthropic/claude-3-7-sonnet-20250219"):
    """Run the agent with a query.
    
    Args:
        query: The user query to process
        model: The LLM model to use
    """
    # Start the mock MCP server
    server = MockMCPServer(port=8000)
    server.start()
    
    try:
        # Create the agent
        agent = MCPEnhancedReactAgent(
            mcp_server_url=server.base_url,
            mcp_namespace="mock_mcp",
            model=model
        )
        
        # Initialize the agent
        await agent.initialize()
        
        # Process the query
        print(f"\n\n{'='*50}")
        print(f"QUERY: {query}")
        print(f"{'='*50}")
        
        # Get the response
        response = await agent.arun(query)
        
        print(f"\nRESPONSE: {response}")
        
        # Close the agent
        await agent.close()
    
    finally:
        # Stop the server
        server.stop()


def main():
    """Run the MCP React Agent example."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run MCP React Agent")
    parser.add_argument(
        "--query", 
        type=str, 
        default="What's the weather like in New York?",
        help="Query to process"
    )
    parser.add_argument(
        "--model", 
        type=str, 
        default="anthropic/claude-3-7-sonnet-20250219",
        help="LLM model to use"
    )
    args = parser.parse_args()
    
    # Run the agent
    asyncio.run(run_agent(args.query, args.model))


if __name__ == "__main__":
    main() 