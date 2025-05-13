"""
Test script for MCP client functionality.
"""
import asyncio
import logging
import os
import sys
from typing import Dict, Any, List

from agents.mcp_client.client import MCPServerConfig, MCPTransportType
from agents.mcp_client.mcp_sdk_client import MCPSDKClient
from agents.mcp_client.sequential_thinking_client import SequentialThinkingMCPClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_basic_client():
    """Test basic MCP client functionality."""
    logger.info("Testing basic MCP client functionality...")
    
    # Create a test server config
    server_config = MCPServerConfig(
        name="test_server",
        transport_type=MCPTransportType.STDIO,
        command="python",
        args=["-m", "mcp.tools.echo_server"]
    )
    
    try:
        # Create and connect to the client
        client = MCPSDKClient(server_config)
        connected = await client.connect()
        
        if connected:
            logger.info("Successfully connected to MCP server")
            
            # List tools
            tools = await client.list_tools()
            logger.info(f"Available tools: {tools}")
            
            # Call a tool (assuming the echo server has an 'echo' tool)
            result = await client.call_tool("echo", {"message": "Hello, MCP!"})
            logger.info(f"Tool call result: {result}")
            
            # Clean up
            await client.disconnect()
            logger.info("Successfully disconnected from MCP server")
        else:
            logger.error("Failed to connect to MCP server")
    except Exception as e:
        logger.error(f"Error during basic client test: {e}")


async def test_sequential_thinking():
    """Test sequential thinking MCP client."""
    logger.info("Testing sequential thinking MCP client...")
    
    # Create a test server config
    server_config = MCPServerConfig(
        name="test_server",
        transport_type=MCPTransportType.STDIO,
        command="python",
        args=["-m", "mcp.tools.echo_server"]
    )
    
    try:
        # Create and connect to the client
        client = SequentialThinkingMCPClient(server_config)
        connected = await client.connect()
        
        if connected:
            logger.info("Successfully connected to MCP server")
            
            # Solve a problem using sequential thinking
            problem = "What are the benefits and drawbacks of implementing a microservices architecture?"
            result = await client.solve_problem_sequentially(problem, max_thoughts=3)
            
            logger.info(f"Sequential thinking result: {result}")
            
            # Clean up
            await client.disconnect()
            logger.info("Successfully disconnected from MCP server")
        else:
            logger.error("Failed to connect to MCP server")
    except Exception as e:
        logger.error(f"Error during sequential thinking test: {e}")


async def main():
    """Run MCP client tests."""
    logger.info("Starting MCP client tests...")
    
    # Check if MCP SDK is installed
    try:
        import mcp
        logger.info("MCP SDK is installed")
    except ImportError:
        logger.warning("MCP SDK is not installed. Install with 'pip install mcp'")
        logger.warning("Tests will be skipped.")
        return
    
    # Run tests
    await test_basic_client()
    await test_sequential_thinking()
    
    logger.info("MCP client tests completed")


if __name__ == "__main__":
    asyncio.run(main()) 