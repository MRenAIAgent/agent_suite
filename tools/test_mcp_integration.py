#!/usr/bin/env python3
"""
MCP Integration Tests

This script tests the various components of the MCP integration:
1. Mock MCP Server
2. MCP SDK Client
3. MCP SDK Wrapper
4. Tool registration with the enhanced tool system
5. Tool selection based on query
6. Tool execution through the MCP client
"""
import asyncio
import logging
import json
import os
import sys
from typing import List, Dict, Any

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tools.mock_mcp_server import MockMCPServer
from tools.mcp_sdk_client import MCPSDKClient
from tools.mcp_sdk_wrapper import MCPSDKToolProvider, MCPSDKToolInstance
from tools.tool_selector import ToolSelector, SelectionMethod
from tools.registry import registry

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def test_mock_server():
    """Test the mock MCP server."""
    print("\n=== Testing Mock MCP Server ===")
    
    # Start the server
    server = MockMCPServer(port=8000)
    server.start()
    print(f"Mock server started at {server.base_url}")
    
    try:
        # Make HTTP requests to test the server
        import aiohttp
        
        async with aiohttp.ClientSession() as session:
            # Test tool discovery
            async with session.get(f"{server.base_url}/tools") as response:
                assert response.status == 200, "Failed to get tools"
                data = await response.json()
                tools = data.get("tools", [])
                print(f"Discovered {len(tools)} tools")
                assert len(tools) > 0, "No tools returned"
            
            # Test tool execution
            calculator_params = {"expression": "10 * 5"}
            async with session.post(
                f"{server.base_url}/execute",
                json={"name": "calculator_compute", "parameters": calculator_params}
            ) as response:
                assert response.status == 200, "Failed to execute calculator tool"
                result = await response.json()
                print(f"Calculator result: {json.dumps(result, indent=2)}")
                assert "result" in result, "Result not in response"
                assert result["result"]["result"] == 50, "Wrong calculation result"
    
    finally:
        server.stop()
        print("Mock server stopped")


async def test_mcp_sdk_client():
    """Test the MCP SDK client."""
    print("\n=== Testing MCP SDK Client ===")
    
    # Start the server
    server = MockMCPServer(port=8000)
    server.start()
    
    try:
        # Create and initialize the client
        client = MCPSDKClient(base_url=server.base_url)
        await client.initialize()
        
        # Test tool discovery
        tools = client.get_available_tools()
        print(f"Discovered {len(tools)} tools via client")
        assert len(tools) > 0, "No tools discovered"
        
        for i, tool in enumerate(tools):
            print(f"  {i+1}. {tool['name']}: {tool['description']}")
        
        # Test tool execution
        weather_result = await client.execute_tool(
            "weather_get_forecast",
            {"location": "New York"}
        )
        print(f"Weather result: {json.dumps(weather_result, indent=2)}")
        assert "forecast" in weather_result, "Forecast not in result"
        
        # Test error handling
        try:
            await client.execute_tool("nonexistent_tool", {})
            assert False, "Should have raised an error for nonexistent tool"
        except ValueError:
            print("Correctly raised error for nonexistent tool")
        
        # Close the client
        await client.close()
    
    finally:
        server.stop()


async def test_mcp_sdk_wrapper():
    """Test the MCP SDK wrapper."""
    print("\n=== Testing MCP SDK Wrapper ===")
    
    # Start the server
    server = MockMCPServer(port=8000)
    server.start()
    
    try:
        # Clear the registry
        registry.clear()
        
        # Create the tool provider
        provider = MCPSDKToolProvider(
            server_url=server.base_url,
            namespace="test_mcp"
        )
        
        # Discover and register tools
        tools = await provider.discover_and_register_tools()
        print(f"Registered {len(tools)} tools with the enhanced tool system")
        assert len(tools) > 0, "No tools registered"
        
        for i, tool in enumerate(tools):
            print(f"  {i+1}. {tool.metadata.name}")
            print(f"     Categories: {[cat.value for cat in tool.metadata.categories]}")
            print(f"     Capabilities: {[cap.value for cap in tool.metadata.capabilities]}")
        
        # Test tool execution via wrapper
        calculator_tool = next(
            (tool for tool in tools if "calculator" in tool.metadata.name.lower()),
            None
        )
        
        assert calculator_tool, "Calculator tool not found"
        
        result = await calculator_tool._mcp_tool_instance.arun(expression="5 * 10")
        print(f"Calculator result: {json.dumps(result, indent=2)}")
        assert "result" in result, "Result not in response"
        assert result["result"] == 50, "Wrong calculation result"
        
        # Close the provider
        await provider.close()
    
    finally:
        server.stop()


async def test_tool_selection():
    """Test tool selection with MCP tools."""
    print("\n=== Testing Tool Selection with MCP Tools ===")
    
    # Start the server
    server = MockMCPServer(port=8000)
    server.start()
    
    try:
        # Clear the registry
        registry.clear()
        
        # Register tools
        provider = MCPSDKToolProvider(
            server_url=server.base_url,
            namespace="test_mcp"
        )
        await provider.discover_and_register_tools()
        
        # Create selector
        selector = ToolSelector(registry)
        
        # Test queries
        test_queries = [
            "What's the weather like in New York?",
            "Calculate 125 * 8 - 32",
            "What's the current stock price of AAPL?",
            "Search the web for information about climate change"
        ]
        
        for query in test_queries:
            print(f"\nQuery: '{query}'")
            
            # Select tools using hybrid method
            tools = selector.select_tools(
                query=query,
                method=SelectionMethod.HYBRID,
                max_tools=2
            )
            
            print(f"Selected {len(tools)} tools:")
            for i, tool in enumerate(tools):
                print(f"  {i+1}. {tool.metadata.name}")
                print(f"     Description: {tool.metadata.description}")
                print(f"     Score: {tool.metadata.relevance_score:.2f}")
        
        # Close the provider
        await provider.close()
    
    finally:
        server.stop()


async def test_end_to_end():
    """Test end-to-end integration."""
    print("\n=== Testing End-to-End Integration ===")
    
    # Start the server
    server = MockMCPServer(port=8000)
    server.start()
    
    try:
        # Import the agent
        from tools.mcp_agent_example import MCPEnhancedReactAgent
        
        # Create and initialize the agent
        agent = MCPEnhancedReactAgent(
            mcp_server_url=server.base_url,
            mcp_namespace="test_mcp",
            max_iterations=3  # Limit iterations for testing
        )
        
        # Initialize
        await agent.initialize()
        
        # Test a simple query to avoid long execution
        query = "Calculate 125 * 8 - 32"
        print(f"Testing query: '{query}'")
        
        # Select tools
        tools = agent._select_tools_for_query(query)
        print(f"Selected {len(tools)} tools:")
        for i, tool in enumerate(tools):
            print(f"  {i+1}. {tool.metadata.name}")
        
        # Test tool execution
        calculator_tool = next(
            (tool for tool in tools if "calculator" in tool.metadata.name.lower()),
            None
        )
        
        if calculator_tool:
            print("\nExecuting calculator tool...")
            result = await calculator_tool._mcp_tool_instance.arun(expression="125 * 8 - 32")
            print(f"Result: {result}")
            assert "result" in result, "Result not in response"
            assert result["result"] == 968, "Wrong calculation result"
        
        # Close the agent
        await agent.close()
    
    finally:
        server.stop()


async def run_tests():
    """Run all tests."""
    tests = [
        test_mock_server,
        test_mcp_sdk_client,
        test_mcp_sdk_wrapper,
        test_tool_selection,
        test_end_to_end
    ]
    
    for test in tests:
        try:
            await test()
            print(f"✅ {test.__name__} passed")
        except Exception as e:
            print(f"❌ {test.__name__} failed: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    print("Running MCP integration tests...")
    asyncio.run(run_tests())
    print("\nAll tests complete!") 