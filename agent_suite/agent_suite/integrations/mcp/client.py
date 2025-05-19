#!/usr/bin/env python3
"""
MCP SDK Client

A client for interacting with the MCP (Model Composition Platform) API.
"""
import aiohttp
import json
import logging
import asyncio
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


class MCPSDKClient:
    """Client for MCP SDK API interactions."""
    
    def __init__(self, base_url: str, api_key: Optional[str] = None):
        """Initialize the MCP SDK client.
        
        Args:
            base_url: Base URL of the MCP server
            api_key: Optional API key for authentication
        """
        self.base_url = base_url
        self.api_key = api_key
        self.session = None
        self._available_tools = []
        self._initialized = False
    
    async def initialize(self):
        """Initialize the client by fetching available tools."""
        if self._initialized:
            return
        
        self.session = aiohttp.ClientSession()
        await self._fetch_tools()
        self._initialized = True
    
    async def _fetch_tools(self):
        """Fetch available tools from the MCP server."""
        if not self.session:
            self.session = aiohttp.ClientSession()
            
        async with self.session.get(f"{self.base_url}/tools") as response:
            if response.status == 200:
                data = await response.json()
                self._available_tools = data.get("tools", [])
                logger.info(f"Fetched {len(self._available_tools)} tools from MCP server")
            else:
                error_text = await response.text()
                logger.error(f"Failed to fetch tools from MCP server: {error_text}")
                raise ValueError(f"Failed to fetch tools: {response.status} - {error_text}")
    
    def get_available_tools(self) -> List[Dict[str, Any]]:
        """Get the list of available tools.
        
        Returns:
            List of available tools with their metadata
        """
        return self._available_tools
    
    async def execute_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Any:
        """Execute a tool with the given parameters.
        
        Args:
            tool_name: Name of the tool to execute
            parameters: Parameters to pass to the tool
            
        Returns:
            Result of the tool execution
            
        Raises:
            ValueError: If the tool is not found or execution fails
        """
        if not self._initialized:
            await self.initialize()
            
        # Check if the tool exists
        tool_exists = any(tool["name"] == tool_name for tool in self._available_tools)
        if not tool_exists:
            available_tools = ", ".join(tool["name"] for tool in self._available_tools)
            raise ValueError(f"Tool '{tool_name}' not found. Available tools: {available_tools}")
        
        # Execute the tool
        request_data = {
            "name": tool_name,
            "parameters": parameters
        }
        
        async with self.session.post(
            f"{self.base_url}/execute", 
            json=request_data
        ) as response:
            if response.status == 200:
                result = await response.json()
                return result
            else:
                error_text = await response.text()
                logger.error(f"Failed to execute tool {tool_name}: {error_text}")
                raise ValueError(f"Tool execution failed: {response.status} - {error_text}")
    
    async def close(self):
        """Close the client session."""
        if self.session:
            await self.session.close()
            self.session = None
            self._initialized = False


async def test_mcp_sdk_client():
    """Test the MCP SDK client with a mock server."""
    # Import the mock server
    from .mock_server import MockMCPServer
    
    # Start a mock server
    server = MockMCPServer(port=8000)
    server.start()
    
    try:
        # Create a client
        client = MCPSDKClient(base_url=server.base_url)
        
        # Initialize the client
        await client.initialize()
        
        # Get available tools
        tools = client.get_available_tools()
        print(f"Available tools ({len(tools)}):")
        for tool in tools:
            print(f"  - {tool['name']}: {tool['description']}")
        
        # Test executing the weather tool
        print("\nExecuting weather_get_forecast tool...")
        weather_result = await client.execute_tool(
            "weather_get_forecast",
            {"location": "New York"}
        )
        print(f"Weather result: {json.dumps(weather_result, indent=2)}")
        
        # Test executing the calculator tool
        print("\nExecuting calculator_compute tool...")
        calc_result = await client.execute_tool(
            "calculator_compute",
            {"expression": "10 * 5 + 2"}
        )
        print(f"Calculator result: {json.dumps(calc_result, indent=2)}")
        
        # Close the client
        await client.close()
    
    finally:
        # Stop the server
        server.stop()


if __name__ == "__main__":
    asyncio.run(test_mcp_sdk_client()) 