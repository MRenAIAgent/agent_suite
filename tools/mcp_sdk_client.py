"""
MCP SDK Client

This module provides a client for connecting to MCP (Model Completion Protocol) servers,
discovering available tools, and executing them.
"""
import asyncio
import json
import logging
import aiohttp
from typing import Dict, Any, List, Optional, Union

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MCPSDKClient:
    """Client for interacting with MCP servers."""
    
    def __init__(self, base_url: str, api_key: Optional[str] = None):
        """Initialize the MCP SDK client.
        
        Args:
            base_url: The base URL of the MCP server
            api_key: Optional API key for authentication
        """
        self.base_url = base_url
        self.api_key = api_key
        self.session = None
        self.tools = {}
        self._initialized = False
    
    async def initialize(self):
        """Initialize the client by discovering available tools."""
        if self._initialized:
            return
        
        # Create a new session if needed
        if self.session is None:
            self.session = aiohttp.ClientSession()
        
        # Discover available tools
        await self.discover_tools()
        self._initialized = True
    
    async def discover_tools(self) -> List[Dict[str, Any]]:
        """Discover available tools from the MCP server.
        
        Returns:
            List of available tools
        """
        # Make sure we have a session
        if self.session is None:
            self.session = aiohttp.ClientSession()
        
        tools_url = f"{self.base_url}/tools"
        headers = {}
        
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        try:
            async with self.session.get(tools_url, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    tools = data.get("tools", [])
                    
                    # Cache the tools by name
                    for tool in tools:
                        self.tools[tool.get("name")] = tool
                    
                    logger.info(f"Discovered {len(tools)} tools from MCP server")
                    return tools
                else:
                    error_text = await response.text()
                    logger.error(f"Failed to discover tools: HTTP {response.status} - {error_text}")
                    return []
        
        except Exception as e:
            logger.error(f"Error discovering tools: {e}")
            return []
    
    async def execute_tool(
        self, 
        tool_name: str, 
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a tool on the MCP server.
        
        Args:
            tool_name: Name of the tool to execute
            parameters: Parameters for the tool
        
        Returns:
            Results from the tool execution
        
        Raises:
            ValueError: If the tool is not found
            RuntimeError: If the tool execution fails
        """
        # Make sure we're initialized
        if not self._initialized:
            await self.initialize()
        
        # Check if the tool exists
        if tool_name not in self.tools:
            raise ValueError(f"Tool not found: {tool_name}")
        
        # Create the request payload
        payload = {
            "name": tool_name,
            "parameters": parameters
        }
        
        # Set up headers
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        # Execute the tool
        try:
            execute_url = f"{self.base_url}/execute"
            
            async with self.session.post(
                execute_url,
                headers=headers,
                json=payload
            ) as response:
                response_data = await response.json()
                
                if response.status == 200:
                    result = response_data.get("result", {})
                    logger.info(f"Successfully executed tool {tool_name}")
                    return result
                else:
                    error = response_data.get("error", f"HTTP Error: {response.status}")
                    logger.error(f"Error executing tool {tool_name}: {error}")
                    raise RuntimeError(f"Error executing tool: {error}")
        
        except Exception as e:
            logger.error(f"Error executing tool {tool_name}: {e}")
            raise RuntimeError(f"Error executing tool {tool_name}: {str(e)}")
    
    def get_tool_metadata(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a specific tool.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            Tool metadata or None if not found
        """
        return self.tools.get(tool_name)
    
    def get_available_tools(self) -> List[Dict[str, Any]]:
        """Get list of available tools.
        
        Returns:
            List of available tools
        """
        return list(self.tools.values())
    
    async def close(self):
        """Close the client session."""
        if self.session:
            await self.session.close()
            self.session = None


async def test_mcp_sdk_client():
    """Test the MCP SDK client with a mock server."""
    # Import the mock server
    from tools.mock_mcp_server import MockMCPServer
    
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