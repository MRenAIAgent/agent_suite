"""
MCP SDK Wrapper

This module connects the MCP SDK client with the enhanced tool system,
allowing tools from an MCP server to be registered and used in agents.
"""
import asyncio
import logging
from typing import Dict, Any, List, Optional, Set

from tools.mcp_sdk_client import MCPSDKClient
from tools.mcp_wrapper import register_mcp_tool, register_mcp_tools, MCPToolWrapper
from tools.registry import registry
from tools.tool_types import (
    ToolCategory, 
    ToolCapability, 
    ToolDomain,
    ToolSource,
    ToolMetadata
)

# Setup logging
logger = logging.getLogger(__name__)


class MCPSDKToolInstance:
    """Wrapper around an MCP SDK tool for execution."""
    
    def __init__(self, client: MCPSDKClient, tool_name: str):
        """Initialize the MCP SDK tool instance.
        
        Args:
            client: The MCP SDK client
            tool_name: The name of the tool
        """
        self.client = client
        self.tool_name = tool_name
    
    async def arun(self, **kwargs) -> Any:
        """Execute the tool asynchronously.
        
        Args:
            **kwargs: Parameters for the tool
            
        Returns:
            Results from the tool execution
        """
        try:
            result = await self.client.execute_tool(self.tool_name, kwargs)
            return result
        except Exception as e:
            logger.error(f"Error executing tool {self.tool_name}: {e}")
            return {"error": str(e)}
    
    def run(self, **kwargs) -> Any:
        """Execute the tool synchronously.
        
        Args:
            **kwargs: Parameters for the tool
            
        Returns:
            Results from the tool execution
        """
        return asyncio.run(self.arun(**kwargs))


class MCPSDKToolProvider:
    """Provider for MCP SDK tools."""
    
    def __init__(self, server_url: str, api_key: Optional[str] = None, namespace: str = "mcp_sdk"):
        """Initialize the MCP SDK tool provider.
        
        Args:
            server_url: URL of the MCP server
            api_key: Optional API key for authentication
            namespace: Namespace for tool registration
        """
        self.server_url = server_url
        self.api_key = api_key
        self.namespace = namespace
        self.client = MCPSDKClient(base_url=server_url, api_key=api_key)
        self.registered_tools: Set[str] = set()
    
    async def discover_and_register_tools(self) -> List[MCPToolWrapper]:
        """Discover tools from the MCP server and register them.
        
        Returns:
            List of registered MCP tool wrappers
        """
        # Initialize the client
        await self.client.initialize()
        
        # Get available tools
        tools = self.client.get_available_tools()
        logger.info(f"Discovered {len(tools)} tools from MCP server at {self.server_url}")
        
        # Prepare tools for registration
        mcp_tools = {}
        for tool in tools:
            tool_name = tool.get("name", "")
            
            # Skip already registered tools
            if tool_name in self.registered_tools:
                continue
            
            # Create the tool data structure
            tool_data = {
                "description": tool.get("description", ""),
                "parameters": tool.get("parameters", {}),
                "categories": self._extract_categories(tool),
                "capabilities": self._extract_capabilities(tool),
                "domains": self._extract_domains(tool),
                "keywords": self._extract_keywords(tool)
            }
            
            # Create the tool instance
            tool_instance = MCPSDKToolInstance(self.client, tool_name)
            
            # Add to the registration dictionary
            mcp_tools[f"{self.namespace}.{tool_name}"] = {
                "instance": tool_instance,
                "data": tool_data,
                "namespace": self.namespace
            }
            
            # Track registered tools
            self.registered_tools.add(tool_name)
        
        # Register the tools
        registered_wrappers = register_mcp_tools(mcp_tools)
        
        logger.info(f"Registered {len(registered_wrappers)} tools with namespace '{self.namespace}'")
        
        return list(registered_wrappers.values())
    
    def _extract_categories(self, tool: Dict[str, Any]) -> List[str]:
        """Extract categories from tool data.
        
        Args:
            tool: Tool data from the MCP server
            
        Returns:
            List of category strings
        """
        name = tool.get("name", "").lower()
        desc = tool.get("description", "").lower()
        
        categories = []
        
        if any(x in name or x in desc for x in ["search", "find", "lookup"]):
            categories.append("search")
        
        if any(x in name or x in desc for x in ["calculate", "compute", "math"]):
            categories.append("utility")
        
        if any(x in name or x in desc for x in ["weather", "forecast", "temperature"]):
            categories.append("utility")
        
        if any(x in name or x in desc for x in ["stock", "price", "market", "financial"]):
            categories.append("finance")
        
        # Default to utility if nothing else matches
        if not categories:
            categories.append("utility")
            
        return categories
    
    def _extract_capabilities(self, tool: Dict[str, Any]) -> List[str]:
        """Extract capabilities from tool data.
        
        Args:
            tool: Tool data from the MCP server
            
        Returns:
            List of capability strings
        """
        name = tool.get("name", "").lower()
        desc = tool.get("description", "").lower()
        
        capabilities = []
        
        if any(x in name or x in desc for x in ["search", "find", "lookup"]):
            capabilities.append("search")
        
        if any(x in name or x in desc for x in ["calculate", "compute", "math"]):
            capabilities.append("compute")
        
        if any(x in name or x in desc for x in ["get", "retrieve", "fetch"]):
            capabilities.append("fetch")
        
        # Default to read if nothing else matches
        if not capabilities:
            capabilities.append("read")
            
        return capabilities
    
    def _extract_domains(self, tool: Dict[str, Any]) -> List[str]:
        """Extract domains from tool data.
        
        Args:
            tool: Tool data from the MCP server
            
        Returns:
            List of domain strings
        """
        name = tool.get("name", "").lower()
        desc = tool.get("description", "").lower()
        
        domains = []
        
        if any(x in name or x in desc for x in ["weather", "forecast", "temperature"]):
            domains.append("general")
        
        if any(x in name or x in desc for x in ["stock", "price", "market", "financial"]):
            domains.append("finance")
        
        if any(x in name or x in desc for x in ["code", "programming", "developer"]):
            domains.append("programming")
        
        # Default to general if nothing else matches
        if not domains:
            domains.append("general")
            
        return domains
    
    def _extract_keywords(self, tool: Dict[str, Any]) -> List[str]:
        """Extract keywords from tool data.
        
        Args:
            tool: Tool data from the MCP server
            
        Returns:
            List of keyword strings
        """
        name = tool.get("name", "").lower()
        desc = tool.get("description", "").lower()
        
        # Split name and description into words
        name_parts = name.replace("_", " ").split()
        desc_words = [word for word in desc.split() if len(word) > 3]
        
        # Combine all potential keywords
        all_keywords = name_parts + desc_words
        
        # Remove duplicates
        keywords = list(set(all_keywords))
        
        return keywords
    
    async def close(self):
        """Close the MCP SDK client."""
        await self.client.close()


async def test_mcp_sdk_wrapper():
    """Test the MCP SDK wrapper with a mock server."""
    # Import the mock server
    from tools.mock_mcp_server import MockMCPServer
    
    # Start a mock server
    server = MockMCPServer(port=8000)
    server.start()
    
    try:
        # Clear the registry
        registry.clear()
        
        # Create the MCP SDK tool provider
        provider = MCPSDKToolProvider(
            server_url=server.base_url,
            namespace="mock_mcp"
        )
        
        # Discover and register tools
        registered_tools = await provider.discover_and_register_tools()
        
        print(f"Registered {len(registered_tools)} tools:")
        for i, tool in enumerate(registered_tools):
            print(f"  {i+1}. {tool.metadata.name}")
            print(f"     Description: {tool.metadata.description}")
            print(f"     Categories: {[cat.value for cat in tool.metadata.categories]}")
            print(f"     Capabilities: {[cap.value for cap in tool.metadata.capabilities]}")
            print(f"     Domains: {[dom.value for dom in tool.metadata.domains]}")
        
        # Test executing a registered tool
        calculator_tool = next(
            (tool for tool in registered_tools if "calculator" in tool.metadata.name.lower()),
            None
        )
        
        if calculator_tool:
            print("\nExecuting calculator tool...")
            result = await calculator_tool._mcp_tool_instance.arun(expression="5 * 10")
            print(f"Result: {result}")
        
        # Close the provider
        await provider.close()
    
    finally:
        # Stop the server
        server.stop()


if __name__ == "__main__":
    asyncio.run(test_mcp_sdk_wrapper()) 