#!/usr/bin/env python3
"""
MCP SDK Wrapper

This module provides a wrapper for the MCP SDK client,
allowing tools from MCP to be used with the enhanced tool system.
"""
import asyncio
import logging
from typing import List, Dict, Any, Optional, Callable

from pydantic import Field

from .client import MCPSDKClient
from ...tools.base import EnhancedTool
from ...tools.registry import registry
from ...tools.types import (
    ToolCategory,
    ToolCapability,
    ToolDomain,
    ToolMetadata,
    ToolSource
)

logger = logging.getLogger(__name__)


class MCPSDKToolInstance:
    """A wrapper around an MCP tool instance that implements the callable interface."""
    
    def __init__(
        self,
        tool_name: str,
        client: MCPSDKClient,
        description: str,
        parameters: Dict[str, Any]
    ):
        """Initialize an MCP SDK tool instance.
        
        Args:
            tool_name: Name of the MCP tool
            client: MCP SDK client instance
            description: Description of the tool
            parameters: Parameter schema for the tool
        """
        self.tool_name = tool_name
        self.name = tool_name  # Add name attribute for ReActAgent compatibility
        self.client = client
        self.description = description
        self.parameters = parameters
        
    async def arun(self, **kwargs) -> Any:
        """Run the tool asynchronously.
        
        Args:
            **kwargs: Tool parameters
            
        Returns:
            Result of the tool execution
        """
        try:
            return await self.client.execute_tool(self.tool_name, kwargs)
        except Exception as e:
            logger.error(f"Error executing MCP tool {self.tool_name}: {e}")
            return {"error": f"Error executing MCP tool {self.tool_name}: {str(e)}"}
            
    def run(self, **kwargs) -> Any:
        """Run the tool synchronously.
        
        Args:
            **kwargs: Tool parameters
            
        Returns:
            Result of the tool execution
        """
        return asyncio.run(self.arun(**kwargs))


class MCPEnhancedTool(EnhancedTool):
    """Enhanced tool wrapper for MCP tools."""
    
    def __init__(
        self,
        metadata: ToolMetadata,
        mcp_tool_instance: MCPSDKToolInstance
    ):
        """Initialize an enhanced MCP tool.
        
        Args:
            metadata: Tool metadata
            mcp_tool_instance: MCP tool instance
        """
        super().__init__(metadata=metadata, source=ToolSource.MCP)
        self._mcp_tool_instance = mcp_tool_instance
        
    async def arun(self, **kwargs) -> Any:
        """Run the tool asynchronously.
        
        Args:
            **kwargs: Tool parameters
            
        Returns:
            Result of the tool execution
        """
        return await self._mcp_tool_instance.arun(**kwargs)
        
    def run(self, **kwargs) -> Any:
        """Run the tool synchronously.
        
        Args:
            **kwargs: Tool parameters
            
        Returns:
            Result of the tool execution
        """
        return self._mcp_tool_instance.run(**kwargs)


class MCPSDKToolProvider:
    """Provider for MCP SDK tools."""
    
    def __init__(
        self,
        server_url: str,
        namespace: str = "mcp",
        api_key: Optional[str] = None
    ):
        """Initialize the MCP SDK tool provider.
        
        Args:
            server_url: URL of the MCP server
            namespace: Namespace for the tools
            api_key: Optional API key for authentication
        """
        self.server_url = server_url
        self.namespace = namespace
        self.api_key = api_key
        self.client = MCPSDKClient(base_url=server_url, api_key=api_key)
        self.tools = []
    
    async def initialize(self):
        """Initialize the provider by connecting to the MCP server."""
        await self.client.initialize()
    
    async def discover_and_register_tools(self) -> List[MCPEnhancedTool]:
        """Discover tools from the MCP server and register them.
        
        Returns:
            List of registered enhanced tools
        """
        # Initialize if not already done
        if not self.client._initialized:
            await self.initialize()
        
        # Get available tools
        mcp_tools = self.client.get_available_tools()
        registered_tools = []
        
        # Register each tool
        for tool_def in mcp_tools:
            tool_name = tool_def.get("name")
            prefixed_name = f"{self.namespace}_{tool_name}"
            description = tool_def.get("description", "")
            parameters = tool_def.get("parameters", {})
            
            # Create tool instance
            tool_instance = MCPSDKToolInstance(
                tool_name=tool_name,
                client=self.client,
                description=description,
                parameters=parameters
            )
            
            # Create metadata
            metadata = ToolMetadata(
                name=prefixed_name,
                display_name=tool_name,
                description=description,
                categories=[ToolCategory.UTILITY],  # Default category
                capabilities=[ToolCapability.READ],  # Default capability
                domains=[ToolDomain.GENERAL],       # Default domain
                keywords=[tool_name.lower()],       # Use tool name as keyword
                source=ToolSource.MCP
            )
            
            # Create enhanced tool
            enhanced_tool = MCPEnhancedTool(
                metadata=metadata,
                mcp_tool_instance=tool_instance
            )
            
            # Register with the registry
            registry.register_tool(enhanced_tool)
            registered_tools.append(enhanced_tool)
            logger.info(f"Registered MCP tool: {prefixed_name}")
        
        self.tools = registered_tools
        return registered_tools
    
    async def close(self):
        """Close the provider and its client."""
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