"""
Multi-Provider MCP tool for accessing various MCP servers.

This tool allows AI agents to interact with different MCP servers
from providers like Zapier, Zendesk, GitHub, Notion, and more.
"""
import asyncio
import json
import logging
import os
from typing import Any, Dict, List, Optional, Union

from pydantic import Field

from tools.tool import Tool
from tools.multi_provider_mcp_client import (
    MultiProviderMCPClient,
    MCPProviderType,
    create_zapier_mcp_client,
    create_zendesk_mcp_client
)

logger = logging.getLogger(__name__)


class MultiProviderMCPTool(Tool):
    """
    Tool for connecting to and interacting with various MCP servers.
    
    Allows AI agents to interact with different MCP providers through
    a unified interface, supporting Zapier, Zendesk, and more.
    """
    
    _client_cache = {}  # Class-level cache of provider:url -> client
    
    def __init__(self):
        super().__init__()
    
    @classmethod
    async def _get_client(cls, 
                         mcp_url: str, 
                         provider_type: Union[MCPProviderType, str] = MCPProviderType.CUSTOM,
                         api_key: Optional[str] = None) -> MultiProviderMCPClient:
        """
        Get or create a MultiProviderMCPClient for the specified URL and provider.
        
        Args:
            mcp_url: URL of the MCP server endpoint
            provider_type: Type of MCP provider
            api_key: Optional API key for authentication
            
        Returns:
            MultiProviderMCPClient instance
        
        Raises:
            RuntimeError: If connection fails
        """
        # Convert string to enum if necessary
        if isinstance(provider_type, str):
            try:
                provider_type = MCPProviderType(provider_type.lower())
            except ValueError:
                provider_type = MCPProviderType.CUSTOM
                
        # Check if we already have a connected client for this URL and provider in the cache
        cache_key = f"{provider_type.value}:{mcp_url}:{api_key}"
        if cache_key in cls._client_cache and cls._client_cache[cache_key]._initialized:
            return cls._client_cache[cache_key]
            
        # Create a new client and test connection
        success, result = await MultiProviderMCPClient.create_and_test_connection(
            mcp_url=mcp_url,
            provider_type=provider_type,
            api_key=api_key
        )
        
        if success:
            client = result
            cls._client_cache[cache_key] = client
            return client
        else:
            error_msg = result
            raise RuntimeError(f"Failed to connect to {provider_type.value} MCP: {error_msg}")


class MCPListTools(MultiProviderMCPTool):
    """Tool for listing available tools from an MCP server."""
    
    mcp_url: str = Field(
        ...,
        description="URL of the MCP server endpoint"
    )
    provider_type: str = Field(
        "custom",
        description="Type of MCP provider (zapier, zendesk, github, notion, custom)"
    )
    api_key: Optional[str] = Field(
        None,
        description="Optional API key for authentication"
    )
    
    async def arun(self, mcp_url: str, provider_type: str = "custom", api_key: Optional[str] = None) -> Any:
        """
        List available tools from an MCP server.
        
        Args:
            mcp_url: URL of the MCP server endpoint
            provider_type: Type of MCP provider
            api_key: Optional API key for authentication
            
        Returns:
            JSON string containing the list of available tools
        """
        client = await self._get_client(mcp_url, provider_type, api_key)
        tools = await client.list_available_tools()
        
        # Format the tools for easier reading
        formatted_tools = []
        for tool in tools:
            formatted = {
                "name": tool.get("name", "Unknown"),
                "provider": tool.get("provider", provider_type),
                "description": tool.get("description", "No description"),
                "parameters": [
                    {
                        "name": param.get("name", "unknown"),
                        "description": param.get("description", "No description"),
                        "required": param.get("required", False)
                    }
                    for param in tool.get("parameters", {}).get("properties", {}).values()
                ]
            }
            formatted_tools.append(formatted)
            
        return json.dumps(formatted_tools, indent=2)
    
    def run(self, mcp_url: str, provider_type: str = "custom", api_key: Optional[str] = None) -> Any:
        """
        Synchronously list available tools from an MCP server.
        
        Args:
            mcp_url: URL of the MCP server endpoint
            provider_type: Type of MCP provider
            api_key: Optional API key for authentication
            
        Returns:
            JSON string containing the list of available tools
        """
        return asyncio.run(self.arun(mcp_url, provider_type, api_key))


class MCPExecuteTool(MultiProviderMCPTool):
    """Tool for executing a tool on an MCP server."""
    
    mcp_url: str = Field(
        ...,
        description="URL of the MCP server endpoint"
    )
    tool_name: str = Field(
        ...,
        description="Name of the tool to execute"
    )
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Parameters for the tool"
    )
    provider_type: str = Field(
        "custom",
        description="Type of MCP provider (zapier, zendesk, github, notion, custom)"
    )
    api_key: Optional[str] = Field(
        None,
        description="Optional API key for authentication"
    )
    
    async def arun(self, mcp_url: str, tool_name: str, parameters: Dict[str, Any] = None, provider_type: str = "custom", api_key: Optional[str] = None) -> Any:
        """
        Execute a tool on an MCP server.
        
        Args:
            mcp_url: URL of the MCP server endpoint
            tool_name: Name of the tool to execute
            parameters: Parameters for the tool
            provider_type: Type of MCP provider
            api_key: Optional API key for authentication
            
        Returns:
            Result of the tool execution
        """
        client = await self._get_client(mcp_url, provider_type, api_key)
        result = await client.execute_tool(tool_name, parameters or {})
        
        # Format the result for easier reading
        if hasattr(result, "content") and hasattr(result.content, "text"):
            return result.content.text
        elif hasattr(result, "text"):
            return result.text
        else:
            return json.dumps(result, indent=2) if isinstance(result, (dict, list)) else str(result)
    
    def run(self, mcp_url: str, tool_name: str, parameters: Dict[str, Any] = None, provider_type: str = "custom", api_key: Optional[str] = None) -> Any:
        """
        Synchronously execute a tool on an MCP server.
        
        Args:
            mcp_url: URL of the MCP server endpoint
            tool_name: Name of the tool to execute
            parameters: Parameters for the tool
            provider_type: Type of MCP provider
            api_key: Optional API key for authentication
            
        Returns:
            Result of the tool execution
        """
        return asyncio.run(self.arun(mcp_url, tool_name, parameters, provider_type, api_key))


class MCPSetupProvider(MultiProviderMCPTool):
    """Tool for setting up and testing a connection to an MCP server."""
    
    provider_type: str = Field(
        "custom",
        description="Type of MCP provider (zapier, zendesk, github, notion, custom)"
    )
    
    async def arun(self, provider_type: str = "custom") -> str:
        """
        Set up a connection to an MCP server interactively.
        
        Args:
            provider_type: Type of MCP provider
            
        Returns:
            Success message with the MCP URL if successful
        """
        try:
            # Get the MCP URL from the user
            mcp_url = await MultiProviderMCPClient.get_mcp_url_from_setup(provider_type)
            
            # Test the connection
            print(f"Testing connection to {provider_type} MCP server at {mcp_url}...")
            success, result = await MultiProviderMCPClient.create_and_test_connection(
                mcp_url=mcp_url,
                provider_type=provider_type
            )
            
            if success:
                client = result
                # List the available tools
                tools = await client.list_available_tools()
                await client.disconnect()
                
                return (
                    f"Successfully connected to {provider_type} MCP server at {mcp_url}\n"
                    f"Found {len(tools)} available tools.\n\n"
                    f"You can now use the MCPListTools and MCPExecuteTool tools "
                    f"with this URL and provider type to interact with the server."
                )
            else:
                error_msg = result
                return f"Failed to connect to {provider_type} MCP server: {error_msg}"
        except Exception as e:
            logger.error(f"Error setting up {provider_type} MCP: {e}")
            return f"Error setting up {provider_type} MCP: {str(e)}"
    
    def run(self, provider_type: str = "custom") -> str:
        """
        Synchronously set up a connection to an MCP server interactively.
        
        Args:
            provider_type: Type of MCP provider
            
        Returns:
            Success message with the MCP URL if successful
        """
        return asyncio.run(self.arun(provider_type)) 