"""
Zapier MCP tool for accessing 7,000+ apps through Model Context Protocol.

This tool allows AI agents to interact with Zapier's MCP servers,
providing access to thousands of apps and actions without complex API integrations.
"""
import asyncio
import json
import logging
import os
from typing import Any, Dict, List, Optional, Union

from pydantic import Field

from tools.tool import Tool
from tools.zapier_mcp_client import ZapierMCPClient

logger = logging.getLogger(__name__)


class ZapierMCPTool(Tool):
    """
    Tool for connecting to and interacting with Zapier MCP servers.
    
    Allows AI agents to access 7,000+ apps and 30,000+ actions through
    Zapier's MCP implementation.
    """
    
    _client_cache = {}  # Class-level cache of server_url -> client
    
    def __init__(self):
        super().__init__()
    
    @classmethod
    async def _get_client(cls, zapier_mcp_url: str, api_key: Optional[str] = None) -> ZapierMCPClient:
        """
        Get or create a ZapierMCPClient for the specified URL.
        
        Args:
            zapier_mcp_url: URL of the Zapier MCP server endpoint
            api_key: Optional API key for authentication
            
        Returns:
            ZapierMCPClient instance
        
        Raises:
            RuntimeError: If connection fails
        """
        # Check if we already have a connected client for this URL in the cache
        cache_key = f"{zapier_mcp_url}:{api_key}"
        if cache_key in cls._client_cache and cls._client_cache[cache_key]._initialized:
            return cls._client_cache[cache_key]
            
        # Create a new client and test connection
        success, result = await ZapierMCPClient.create_and_test_connection(
            zapier_mcp_url=zapier_mcp_url,
            api_key=api_key
        )
        
        if success:
            client = result
            cls._client_cache[cache_key] = client
            return client
        else:
            error_msg = result
            raise RuntimeError(f"Failed to connect to Zapier MCP: {error_msg}")


class ZapierListActions(ZapierMCPTool):
    """Tool for listing available actions from a Zapier MCP server."""
    
    zapier_mcp_url: str = Field(
        ...,
        description="URL of the Zapier MCP server endpoint"
    )
    api_key: Optional[str] = Field(
        None,
        description="Optional API key for authentication"
    )
    
    async def arun(self, zapier_mcp_url: str, api_key: Optional[str] = None) -> Any:
        """
        List available actions from a Zapier MCP server.
        
        Args:
            zapier_mcp_url: URL of the Zapier MCP server endpoint
            api_key: Optional API key for authentication
            
        Returns:
            JSON string containing the list of available actions
        """
        client = await self._get_client(zapier_mcp_url, api_key)
        actions = await client.list_available_actions()
        
        # Format the actions for easier reading
        formatted_actions = []
        for action in actions:
            formatted = {
                "name": action.get("name", "Unknown"),
                "app": action.get("app", "Unknown"),
                "description": action.get("description", "No description"),
                "parameters": [
                    {
                        "name": param.get("name", "unknown"),
                        "description": param.get("description", "No description"),
                        "required": param.get("required", False)
                    }
                    for param in action.get("parameters", {}).get("properties", {}).values()
                ]
            }
            formatted_actions.append(formatted)
            
        return json.dumps(formatted_actions, indent=2)
    
    def run(self, zapier_mcp_url: str, api_key: Optional[str] = None) -> Any:
        """
        Synchronously list available actions from a Zapier MCP server.
        
        Args:
            zapier_mcp_url: URL of the Zapier MCP server endpoint
            api_key: Optional API key for authentication
            
        Returns:
            JSON string containing the list of available actions
        """
        return asyncio.run(self.arun(zapier_mcp_url, api_key))


class ZapierExecuteAction(ZapierMCPTool):
    """Tool for executing an action on a Zapier MCP server."""
    
    zapier_mcp_url: str = Field(
        ...,
        description="URL of the Zapier MCP server endpoint"
    )
    action_name: str = Field(
        ...,
        description="Name of the Zapier action to execute"
    )
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Parameters for the action"
    )
    api_key: Optional[str] = Field(
        None,
        description="Optional API key for authentication"
    )
    
    async def arun(self, zapier_mcp_url: str, action_name: str, parameters: Dict[str, Any] = None, api_key: Optional[str] = None) -> Any:
        """
        Execute an action on a Zapier MCP server.
        
        Args:
            zapier_mcp_url: URL of the Zapier MCP server endpoint
            action_name: Name of the Zapier action to execute
            parameters: Parameters for the action
            api_key: Optional API key for authentication
            
        Returns:
            Result of the action execution
        """
        client = await self._get_client(zapier_mcp_url, api_key)
        result = await client.execute_zapier_action(action_name, parameters or {})
        
        # Format the result for easier reading
        if hasattr(result, "content") and hasattr(result.content, "text"):
            return result.content.text
        elif hasattr(result, "text"):
            return result.text
        else:
            return json.dumps(result, indent=2) if isinstance(result, (dict, list)) else str(result)
    
    def run(self, zapier_mcp_url: str, action_name: str, parameters: Dict[str, Any] = None, api_key: Optional[str] = None) -> Any:
        """
        Synchronously execute an action on a Zapier MCP server.
        
        Args:
            zapier_mcp_url: URL of the Zapier MCP server endpoint
            action_name: Name of the Zapier action to execute
            parameters: Parameters for the action
            api_key: Optional API key for authentication
            
        Returns:
            Result of the action execution
        """
        return asyncio.run(self.arun(zapier_mcp_url, action_name, parameters, api_key))


class ZapierSetupMCP(ZapierMCPTool):
    """Tool for setting up and testing a connection to a Zapier MCP server."""
    
    async def arun(self) -> str:
        """
        Set up a connection to a Zapier MCP server interactively.
        
        Returns:
            Success message with the Zapier MCP URL if successful
        """
        try:
            # Get the Zapier MCP URL from the user
            zapier_mcp_url = await ZapierMCPClient.get_zapier_mcp_url_from_setup()
            
            # Test the connection
            print(f"Testing connection to {zapier_mcp_url}...")
            success, result = await ZapierMCPClient.create_and_test_connection(zapier_mcp_url)
            
            if success:
                client = result
                # List the available actions
                actions = await client.list_available_actions()
                await client.disconnect()
                
                return (
                    f"Successfully connected to Zapier MCP server at {zapier_mcp_url}\n"
                    f"Found {len(actions)} available actions.\n\n"
                    f"You can now use the ZapierListActions and ZapierExecuteAction tools "
                    f"with this URL to interact with Zapier."
                )
            else:
                error_msg = result
                return f"Failed to connect to Zapier MCP server: {error_msg}"
        except Exception as e:
            logger.error(f"Error setting up Zapier MCP: {e}")
            return f"Error setting up Zapier MCP: {str(e)}"
    
    def run(self) -> str:
        """
        Synchronously set up a connection to a Zapier MCP server interactively.
        
        Returns:
            Success message with the Zapier MCP URL if successful
        """
        return asyncio.run(self.arun()) 