"""
Multi-Provider MCP client implementation for connecting to different MCP servers.

This client allows interaction with various MCP servers, including Zapier, Zendesk,
and other providers, making it easy to switch between different integrations.

For more information: https://modelcontextprotocol.io
"""
import asyncio
import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple, Union
from enum import Enum

from agents.mcp_client.client import MCPClient, MCPServerConfig, MCPTransportType
from agents.mcp_client.mcp_sdk_client import MCPSDKClient

logger = logging.getLogger(__name__)


class MCPProviderType(Enum):
    """Enum for supported MCP server providers."""
    ZAPIER = "zapier"
    ZENDESK = "zendesk"
    GITHUB = "github"
    NOTION = "notion"
    CUSTOM = "custom"


class MultiProviderMCPClient(MCPSDKClient):
    """
    Multi-Provider MCP client for interacting with different MCP servers.
    
    This client extends the standard MCPSDKClient to provide specialized
    functionality for different MCP providers, handling provider-specific
    endpoint formats and authentication methods.
    """
    
    def __init__(self, 
                 mcp_url: str,
                 provider_type: Union[MCPProviderType, str] = MCPProviderType.CUSTOM,
                 api_key: Optional[str] = None,
                 server_name: Optional[str] = None):
        """
        Initialize the Multi-Provider MCP client.
        
        Args:
            mcp_url: URL of the MCP server endpoint
            provider_type: Type of MCP provider (zapier, zendesk, etc.)
            api_key: Optional API key for authentication
            server_name: Name for this server configuration (defaults to provider type + timestamp)
        """
        # Convert string to enum if necessary
        if isinstance(provider_type, str):
            try:
                provider_type = MCPProviderType(provider_type.lower())
            except ValueError:
                provider_type = MCPProviderType.CUSTOM
        
        # Set default server name if not provided
        if server_name is None:
            import time
            timestamp = int(time.time())
            server_name = f"{provider_type.value}-mcp-{timestamp}"
        
        # Determine the correct transport type based on provider
        transport_type = self._get_transport_type_for_provider(provider_type)
        
        # Create server config
        server_config = MCPServerConfig(
            name=server_name,
            transport_type=transport_type,
            url=mcp_url if transport_type == MCPTransportType.SSE else None,
            command=self._get_command_for_provider(provider_type) if transport_type == MCPTransportType.STDIO else None,
            args=self._get_args_for_provider(provider_type, mcp_url) if transport_type == MCPTransportType.STDIO else None,
            env={"API_KEY": api_key} if api_key else {}
        )
        
        super().__init__(server_config)
        self.mcp_url = mcp_url
        self.provider_type = provider_type
        self.api_key = api_key
    
    @staticmethod
    def _get_transport_type_for_provider(provider_type: MCPProviderType) -> MCPTransportType:
        """
        Get the appropriate transport type for a provider.
        
        Args:
            provider_type: Type of MCP provider
            
        Returns:
            Appropriate MCPTransportType
        """
        # Most modern MCP servers use SSE (Server-Sent Events)
        provider_transport_map = {
            MCPProviderType.ZAPIER: MCPTransportType.SSE,
            MCPProviderType.ZENDESK: MCPTransportType.SSE,
            # Some providers might use STDIO for local development
            MCPProviderType.GITHUB: MCPTransportType.STDIO,
            MCPProviderType.NOTION: MCPTransportType.SSE,
            MCPProviderType.CUSTOM: MCPTransportType.SSE
        }
        
        return provider_transport_map.get(provider_type, MCPTransportType.SSE)
    
    @staticmethod
    def _get_command_for_provider(provider_type: MCPProviderType) -> Optional[str]:
        """
        Get the command to run a local MCP server for STDIO transport.
        
        Args:
            provider_type: Type of MCP provider
            
        Returns:
            Command string or None if not applicable
        """
        # Only needed for STDIO transport
        if provider_type == MCPProviderType.GITHUB:
            return "python"  # Or the appropriate command
        return None
    
    @staticmethod
    def _get_args_for_provider(provider_type: MCPProviderType, url: str) -> List[str]:
        """
        Get the arguments for a local MCP server command for STDIO transport.
        
        Args:
            provider_type: Type of MCP provider
            url: URL or path for the MCP server
            
        Returns:
            List of arguments or empty list if not applicable
        """
        # Only needed for STDIO transport
        if provider_type == MCPProviderType.GITHUB:
            return ["-m", "mcp_github_cli"]  # Or the appropriate arguments
        return []
    
    async def list_available_tools(self) -> List[Dict[str, Any]]:
        """
        List all available tools from the MCP server with provider context.
        
        Returns:
            List of available tools with provider information
        """
        tools = await self.list_tools()
        
        # Add provider context
        return [{
            **tool,
            "provider": self.provider_type.value,
            f"is_{self.provider_type.value}_tool": True
        } for tool in tools]
    
    async def execute_tool(self, 
                          tool_name: str, 
                          parameters: Dict[str, Any]) -> Any:
        """
        Execute a tool through the MCP server.
        
        Args:
            tool_name: Name of the tool to execute
            parameters: Parameters for the tool
            
        Returns:
            Result of the tool execution
        """
        return await self.call_tool(tool_name, parameters)
    
    @classmethod
    async def create_and_test_connection(cls, 
                                         mcp_url: str,
                                         provider_type: Union[MCPProviderType, str] = MCPProviderType.CUSTOM,
                                         api_key: Optional[str] = None,
                                         server_name: Optional[str] = None) -> Tuple[bool, Union["MultiProviderMCPClient", str]]:
        """
        Create a client and test the connection to an MCP server.
        
        This is a convenience method that creates a client and tests the connection
        in one go, returning either the connected client or an error message.
        
        Args:
            mcp_url: URL of the MCP server endpoint
            provider_type: Type of MCP provider
            api_key: Optional API key for authentication
            server_name: Name for this server configuration
            
        Returns:
            Tuple of (success, client_or_error_message)
        """
        try:
            client = cls(mcp_url, provider_type, api_key, server_name)
            success = await client.connect()
            
            if success:
                # Test by listing tools to ensure we can communicate
                tools = await client.list_tools()
                if tools:
                    return True, client
                else:
                    await client.disconnect()
                    return False, f"Connected to {provider_type} MCP server but no tools were returned"
            else:
                return False, f"Failed to connect to {provider_type} MCP server"
        except Exception as e:
            logger.error(f"Error connecting to {provider_type} MCP: {e}")
            return False, f"Error: {str(e)}"
    
    @staticmethod
    async def get_mcp_url_from_setup(provider_type: Union[MCPProviderType, str] = MCPProviderType.CUSTOM) -> str:
        """
        Get the MCP URL from an interactive setup process.
        
        Args:
            provider_type: Type of MCP provider
            
        Returns:
            The MCP URL
        """
        # Convert string to enum if necessary
        if isinstance(provider_type, str):
            try:
                provider_type = MCPProviderType(provider_type.lower())
            except ValueError:
                provider_type = MCPProviderType.CUSTOM
        
        provider_name = provider_type.value.capitalize()
        setup_instructions = {
            MCPProviderType.ZAPIER: [
                f"Go to https://zapier.com/mcp and sign in",
                f"Click 'Get started'",
                f"Generate your MCP endpoint",
                f"Copy the URL"
            ],
            MCPProviderType.ZENDESK: [
                f"Log in to your Zendesk account",
                f"Navigate to the Admin Center",
                f"Find the MCP integration settings",
                f"Generate your MCP endpoint",
                f"Copy the URL"
            ],
            MCPProviderType.GITHUB: [
                f"Install the GitHub MCP server",
                f"Generate your access token with proper permissions",
                f"Set up your MCP endpoint",
                f"Copy the URL or path"
            ],
            MCPProviderType.NOTION: [
                f"Go to https://www.notion.so/my-integrations",
                f"Create a new integration",
                f"Copy your integration token",
                f"Use it with the Notion MCP server"
            ],
            MCPProviderType.CUSTOM: [
                f"Follow your provider's instructions to set up an MCP endpoint",
                f"Ensure you have the correct authentication credentials if needed"
            ]
        }
        
        instructions = setup_instructions.get(provider_type, setup_instructions[MCPProviderType.CUSTOM])
        
        print(f"To use {provider_name} MCP, you need to set up an MCP endpoint")
        print("Follow these steps:")
        for i, step in enumerate(instructions, 1):
            print(f"{i}. {step}")
        
        mcp_url = input(f"Enter your {provider_name} MCP URL: ")
        return mcp_url.strip()


# Convenient factory functions for specific providers

async def create_zapier_mcp_client(mcp_url: str, api_key: Optional[str] = None, server_name: Optional[str] = None) -> "MultiProviderMCPClient":
    """
    Create and connect to a Zapier MCP server.
    
    Args:
        mcp_url: URL of the Zapier MCP server endpoint
        api_key: Optional API key for authentication
        server_name: Name for this server configuration
        
    Returns:
        Connected MultiProviderMCPClient for Zapier
        
    Raises:
        RuntimeError: If connection fails
    """
    success, result = await MultiProviderMCPClient.create_and_test_connection(
        mcp_url=mcp_url,
        provider_type=MCPProviderType.ZAPIER,
        api_key=api_key,
        server_name=server_name
    )
    
    if success:
        return result
    else:
        raise RuntimeError(f"Failed to connect to Zapier MCP: {result}")

async def create_zendesk_mcp_client(mcp_url: str, api_key: Optional[str] = None, server_name: Optional[str] = None) -> "MultiProviderMCPClient":
    """
    Create and connect to a Zendesk MCP server.
    
    Args:
        mcp_url: URL of the Zendesk MCP server endpoint
        api_key: Optional API key for authentication
        server_name: Name for this server configuration
        
    Returns:
        Connected MultiProviderMCPClient for Zendesk
        
    Raises:
        RuntimeError: If connection fails
    """
    success, result = await MultiProviderMCPClient.create_and_test_connection(
        mcp_url=mcp_url,
        provider_type=MCPProviderType.ZENDESK,
        api_key=api_key,
        server_name=server_name
    )
    
    if success:
        return result
    else:
        raise RuntimeError(f"Failed to connect to Zendesk MCP: {result}") 