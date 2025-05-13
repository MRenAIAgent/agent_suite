"""
Zapier MCP client implementation for connecting to Zapier MCP servers.

This client allows interaction with Zapier's MCP servers to access 7,000+ apps and
30,000+ actions without complex API integrations.

For more information: https://zapier.com/mcp
"""
import asyncio
import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple, Union

from agents.mcp_client.client import MCPClient, MCPServerConfig, MCPTransportType
from agents.mcp_client.mcp_sdk_client import MCPSDKClient

logger = logging.getLogger(__name__)

class ZapierMCPClient(MCPSDKClient):
    """
    Zapier MCP client for interacting with Zapier MCP servers.
    
    This client extends the standard MCPSDKClient to provide specialized
    functionality for Zapier's MCP implementation, including authentication
    and handling of Zapier-specific endpoints.
    """
    
    def __init__(self, 
                 zapier_mcp_url: str,
                 api_key: Optional[str] = None,
                 server_name: str = "zapier-mcp"):
        """
        Initialize the Zapier MCP client.
        
        Args:
            zapier_mcp_url: URL of the Zapier MCP server endpoint
            api_key: Optional API key for authentication
            server_name: Name for this server configuration
        """
        # Create server config for SSE (Server-Sent Events) transport
        # which is what Zapier MCP uses
        server_config = MCPServerConfig(
            name=server_name,
            transport_type=MCPTransportType.SSE,
            url=zapier_mcp_url,
            env={"API_KEY": api_key} if api_key else {}
        )
        
        super().__init__(server_config)
        self.zapier_mcp_url = zapier_mcp_url
        self.api_key = api_key
    
    async def list_available_actions(self) -> List[Dict[str, Any]]:
        """
        List all available actions from the Zapier MCP server.
        
        This is a convenience method that wraps list_tools() but provides
        more context about Zapier's actions.
        
        Returns:
            List of available Zapier actions
        """
        tools = await self.list_tools()
        
        # Add some context about what these tools represent in Zapier
        return [{
            **tool,
            "app": tool.get("name", "").split("_")[0] if "_" in tool.get("name", "") else "Unknown",
            "is_zapier_action": True
        } for tool in tools]
    
    async def execute_zapier_action(self, 
                                   action_name: str, 
                                   parameters: Dict[str, Any]) -> Any:
        """
        Execute a Zapier action through the MCP server.
        
        Args:
            action_name: Name of the Zapier action to execute
            parameters: Parameters for the action
            
        Returns:
            Result of the action execution
        """
        return await self.call_tool(action_name, parameters)
    
    @classmethod
    async def create_and_test_connection(cls, 
                                         zapier_mcp_url: str,
                                         api_key: Optional[str] = None,
                                         server_name: str = "zapier-mcp") -> Tuple[bool, Union["ZapierMCPClient", str]]:
        """
        Create a client and test the connection to Zapier MCP.
        
        This is a convenience method that creates a client and tests the connection
        in one go, returning either the connected client or an error message.
        
        Args:
            zapier_mcp_url: URL of the Zapier MCP server endpoint
            api_key: Optional API key for authentication
            server_name: Name for this server configuration
            
        Returns:
            Tuple of (success, client_or_error_message)
        """
        try:
            client = cls(zapier_mcp_url, api_key, server_name)
            success = await client.connect()
            
            if success:
                # Test by listing tools to ensure we can communicate
                tools = await client.list_tools()
                if tools:
                    return True, client
                else:
                    await client.disconnect()
                    return False, "Connected to server but no tools were returned"
            else:
                return False, "Failed to connect to Zapier MCP server"
        except Exception as e:
            logger.error(f"Error connecting to Zapier MCP: {e}")
            return False, f"Error: {str(e)}"
    
    @staticmethod
    async def get_zapier_mcp_url_from_setup() -> str:
        """
        Get the Zapier MCP URL from the setup process.
        
        Users need to generate their MCP endpoint in the Zapier interface:
        https://zapier.com/mcp
        
        Returns:
            The Zapier MCP URL
        """
        print("To use Zapier MCP, you need to generate an MCP endpoint from https://zapier.com/mcp")
        print("Follow these steps:")
        print("1. Go to https://zapier.com/mcp and sign in")
        print("2. Click 'Get started'")
        print("3. Generate your MCP endpoint")
        print("4. Copy the URL and paste it below")
        
        mcp_url = input("Enter your Zapier MCP URL: ")
        return mcp_url.strip() 