"""
MCP client implementation using the Python MCP SDK.
"""
import asyncio
import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple, Union

from agents.mcp_client.client import MCPClient, MCPServerConfig, MCPTransportType

logger = logging.getLogger(__name__)

try:
    # Attempt to import MCP SDK
    from mcp import ClientSession, StdioServerParameters, SseServerParameters
    from mcp.client.stdio import stdio_client
    from mcp.client.sse import sse_client
    HAS_MCP_SDK = True
except ImportError:
    logger.warning("MCP SDK not installed. Install with 'pip install mcp'")
    HAS_MCP_SDK = False


class MCPSDKClient(MCPClient):
    """
    MCP client implementation using the Python MCP SDK.
    
    This client provides a complete implementation of the MCP client interface
    using the official MCP Python SDK.
    """
    
    def __init__(self, server_config: MCPServerConfig):
        """
        Initialize the MCP SDK client.
        
        Args:
            server_config: Configuration for connecting to an MCP server
        """
        super().__init__(server_config)
        self._client_session = None
        self._read_stream = None
        self._write_stream = None
        
        if not HAS_MCP_SDK:
            raise ImportError("MCP SDK not installed. Install with 'pip install mcp'")
    
    async def connect(self) -> bool:
        """
        Connect to the MCP server defined in the server_config.
        
        Returns:
            True if connection was successful, False otherwise
        """
        try:
            if self.server_config.transport_type == MCPTransportType.STDIO:
                if not self.server_config.command:
                    raise ValueError("Command is required for STDIO transport")
                
                server_params = StdioServerParameters(
                    command=self.server_config.command,
                    args=self.server_config.args,
                    env=self.server_config.env
                )
                
                self._read_stream, self._write_stream = await stdio_client(server_params)
                
            elif self.server_config.transport_type == MCPTransportType.SSE:
                if not self.server_config.url:
                    raise ValueError("URL is required for SSE transport")
                
                server_params = SseServerParameters(
                    url=self.server_config.url
                )
                
                self._read_stream, self._write_stream = await sse_client(server_params)
                
            else:
                raise ValueError(f"Unsupported transport type: {self.server_config.transport_type}")
            
            # Create a client session
            self._client_session = ClientSession(
                self._read_stream,
                self._write_stream,
                sampling_callback=self._handle_sampling
            )
            
            # Initialize the session
            await self._client_session.initialize()
            
            self._initialized = True
            logger.info(f"Connected to MCP server: {self.server_config.name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to MCP server: {e}")
            return False
    
    async def disconnect(self) -> bool:
        """
        Disconnect from the MCP server.
        
        Returns:
            True if disconnect was successful, False otherwise
        """
        try:
            if self._client_session:
                await self._client_session.__aexit__(None, None, None)
                
            self._initialized = False
            self._client_session = None
            self._read_stream = None
            self._write_stream = None
            
            logger.info(f"Disconnected from MCP server: {self.server_config.name}")
            return True
        except Exception as e:
            logger.error(f"Failed to disconnect from MCP server: {e}")
            return False
    
    async def list_tools(self) -> List[Dict[str, Any]]:
        """
        List available tools from the MCP server.
        
        Returns:
            List of available tools
        """
        if not self._initialized:
            raise RuntimeError("Client not connected to MCP server")
            
        try:
            result = await self._client_session.list_tools()
            return result.tools
        except Exception as e:
            logger.error(f"Failed to list tools: {e}")
            return []
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """
        Call a tool on the MCP server.
        
        Args:
            tool_name: Name of the tool to call
            arguments: Arguments for the tool
            
        Returns:
            Result of the tool call
        """
        if not self._initialized:
            raise RuntimeError("Client not connected to MCP server")
            
        try:
            result = await self._client_session.call_tool(tool_name, arguments)
            return result
        except Exception as e:
            logger.error(f"Failed to call tool '{tool_name}': {e}")
            raise
    
    async def list_resources(self) -> List[Dict[str, Any]]:
        """
        List available resources from the MCP server.
        
        Returns:
            List of available resources
        """
        if not self._initialized:
            raise RuntimeError("Client not connected to MCP server")
            
        try:
            result = await self._client_session.list_resources()
            return result.resources
        except Exception as e:
            logger.error(f"Failed to list resources: {e}")
            return []
    
    async def read_resource(self, resource_uri: str) -> Tuple[Any, str]:
        """
        Read a resource from the MCP server.
        
        Args:
            resource_uri: URI of the resource to read
            
        Returns:
            Tuple of (resource content, MIME type)
        """
        if not self._initialized:
            raise RuntimeError("Client not connected to MCP server")
            
        try:
            content, mime_type = await self._client_session.read_resource(resource_uri)
            return content, mime_type
        except Exception as e:
            logger.error(f"Failed to read resource '{resource_uri}': {e}")
            raise
    
    async def list_prompts(self) -> List[Dict[str, Any]]:
        """
        List available prompts from the MCP server.
        
        Returns:
            List of available prompts
        """
        if not self._initialized:
            raise RuntimeError("Client not connected to MCP server")
            
        try:
            result = await self._client_session.list_prompts()
            return result.prompts
        except Exception as e:
            logger.error(f"Failed to list prompts: {e}")
            return []
    
    async def get_prompt(self, prompt_name: str, arguments: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Get a prompt from the MCP server.
        
        Args:
            prompt_name: Name of the prompt to get
            arguments: Arguments for the prompt
            
        Returns:
            Prompt content
        """
        if not self._initialized:
            raise RuntimeError("Client not connected to MCP server")
            
        try:
            result = await self._client_session.get_prompt(prompt_name, arguments or {})
            return result
        except Exception as e:
            logger.error(f"Failed to get prompt '{prompt_name}': {e}")
            raise
    
    async def _handle_sampling(self, message):
        """
        Handle sampling requests from the MCP server.
        
        This is called when the MCP server requests the client to sample from an LLM.
        
        Args:
            message: Sampling request message
        
        Returns:
            Sampling result message
        """
        # Placeholder implementation - in a real implementation, this would
        # actually perform sampling using a specified LLM provider
        
        logger.info(f"Received sampling request: {message}")
        
        # Return a dummy response
        return {
            "role": "assistant",
            "content": {
                "type": "text",
                "text": "This is a placeholder response from the sampling callback."
            },
            "model": "dummy-model",
            "stopReason": "endTurn"
        } 