"""
MCP tool for interfacing with MCP servers.
"""
import asyncio
import json
import logging
import os
from typing import Any, Dict, List, Optional, Union

from pydantic import Field

from tools.tool import Tool
from agents.mcp_client.client import MCPTransportType, MCPServerConfig
from agents.mcp_client.mcp_sdk_client import MCPSDKClient
from agents.mcp_client.sequential_thinking_client import SequentialThinkingMCPClient

logger = logging.getLogger(__name__)


class MCPToolBase(Tool):
    """Base class for MCP tools."""
    
    def __init__(self):
        super().__init__()
        self.clients = {}  # Dictionary of server_name -> client
    
    async def _get_client(self, server_name: str) -> MCPSDKClient:
        """
        Get an MCP client for the specified server.
        
        Args:
            server_name: Name of the MCP server to connect to
            
        Returns:
            MCPSDKClient instance
        """
        if server_name in self.clients and self.clients[server_name]._initialized:
            return self.clients[server_name]
            
        # Create a new client
        client_config = self._get_server_config(server_name)
        
        # Use sequential thinking client if available
        try:
            client = SequentialThinkingMCPClient(client_config)
        except ImportError:
            client = MCPSDKClient(client_config)
            
        # Connect to the server
        success = await client.connect()
        if not success:
            raise RuntimeError(f"Failed to connect to MCP server: {server_name}")
            
        # Store the client
        self.clients[server_name] = client
        return client
    
    def _get_server_config(self, server_name: str) -> MCPServerConfig:
        """
        Get the configuration for an MCP server.
        
        Args:
            server_name: Name of the MCP server
            
        Returns:
            MCPServerConfig instance
        """
        # This is a placeholder implementation. In a real implementation,
        # this would read from a configuration file or environment variables.
        
        # For demonstration, let's define a couple of example servers
        if server_name == "test_server":
            return MCPServerConfig(
                name="test_server",
                transport_type=MCPTransportType.STDIO,
                command="python",
                args=["-m", "mcp.tools.echo_server"]
            )
        elif server_name == "github":
            return MCPServerConfig(
                name="github",
                transport_type=MCPTransportType.STDIO,
                command="python",
                args=["-m", "mcp_github_cli"]
            )
        else:
            # Default to a local test server
            return MCPServerConfig(
                name=server_name,
                transport_type=MCPTransportType.STDIO,
                command="python",
                args=["-m", "mcp.tools.echo_server"]
            )


class MCPListTools(MCPToolBase):
    """Tool for listing available tools from an MCP server."""
    
    server_name: str = Field(
        default="test_server",
        description="Name of the MCP server to connect to"
    )
    
    async def arun(self, server_name: str = "test_server") -> Any:
        """
        List available tools from an MCP server.
        
        Args:
            server_name: Name of the MCP server to connect to
            
        Returns:
            List of available tools
        """
        client = await self._get_client(server_name)
        tools = await client.list_tools()
        return json.dumps(tools, indent=2)
    
    def run(self, server_name: str = "test_server") -> Any:
        """
        Synchronously list available tools from an MCP server.
        
        Args:
            server_name: Name of the MCP server to connect to
            
        Returns:
            List of available tools
        """
        return asyncio.run(self.arun(server_name))


class MCPCallTool(MCPToolBase):
    """Tool for calling a tool on an MCP server."""
    
    server_name: str = Field(
        default="test_server",
        description="Name of the MCP server to connect to"
    )
    tool_name: str = Field(
        ...,
        description="Name of the tool to call"
    )
    arguments: Dict[str, Any] = Field(
        default_factory=dict,
        description="Arguments for the tool call"
    )
    
    async def arun(self, server_name: str = "test_server", tool_name: str = None, arguments: Dict[str, Any] = None) -> Any:
        """
        Call a tool on an MCP server.
        
        Args:
            server_name: Name of the MCP server to connect to
            tool_name: Name of the tool to call
            arguments: Arguments for the tool call
            
        Returns:
            Result of the tool call
        """
        client = await self._get_client(server_name)
        result = await client.call_tool(tool_name, arguments or {})
        
        # Format the result nicely
        if hasattr(result, "content") and hasattr(result.content, "text"):
            return result.content.text
        elif hasattr(result, "text"):
            return result.text
        else:
            return str(result)
    
    def run(self, server_name: str = "test_server", tool_name: str = None, arguments: Dict[str, Any] = None) -> Any:
        """
        Synchronously call a tool on an MCP server.
        
        Args:
            server_name: Name of the MCP server to connect to
            tool_name: Name of the tool to call
            arguments: Arguments for the tool call
            
        Returns:
            Result of the tool call
        """
        return asyncio.run(self.arun(server_name, tool_name, arguments))


class MCPSequentialThinking(MCPToolBase):
    """Tool for solving problems using sequential thinking with an MCP server."""
    
    server_name: str = Field(
        default="test_server",
        description="Name of the MCP server to connect to"
    )
    problem: str = Field(
        ...,
        description="Problem statement to solve using sequential thinking"
    )
    max_thoughts: int = Field(
        default=5,
        description="Maximum number of thoughts to generate"
    )
    
    async def arun(self, server_name: str = "test_server", problem: str = None, max_thoughts: int = 5) -> Any:
        """
        Solve a problem using sequential thinking with an MCP server.
        
        Args:
            server_name: Name of the MCP server to connect to
            problem: Problem statement to solve using sequential thinking
            max_thoughts: Maximum number of thoughts to generate
            
        Returns:
            Result of the sequential thinking process
        """
        client = await self._get_client(server_name)
        
        # Check if the client supports sequential thinking
        if not isinstance(client, SequentialThinkingMCPClient):
            return "Error: Server client does not support sequential thinking"
            
        result = await client.solve_problem_sequentially(problem, max_thoughts)
        
        # Format the result as a nice markdown output
        output = f"# Sequential Thinking: {problem}\n\n"
        
        for thought in result["thoughts"]:
            output += f"## Thought {thought['number']}\n\n{thought['content']}\n\n"
            
        output += f"## Conclusion\n\n{result['conclusion']}\n"
        
        return output
    
    def run(self, server_name: str = "test_server", problem: str = None, max_thoughts: int = 5) -> Any:
        """
        Synchronously solve a problem using sequential thinking with an MCP server.
        
        Args:
            server_name: Name of the MCP server to connect to
            problem: Problem statement to solve using sequential thinking
            max_thoughts: Maximum number of thoughts to generate
            
        Returns:
            Result of the sequential thinking process
        """
        return asyncio.run(self.arun(server_name, problem, max_thoughts)) 