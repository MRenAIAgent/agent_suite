"""
MCP Tool Adapter for integrating tools from MCP servers into the enhanced tool system.

This module provides classes for adapting MCP tools to the EnhancedTool format
and registering them with the tool registry system.
"""
import asyncio
import logging
from typing import Any, Dict, List, Optional, Union, Type

from tools.types import (
    ToolCategory, ToolCapability, ToolDomain, ToolMetadata, ToolSource
)
from tools.base import EnhancedTool
from tools.registry import registry

# Import necessary MCP client classes
from agents.mcp_client.client import MCPClient, MCPServerConfig, MCPTransportType
from tools.multi_provider_mcp_client import (
    MultiProviderMCPClient, MCPProviderType
)

# Configure logging
logger = logging.getLogger(__name__)


class MCPToolAdapter:
    """
    Adapter for integrating MCP tools into the enhanced tool system.
    
    This adapter scans for available tools from an MCP server, creates
    EnhancedTool wrappers for them, and registers them with the tool registry.
    """
    
    def __init__(
        self,
        registry_instance=None,
        namespace_prefix: Optional[str] = None
    ):
        """
        Initialize the MCP tool adapter.
        
        Args:
            registry_instance: Optional custom tool registry to use
            namespace_prefix: Optional custom namespace prefix for MCP tools
        """
        self.registry = registry_instance or registry
        self.namespace_prefix = namespace_prefix or "mcp"
        self.clients = {}  # server_name -> client mapping
        
    async def connect_to_server(
        self,
        mcp_url: str,
        provider_type: Union[MCPProviderType, str] = MCPProviderType.CUSTOM,
        api_key: Optional[str] = None,
        server_name: Optional[str] = None
    ) -> bool:
        """
        Connect to an MCP server and scan for available tools.
        
        Args:
            mcp_url: URL of the MCP server
            provider_type: Type of MCP provider (e.g., GITHUB, ZAPIER, ZENDESK)
            api_key: Optional API key for authentication
            server_name: Optional custom name for this server
            
        Returns:
            True if connection was successful, False otherwise
        """
        # Generate a default server name if not provided
        if not server_name:
            if isinstance(provider_type, str):
                server_name = f"{provider_type.lower()}_server"
            else:
                server_name = f"{provider_type.value}_server"
        
        try:
            # Create a MultiProviderMCPClient
            success, result = await MultiProviderMCPClient.create_and_test_connection(
                mcp_url=mcp_url,
                provider_type=provider_type,
                api_key=api_key,
                server_name=server_name
            )
            
            if success:
                client = result
                self.clients[server_name] = client
                
                # Scan for available tools
                try:
                    await self.scan_server_tools(server_name)
                    return True
                except Exception as e:
                    logger.error(f"Error scanning tools from server {server_name}: {e}")
                    await client.disconnect()
                    del self.clients[server_name]
                    return False
            else:
                error_msg = result
                logger.error(f"Failed to connect to server {server_name}: {error_msg}")
                return False
        
        except Exception as e:
            logger.error(f"Error connecting to server {server_name}: {e}")
            return False
    
    async def scan_server_tools(self, server_name: str) -> List[EnhancedTool]:
        """
        Scan an MCP server for available tools and register them.
        
        Args:
            server_name: Name of the server to scan
            
        Returns:
            List of registered EnhancedTool instances
        """
        client = self.clients.get(server_name)
        if not client:
            raise ValueError(f"Client for server {server_name} not found")
        
        # Get list of available tools from the server
        tool_list = await client.list_tools()
        if not tool_list:
            logger.warning(f"No tools found on server {server_name}")
            return []
        
        # Namespace for these tools
        namespace = f"{self.namespace_prefix}.{server_name}"
        
        # Create EnhancedTool wrappers for each MCP tool
        created_tools = []
        
        for tool_info in tool_list:
            try:
                # Extract basic tool information
                tool_name = tool_info.get("name")
                description = tool_info.get("description", "")
                parameters = tool_info.get("parameters", {})
                
                # Create an EnhancedTool wrapper
                tool_instance = self._create_mcp_tool(
                    tool_name=tool_name,
                    description=description,
                    parameters=parameters,
                    server_name=server_name,
                    client=client
                )
                
                created_tools.append(tool_instance)
                
            except Exception as e:
                logger.error(f"Error creating tool wrapper for {tool_info.get('name')}: {e}")
        
        # Register all created tools under the namespace
        if created_tools:
            self.registry.register_tools_with_namespace(namespace, created_tools)
            logger.info(f"Registered {len(created_tools)} tools from server {server_name}")
        
        return created_tools
    
    def _create_mcp_tool(
        self,
        tool_name: str,
        description: str,
        parameters: Dict[str, Any],
        server_name: str,
        client: MultiProviderMCPClient
    ) -> EnhancedTool:
        """
        Create an EnhancedTool wrapper for an MCP tool.
        
        Args:
            tool_name: Name of the MCP tool
            description: Description of the tool
            parameters: Parameter schema for the tool
            server_name: Name of the server this tool belongs to
            client: MCP client instance
            
        Returns:
            EnhancedTool instance wrapping the MCP tool
        """
        # Define the async execute function for this MCP tool
        async def execute_mcp_tool(**kwargs):
            try:
                result = await client.call_tool(tool_name, kwargs)
                
                # Format the result nicely
                if hasattr(result, "content") and hasattr(result.content, "text"):
                    return result.content.text
                elif hasattr(result, "text"):
                    return result.text
                else:
                    return str(result)
            except Exception as e:
                logger.error(f"Error executing MCP tool {tool_name}: {e}")
                raise
        
        # Create enhanced metadata
        metadata = ToolMetadata(
            categories=[ToolCategory.CUSTOM],
            domains=[ToolDomain.GENERAL],
            capabilities=[ToolCapability.CUSTOM],
            semantic_description=description
        )
        
        # Try to infer more specific metadata from the tool name and description
        if any(kw in tool_name.lower() for kw in ["search", "find", "lookup", "query"]):
            metadata.categories.append(ToolCategory.SEARCH)
            metadata.capabilities.append(ToolCapability.SEARCH)
        
        if any(kw in tool_name.lower() for kw in ["web", "http", "browser"]):
            metadata.categories.append(ToolCategory.WEB_BROWSING)
            metadata.capabilities.append(ToolCapability.FETCH)
        
        if any(kw in tool_name.lower() for kw in ["file", "read", "write", "save"]):
            metadata.categories.append(ToolCategory.FILE_SYSTEM)
            if "read" in tool_name.lower():
                metadata.capabilities.append(ToolCapability.READ)
            if "write" in tool_name.lower() or "save" in tool_name.lower():
                metadata.capabilities.append(ToolCapability.WRITE)
        
        # Create the tool class
        tool_cls = EnhancedTool.create_tool(
            name=tool_name,
            description=description,
            parameters=parameters.get("properties", {}),
            execute_fn=execute_mcp_tool,
            metadata=metadata,
            source=ToolSource.MCP
        )
        
        # Create an instance
        tool_instance = tool_cls()
        
        # Add server information to the instance
        tool_instance.server_name = server_name
        
        return tool_instance


class MCPDynamicTool(EnhancedTool):
    """
    Dynamic tool for calling any MCP tool on a server.
    
    This tool allows calling any tool on an MCP server by name,
    which is useful when you don't want to create individual tool
    wrappers for all tools on a server.
    """
    
    server_name: str = None
    tool_name: str = None
    arguments: Dict[str, Any] = {}
    
    def __init__(self, server_name: str, **data):
        """Initialize the MCP dynamic tool."""
        super().__init__(**data)
        self.server_name = server_name
        self.clients = {}  # server_name -> client mapping
    
    async def _get_client(self, server_name: str) -> MultiProviderMCPClient:
        """
        Get an MCP client for the specified server.
        
        Args:
            server_name: Name of the MCP server to connect to
            
        Returns:
            MultiProviderMCPClient instance
        """
        if server_name in self.clients:
            return self.clients[server_name]
        
        # This method should be implemented to retrieve
        # or create a client for the specified server
        raise NotImplementedError(
            "MCPDynamicTool._get_client must be implemented to retrieve MCP clients"
        )
    
    async def arun(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """
        Execute any tool on the MCP server.
        
        Args:
            tool_name: Name of the tool to call
            arguments: Arguments for the tool
            
        Returns:
            Result of the tool call
        """
        client = await self._get_client(self.server_name)
        
        try:
            result = await client.call_tool(tool_name, arguments)
            
            # Format the result nicely
            if hasattr(result, "content") and hasattr(result.content, "text"):
                return result.content.text
            elif hasattr(result, "text"):
                return result.text
            else:
                return str(result)
        except Exception as e:
            logger.error(f"Error executing MCP tool {tool_name}: {e}")
            raise 