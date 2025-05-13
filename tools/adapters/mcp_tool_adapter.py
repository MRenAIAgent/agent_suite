"""
Adapter to convert original MCP tools to use the enhanced tool system.

This module provides adapters for the original MCP tools to make them
compatible with the enhanced tool registration system.
"""
import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, Union

from tools.types import (
    ToolCategory, ToolCapability, ToolDomain, ToolMetadata, ToolSource
)
from tools.base import EnhancedTool
from tools.registry import registry
from tools.mcp_tool import MCPToolBase, MCPListTools, MCPCallTool, MCPSequentialThinking

logger = logging.getLogger(__name__)


class EnhancedMCPToolBase(EnhancedTool):
    """Base class for enhanced MCP tools."""
    
    server_name: str
    
    def __init__(self, **data):
        """Initialize the tool with enhanced metadata."""
        # Create default metadata if not provided
        if "metadata" not in data:
            data["metadata"] = ToolMetadata(
                categories=[ToolCategory.UTILITY],
                domains=[ToolDomain.GENERAL],
                capabilities=[ToolCapability.FETCH, ToolCapability.COMMUNICATE],
                semantic_description="Base class for tools that interact with MCP servers.",
                keywords=["mcp", "server", "client"]
            )
        
        super().__init__(**data)
        
        # Initialize the base MCP tool
        self.mcp_tool = MCPToolBase()
        
    async def _get_client(self, server_name: str):
        """Get an MCP client for the specified server."""
        return await self.mcp_tool._get_client(server_name)


class EnhancedMCPListTools(EnhancedMCPToolBase):
    """Enhanced tool for listing available tools from an MCP server."""
    
    server_name: str
    
    def __init__(self, **data):
        """Initialize with enhanced metadata."""
        if "metadata" not in data:
            data["metadata"] = ToolMetadata(
                categories=[ToolCategory.UTILITY],
                domains=[ToolDomain.GENERAL],
                capabilities=[ToolCapability.FETCH, ToolCapability.SEARCH],
                applicable_roles=["developer", "tool_manager", "system_administrator"],
                task_patterns=["list mcp tools", "find available tools", "show tools"],
                examples=[
                    "List tools available on the GitHub MCP server",
                    "Show available tools on the test_server"
                ],
                keywords=["mcp", "list", "tools", "available"]
            )
        
        super().__init__(**data)
        
        # Initialize the original MCP list tools
        self.list_tools = MCPListTools()
    
    async def arun(self, server_name: str) -> Any:
        """
        List available tools from an MCP server.
        
        Args:
            server_name: Name of the MCP server to connect to
            
        Returns:
            List of available tools
        """
        return await self.list_tools.arun(server_name)


class EnhancedMCPCallTool(EnhancedMCPToolBase):
    """Enhanced tool for calling a tool on an MCP server."""
    
    server_name: str
    tool_name: str
    arguments: Dict[str, Any]
    
    def __init__(self, **data):
        """Initialize with enhanced metadata."""
        if "metadata" not in data:
            data["metadata"] = ToolMetadata(
                categories=[ToolCategory.UTILITY],
                domains=[ToolDomain.GENERAL],
                capabilities=[ToolCapability.FETCH, ToolCapability.COMMUNICATE],
                applicable_roles=["developer", "assistant", "automation_engineer"],
                task_patterns=["call mcp tool", "execute mcp action", "run mcp function"],
                examples=[
                    "Call the 'github_create_issue' tool on the github server",
                    "Execute 'search_repositories' with query='python libraries'"
                ],
                keywords=["mcp", "call", "execute", "run", "tool"]
            )
        
        super().__init__(**data)
        
        # Initialize the original MCP call tool
        self.call_tool = MCPCallTool()
    
    async def arun(self, server_name: str, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """
        Call a tool on an MCP server.
        
        Args:
            server_name: Name of the MCP server to connect to
            tool_name: Name of the tool to call
            arguments: Arguments for the tool call
            
        Returns:
            Result of the tool call
        """
        return await self.call_tool.arun(server_name, tool_name, arguments)


class EnhancedMCPSequentialThinking(EnhancedMCPToolBase):
    """Enhanced tool for solving problems using sequential thinking with an MCP server."""
    
    server_name: str
    problem: str
    max_thoughts: int = 5
    
    def __init__(self, **data):
        """Initialize with enhanced metadata."""
        if "metadata" not in data:
            data["metadata"] = ToolMetadata(
                categories=[ToolCategory.UTILITY, ToolCategory.CONTENT_GENERATION],
                domains=[ToolDomain.GENERAL],
                capabilities=[ToolCapability.ANALYZE, ToolCapability.GENERATE],
                applicable_roles=["problem_solver", "researcher", "analyst"],
                task_patterns=["solve problem", "think step by step", "analyze problem"],
                examples=[
                    "Solve the problem 'How to optimize a sorting algorithm?'",
                    "Think through 'What are the implications of quantum computing?'"
                ],
                keywords=["sequential thinking", "step by step", "problem", "analysis"]
            )
        
        super().__init__(**data)
        
        # Initialize the original MCP sequential thinking tool
        self.sequential_thinking = MCPSequentialThinking()
    
    async def arun(self, server_name: str, problem: str, max_thoughts: int = 5) -> Any:
        """
        Solve a problem using sequential thinking with an MCP server.
        
        Args:
            server_name: Name of the MCP server to connect to
            problem: Problem statement to solve using sequential thinking
            max_thoughts: Maximum number of thoughts to generate
            
        Returns:
            Result of the sequential thinking process
        """
        return await self.sequential_thinking.arun(server_name, problem, max_thoughts)


def register_mcp_tools():
    """Register all MCP tools with the registry."""
    # Create and register the enhanced MCP tools
    registry.register_tool(EnhancedMCPListTools())
    registry.register_tool(EnhancedMCPCallTool())
    registry.register_tool(EnhancedMCPSequentialThinking())
    
    logger.info("Registered MCP tools with the registry") 