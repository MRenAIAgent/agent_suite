#!/usr/bin/env python3
"""
Simplified Tool Registration Example

This script demonstrates how to register tools from different sources:
1. Native agent_suite tools
2. Mock LangChain tools (no actual dependency on LangChain)
3. Mock MCP tools (no actual dependency on MCP)

This version uses mock implementations so it can run without external dependencies.
"""
import os
import sys
import asyncio
import logging
from typing import List, Dict, Any, Optional, ClassVar, Type

# Add the parent directory to the path to make imports work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Import agent_suite components
from tools.registry import registry
from tools.base import EnhancedTool
from tools.tool_types import (
    ToolCategory, ToolCapability, ToolDomain, ToolMetadata, ToolSource
)
from tools.serper_api import SerperSearchTool
from tools.tool import Tool

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def register_native_tools() -> List[EnhancedTool]:
    """Register native agent_suite tools."""
    logger.info("Registering native agent_suite tools...")
    registered_tools = []
    
    # Clear registry to start fresh
    registry.clear()
    
    # Example 1: Register SerperSearchTool
    serper_tool = SerperSearchTool(query="example query")
    enhanced_tool = serper_tool.to_enhanced_tool()
    registry.register_tool(enhanced_tool, override=True)
    registered_tools.append(enhanced_tool)
    logger.info(f"Registered SerperSearchTool as {enhanced_tool.__class__.__name__}")
    
    # Example 2: Create and register a custom tool directly
    class CustomCalculatorTool(EnhancedTool):
        """A simple calculator tool for addition."""
        
        # Tool parameters
        a: int 
        b: int
        
        # Tool metadata
        metadata: ClassVar[ToolMetadata] = ToolMetadata(
            name="calculator_add",
            display_name="Calculator Add",
            description="Add two numbers together",
            categories=[ToolCategory.UTILITY],
            domains=[ToolDomain.GENERAL],
            capabilities=[ToolCapability.ANALYZE],
            examples=["Calculate 5 + 3"]
        )
        
        async def arun(self, **kwargs) -> str:
            """Add two numbers."""
            a = kwargs.get("a", self.a)
            b = kwargs.get("b", self.b)
            return f"{a} + {b} = {a + b}"
    
    calculator_tool = CustomCalculatorTool(a=0, b=0)
    registry.register_tool(calculator_tool, override=True)
    registered_tools.append(calculator_tool)
    logger.info(f"Registered CustomCalculatorTool")
    
    return registered_tools


def register_mock_langchain_tools() -> List[EnhancedTool]:
    """Register mock tools that simulate LangChain tools."""
    logger.info("Registering mock LangChain tools...")
    registered_tools = []
    
    # Create a mock LangChain tool class
    class MockLangChainTool:
        """Mock LangChain tool for demonstration purposes."""
        
        def __init__(self, name, description):
            self.name = name
            self.description = description
            
        def run(self, input_str):
            return f"Mock result for {self.name} with input: {input_str}"
            
        async def arun(self, input_str):
            return self.run(input_str)
    
    # Create enhanced tool wrappers for LangChain tools
    class DuckDuckGoSearchWrapper(EnhancedTool):
        """Enhanced tool wrapper for DuckDuckGo Search."""
        
        # The actual LangChain tool instance
        _lc_tool: Any
        
        # Tool parameter
        query: str = ""
        
        # Metadata using ClassVar to avoid Pydantic issues
        metadata: ClassVar[ToolMetadata] = ToolMetadata(
            name="duckduckgo_search",
            display_name="DuckDuckGo Search",
            description="Search the web using DuckDuckGo",
            categories=[ToolCategory.SEARCH],
            domains=[ToolDomain.GENERAL],
            capabilities=[ToolCapability.SEARCH],
            source=ToolSource.CUSTOM
        )
        
        async def arun(self, **kwargs) -> str:
            """Execute the LangChain tool."""
            query = kwargs.get("query", self.query)
            # Call the underlying LangChain tool
            result = await self._lc_tool.arun(query)
            return result
    
    class WikipediaQueryWrapper(EnhancedTool):
        """Enhanced tool wrapper for Wikipedia Query."""
        
        # The actual LangChain tool instance
        _lc_tool: Any
        
        # Tool parameter
        query: str = ""
        
        # Metadata using ClassVar to avoid Pydantic issues
        metadata: ClassVar[ToolMetadata] = ToolMetadata(
            name="wikipedia_query",
            display_name="Wikipedia Query",
            description="Query Wikipedia for information",
            categories=[ToolCategory.SEARCH],
            domains=[ToolDomain.GENERAL],
            capabilities=[ToolCapability.SEARCH, ToolCapability.READ],
            source=ToolSource.CUSTOM
        )
        
        async def arun(self, **kwargs) -> str:
            """Execute the LangChain tool."""
            query = kwargs.get("query", self.query)
            # Call the underlying LangChain tool
            result = await self._lc_tool.arun(query)
            return result
    
    # Create a LangChain adapter class
    class MockLangChainAdapter:
        """Mock adapter for LangChain tools."""
        
        def __init__(self, registry_instance=None):
            self.registry = registry_instance or registry
        
        def register_langchain_tool(self, lc_tool, namespace=None) -> EnhancedTool:
            """Register a mock LangChain tool."""
            # Create appropriate wrapper based on the tool name
            if lc_tool.name == "DuckDuckGoSearch":
                wrapper = DuckDuckGoSearchWrapper(_lc_tool=lc_tool)
            elif lc_tool.name == "WikipediaQuery":
                wrapper = WikipediaQueryWrapper(_lc_tool=lc_tool)
            else:
                raise ValueError(f"Unknown LangChain tool: {lc_tool.name}")
            
            # Register with the registry - use override to prevent duplicates
            if namespace:
                registry.namespaces[namespace] = registry.namespaces.get(namespace, set())
                if f"{namespace}.{wrapper.__class__.__name__}" in registry.tools:
                    # Remove existing tool with same name to avoid duplicates
                    registry.namespaces[namespace].discard(f"{namespace}.{wrapper.__class__.__name__}")
                
                # Register with the namespace
                self.registry.register_tools_with_namespace(namespace, [wrapper])
            else:
                self.registry.register_tool(wrapper, override=True)
                
            return wrapper
    
    # Create mock LangChain tool instances
    mock_search = MockLangChainTool(
        name="DuckDuckGoSearch",
        description="Search the web using DuckDuckGo"
    )
    
    mock_wiki = MockLangChainTool(
        name="WikipediaQuery",
        description="Query Wikipedia for information"
    )
    
    # Create the adapter
    adapter = MockLangChainAdapter()
    
    # Register the tools
    enhanced_search = adapter.register_langchain_tool(mock_search, namespace="langchain")
    enhanced_wiki = adapter.register_langchain_tool(mock_wiki, namespace="langchain")
    
    registered_tools.extend([enhanced_search, enhanced_wiki])
    logger.info(f"Registered 2 mock LangChain tools")
    
    return registered_tools


def register_mock_mcp_tools() -> List[EnhancedTool]:
    """Register mock MCP tools."""
    logger.info("Registering mock MCP tools...")
    registered_tools = []
    
    # Create a mock MCP tool class
    class MockMCPTool(Tool):
        """Base class for mock MCP tools."""
        
        server_name: str
        
        def run(self, **kwargs):
            return f"Mock MCP result for server {self.server_name}"
        
        async def arun(self, **kwargs):
            return self.run(**kwargs)
    
    # Create specific MCP tool types
    class MockMCPListTools(MockMCPTool):
        """Mock tool for listing MCP tools."""
        
        server_name: str
        
        async def arun(self, **kwargs):
            """List tools on the MCP server."""
            return (
                "Available tools on server:\n"
                "1. github_create_issue\n"
                "2. github_list_issues\n"
                "3. web_search\n"
                "4. file_upload"
            )
    
    class MockMCPCallTool(MockMCPTool):
        """Mock tool for calling MCP tools."""
        
        server_name: str
        tool_name: str
        arguments: Dict[str, Any] = {}
        
        async def arun(self, **kwargs):
            """Call a tool on the MCP server."""
            server = kwargs.get("server_name", self.server_name)
            tool = kwargs.get("tool_name", self.tool_name)
            args = kwargs.get("arguments", self.arguments)
            
            return f"Executed {tool} on {server} with arguments: {args}"
    
    # Create MCP enhanced tool wrappers
    class MCPListToolsWrapper(EnhancedTool):
        """Enhanced wrapper for MCP List Tools."""
        
        # Original tool being wrapped
        original_tool: MockMCPListTools
        
        # Tool metadata
        metadata: ClassVar[ToolMetadata] = ToolMetadata(
            name="mcp_list_tools",
            display_name="MCP List Tools",
            description="List available tools on an MCP server",
            categories=[ToolCategory.UTILITY],
            domains=[ToolDomain.GENERAL],
            capabilities=[ToolCapability.FETCH],
            source=ToolSource.CUSTOM
        )
        
        async def arun(self, **kwargs) -> str:
            """Execute the wrapped tool."""
            return await self.original_tool.arun(**kwargs)
    
    class MCPCallToolWrapper(EnhancedTool):
        """Enhanced wrapper for MCP Call Tool."""
        
        # Original tool being wrapped
        original_tool: MockMCPCallTool
        
        # Tool metadata
        metadata: ClassVar[ToolMetadata] = ToolMetadata(
            name="mcp_call_tool",
            display_name="MCP Call Tool",
            description="Execute a specific tool on an MCP server",
            categories=[ToolCategory.UTILITY],
            domains=[ToolDomain.GENERAL],
            capabilities=[ToolCapability.FETCH, ToolCapability.WRITE],
            source=ToolSource.CUSTOM
        )
        
        async def arun(self, **kwargs) -> str:
            """Execute the wrapped tool."""
            return await self.original_tool.arun(**kwargs)
    
    # Setup the MCP namespace
    namespace = "mcp"
    registry.namespaces[namespace] = registry.namespaces.get(namespace, set())
    
    # Create and register mock MCP tools - avoid duplicates by checking registry first
    # List tools
    list_tool = MockMCPListTools(server_name="mock_server")
    list_wrapper = MCPListToolsWrapper(original_tool=list_tool)
    
    if f"{namespace}.{list_wrapper.__class__.__name__}" in registry.tools:
        registry.namespaces[namespace].discard(f"{namespace}.{list_wrapper.__class__.__name__}")
    
    registry.register_tools_with_namespace(namespace, [list_wrapper])
    registered_tools.append(list_wrapper)
    
    # Call tool
    call_tool = MockMCPCallTool(
        server_name="mock_server",
        tool_name="github_create_issue",
        arguments={"title": "Example issue", "body": "This is a test"}
    )
    call_wrapper = MCPCallToolWrapper(original_tool=call_tool)
    
    if f"{namespace}.{call_wrapper.__class__.__name__}" in registry.tools:
        registry.namespaces[namespace].discard(f"{namespace}.{call_wrapper.__class__.__name__}")
    
    registry.register_tools_with_namespace(namespace, [call_wrapper])
    registered_tools.append(call_wrapper)
    
    logger.info(f"Registered 2 mock MCP tools")
    
    return registered_tools


def print_registered_tools():
    """Print details about all registered tools."""
    all_tools = registry.get_all_tools()
    
    print("\n=== Registered Tools ===")
    print(f"Total tools registered: {len(all_tools)}")
    
    # Sort tools by namespace
    namespace_tools = {}
    for tool in all_tools:
        # Find the namespace from the registry
        namespace = "default"
        for ns, tools in registry.namespaces.items():
            for tool_name in tools:
                if registry.tools.get(tool_name) == tool:
                    namespace = ns
                    break
        
        if namespace not in namespace_tools:
            namespace_tools[namespace] = []
        namespace_tools[namespace].append(tool)
    
    # Print tools by namespace
    for namespace, tools in sorted(namespace_tools.items()):
        print(f"\n--- Namespace: {namespace} ({len(tools)} tools) ---")
        for i, tool in enumerate(tools, 1):
            print(f"{i}. {tool.__class__.__name__}")
            print(f"   Display name: {tool.metadata.display_name if hasattr(tool.metadata, 'display_name') else 'N/A'}")
            print(f"   Description: {tool.__doc__.strip() if tool.__doc__ else 'No description'}")
            print(f"   Categories: {[c.value for c in tool.metadata.categories] if hasattr(tool.metadata, 'categories') else []}")
            print(f"   Source: {tool.source.value if hasattr(tool, 'source') else 'Unknown'}")
            print(f"   Capabilities: {[c.value for c in tool.metadata.capabilities] if hasattr(tool.metadata, 'capabilities') else []}")


def main():
    """Register tools from all sources and display results."""
    print("Starting Simplified Tool Registration Example")
    
    # Register tools from each source
    native_tools = register_native_tools()
    langchain_tools = register_mock_langchain_tools()
    mcp_tools = register_mock_mcp_tools()
    
    # Print registered tools
    print_registered_tools()
    
    # Print stats by category
    categories = {c.value: 0 for c in ToolCategory}
    for tool in registry.get_all_tools():
        for category in tool.metadata.categories:
            categories[category.value] += 1
    
    print("\n=== Tools by Category ===")
    for category, count in sorted(categories.items()):
        if count > 0:
            print(f"{category}: {count} tools")
    
    # Print stats by capability
    capabilities = {}
    for tool in registry.get_all_tools():
        for capability in tool.metadata.capabilities:
            cap_value = capability.value
            capabilities[cap_value] = capabilities.get(cap_value, 0) + 1
    
    print("\n=== Tools by Capability ===")
    for capability, count in sorted(capabilities.items()):
        print(f"{capability}: {count} tools")


if __name__ == "__main__":
    main() 