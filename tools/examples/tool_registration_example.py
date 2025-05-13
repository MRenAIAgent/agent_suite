#!/usr/bin/env python3
"""
Tool Registration Example

This script demonstrates how to register tools from different sources:
1. Native agent_suite tools
2. LangChain tools
3. MCP tools

Run this script to see how tools from different sources can be integrated
into the unified tool registry system.
"""
import os
import sys
import asyncio
import logging
from typing import List, Dict, Any

# Add the parent directory to the path to make imports work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Import agent_suite components
from tools.registry import registry
from tools.base import EnhancedTool
from tools.tool_types import (
    ToolCategory, ToolCapability, ToolDomain, ToolMetadata, ToolSource
)
from tools.serper_api import SerperSearchTool

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
    registry.register_tool(enhanced_tool)
    registered_tools.append(enhanced_tool)
    logger.info(f"Registered SerperSearchTool as {enhanced_tool.__class__.__name__}")
    
    # Example 2: Create and register a custom tool directly
    class CustomCalculatorTool(EnhancedTool):
        """A simple calculator tool for addition."""
        
        # Tool parameters
        a: int 
        b: int
        
        # Tool metadata
        metadata: ToolMetadata = ToolMetadata(
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
    registry.register_tool(calculator_tool)
    registered_tools.append(calculator_tool)
    logger.info(f"Registered CustomCalculatorTool")
    
    return registered_tools


def register_langchain_tools() -> List[EnhancedTool]:
    """Register tools from LangChain."""
    logger.info("Registering LangChain tools...")
    registered_tools = []
    
    try:
        # Import LangChain components (wrapped in try/except in case LangChain is not installed)
        from langchain.tools import DuckDuckGoSearchRun, WikipediaQueryRun
        from langchain.utilities import WikipediaAPIWrapper
        
        # Import the LangChain adapter
        from tools.adapters.langchain_adapter import LangChainToolAdapter
        
        # Create the adapter
        langchain_adapter = LangChainToolAdapter()
        
        # Example 1: Register DuckDuckGo search
        search_tool = DuckDuckGoSearchRun()
        enhanced_search = langchain_adapter.register_langchain_tool(search_tool)
        registered_tools.append(enhanced_search)
        logger.info(f"Registered LangChain DuckDuckGoSearchRun")
        
        # Example 2: Register Wikipedia query
        wiki_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
        enhanced_wiki = langchain_adapter.register_langchain_tool(wiki_tool)
        registered_tools.append(enhanced_wiki)
        logger.info(f"Registered LangChain WikipediaQueryRun")
        
    except ImportError as e:
        logger.warning(f"LangChain not installed or components missing: {e}")
    
    return registered_tools


async def register_mcp_tools() -> List[EnhancedTool]:
    """Register tools from MCP services."""
    logger.info("Registering MCP tools...")
    registered_tools = []
    
    try:
        # Import MCP adapter
        from tools.adapters.mcp_adapter import MCPToolAdapter
        
        # Create the adapter
        mcp_adapter = MCPToolAdapter()
        
        # Connect to a sample MCP server (replace with your actual server details)
        # This is just a demonstration - modify with your actual server details
        server_url = os.environ.get("MCP_SERVER_URL", "http://localhost:8000")
        server_name = "example_server"
        
        try:
            # Connect to the server
            connected = await mcp_adapter.connect_to_server(
                mcp_url=server_url,
                server_name=server_name
            )
            
            if connected:
                # Scan for available tools
                tools = await mcp_adapter.scan_server_tools(server_name)
                registered_tools.extend(tools)
                logger.info(f"Registered {len(tools)} tools from MCP server {server_name}")
            else:
                logger.warning(f"Failed to connect to MCP server {server_name}")
                
        except Exception as e:
            logger.error(f"Error connecting to MCP server: {e}")
            
            # Fall back to registering the MCP utility tools
            # These tools allow calling MCP tools without pre-registering them
            try:
                from tools.adapters.mcp_tool_adapter import (
                    EnhancedMCPListTools, EnhancedMCPCallTool, register_mcp_tools
                )
                
                # Register the MCP utility tools
                register_mcp_tools()
                logger.info("Registered MCP utility tools")
                
                # Get the registered tools
                registered_tools = registry.get_tools_by_domain(ToolDomain.GENERAL)
                
            except Exception as nested_e:
                logger.error(f"Error registering MCP utility tools: {nested_e}")
        
    except ImportError as e:
        logger.warning(f"MCP components not available: {e}")
    
    return registered_tools


def print_registered_tools():
    """Print details about all registered tools."""
    all_tools = registry.get_all_tools()
    
    print("\n=== Registered Tools ===")
    print(f"Total tools registered: {len(all_tools)}")
    
    for i, tool in enumerate(all_tools, 1):
        print(f"\n{i}. {tool.__class__.__name__}")
        print(f"   Display name: {tool.metadata.display_name if hasattr(tool.metadata, 'display_name') else 'N/A'}")
        print(f"   Description: {tool.__doc__.strip() if tool.__doc__ else 'No description'}")
        print(f"   Categories: {[c.value for c in tool.metadata.categories] if hasattr(tool.metadata, 'categories') else []}")
        print(f"   Source: {tool.source.value if hasattr(tool, 'source') else 'Unknown'}")


async def main():
    """Register tools from all sources and display results."""
    print("Starting Tool Registration Example")
    
    # Register tools from each source
    native_tools = register_native_tools()
    langchain_tools = register_langchain_tools()
    mcp_tools = await register_mcp_tools()
    
    # Print registered tools
    print_registered_tools()


if __name__ == "__main__":
    asyncio.run(main()) 