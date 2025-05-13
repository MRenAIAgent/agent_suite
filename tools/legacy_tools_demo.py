"""
Demo script for using original tools with the enhanced tool system.

This script demonstrates how to:
1. Register original tools with the enhanced tool system
2. Select appropriate tools for a task
3. Use both adapted tools and automatically converted legacy tools
"""
import asyncio
import logging
from typing import List, Dict, Any, Optional

from tools.types import (
    Task, Role, ToolDomain, ToolCapability, ToolCategory
)
from tools.registry import registry
from tools.selection import tool_selector
from tools.adapters.integration import (
    register_all_adapted_tools, auto_convert_legacy_tools
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def demonstrate_mcp_tools():
    """Demonstrate using MCP tools with the enhanced tool system."""
    logger.info("\n=== Demonstrating MCP Tools ===")
    
    # Register MCP tools
    from tools.adapters.mcp_tool_adapter import register_mcp_tools
    register_mcp_tools()
    
    # Create a developer role
    developer_role = Role(
        name="developer",
        description="Software developer who uses MCP tools",
        responsibilities=["Build software", "Integrate with services"],
        required_capabilities=[ToolCapability.COMMUNICATE, ToolCapability.FETCH],
        domains=[ToolDomain.PROGRAMMING]
    )
    
    # Select tools for the developer role
    developer_tools = tool_selector.select_tools_for_role(developer_role)
    
    logger.info(f"Selected tools for developer role: {[t.__class__.__name__ for t in developer_tools]}")
    
    # Create a task for working with MCP servers
    mcp_task = Task(
        description="List available tools on the GitHub MCP server",
        goal="Get a list of GitHub-related tools",
        type="tool discovery",
        domain=ToolDomain.PROGRAMMING,
        required_capabilities=[ToolCapability.FETCH],
        expected_outputs=["List of available tools"]
    )
    
    # Select tools for the MCP task
    mcp_tools = tool_selector.select_tools_for_task(mcp_task)
    
    logger.info(f"Selected tools for MCP task: {[t.__class__.__name__ for t in mcp_tools]}")
    
    # Get the MCP list tools tool
    mcp_list_tools = None
    for tool in mcp_tools:
        if tool.__class__.__name__ == "EnhancedMCPListTools":
            mcp_list_tools = tool
            break
    
    if mcp_list_tools:
        try:
            # This is a mock execution - in a real environment, you would
            # need a working MCP server
            logger.info("Would execute: mcp_list_tools.execute(server_name='github')")
            logger.info("(Not actually executing due to demo mode)")
        except Exception as e:
            logger.error(f"Error demonstrating MCP tools: {e}")
    else:
        logger.warning("MCP list tools tool not found")


async def demonstrate_search_tools():
    """Demonstrate using search tools with the enhanced tool system."""
    logger.info("\n=== Demonstrating Search Tools ===")
    
    # Register SerpAPI tools
    from tools.adapters.serpapi_tool_adapter import register_serpapi_tools
    register_serpapi_tools()
    
    # Create a researcher role
    researcher_role = Role(
        name="researcher",
        description="Researcher who needs to find information",
        responsibilities=["Find information", "Analyze data"],
        required_capabilities=[ToolCapability.SEARCH, ToolCapability.ANALYZE],
        domains=[ToolDomain.GENERAL, ToolDomain.SCIENCE]
    )
    
    # Select tools for the researcher role
    researcher_tools = tool_selector.select_tools_for_role(researcher_role)
    
    logger.info(f"Selected tools for researcher role: {[t.__class__.__name__ for t in researcher_tools]}")
    
    # Create a search task
    search_task = Task(
        description="Search for information about Python programming",
        goal="Find resources for learning Python",
        type="research",
        domain=ToolDomain.GENERAL,
        required_capabilities=[ToolCapability.SEARCH],
        expected_outputs=["Search results"]
    )
    
    # Select tools for the search task
    search_tools = tool_selector.select_tools_for_task(search_task)
    
    logger.info(f"Selected tools for search task: {[t.__class__.__name__ for t in search_tools]}")
    
    # Get the SerpAPI search tool
    serp_search_tool = None
    for tool in search_tools:
        if tool.__class__.__name__ == "EnhancedSerpApiSearchTool":
            serp_search_tool = tool
            break
    
    if serp_search_tool:
        try:
            # This is a mock execution - in a real environment, you would
            # need a valid SERPAPI_API_KEY
            logger.info("Would execute: serp_search_tool.execute(query='Python programming tutorials')")
            logger.info("(Not actually executing due to demo mode)")
        except Exception as e:
            logger.error(f"Error demonstrating search tools: {e}")
    else:
        logger.warning("SerpAPI search tool not found")


async def demonstrate_auto_conversion():
    """Demonstrate auto-converting original tools to enhanced tools."""
    logger.info("\n=== Demonstrating Auto-Conversion of Original Tools ===")
    
    # Auto-convert original tools
    converted_tools = auto_convert_legacy_tools()
    
    logger.info(f"Auto-converted {len(converted_tools)} tools")
    
    if converted_tools:
        for tool in converted_tools:
            logger.info(f"- {tool.__class__.__name__}")
            
        # Create a general task
        general_task = Task(
            description="Perform various operations",
            goal="Complete a mix of tasks",
            type="general",
            domain=ToolDomain.GENERAL,
            required_capabilities=[],
            expected_outputs=["Various results"]
        )
        
        # Select tools for the general task
        general_tools = tool_selector.select_tools_for_task(general_task)
        
        logger.info(f"Selected tools for general task: {[t.__class__.__name__ for t in general_tools]}")
    else:
        logger.warning("No tools were auto-converted")


async def run_demo():
    """Run the legacy tools demo."""
    logger.info("Starting Legacy Tools Demo with Enhanced Tool System")
    
    # Clear the registry to start fresh
    registry.clear()
    
    # Option 1: Register specific legacy tool adapters
    await demonstrate_mcp_tools()
    await demonstrate_search_tools()
    
    # Clear the registry again
    registry.clear()
    
    # Option 2: Register all adapted tools at once
    logger.info("\n=== Registering All Adapted Tools ===")
    tools = register_all_adapted_tools()
    logger.info(f"Registered {len(tools)} adapted tools")
    
    # Option 3: Automatically convert any original tools
    await demonstrate_auto_conversion()
    
    logger.info("\nDemo completed!")


# Run the demo
if __name__ == "__main__":
    asyncio.run(run_demo()) 