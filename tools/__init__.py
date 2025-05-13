"""
Tools module for agent_suite.

This module provides a registration-based tool system with enhanced metadata
and tool selection capabilities.
"""
# Expose original Tool class for backward compatibility
from tools.tool import Tool

# Expose core types
from tools.tool_types import (
    ToolCategory, ToolCapability, ToolDomain, ToolMetadata,
    ToolSource, Task, Role, ToolUsageStats
)

# Expose enhanced tool classes
from tools.base import EnhancedTool

# Expose registry
from tools.registry import registry, ToolRegistry

# Expose tool selection service
from tools.selection import tool_selector, ToolSelectionService

# Import tool adapters
try:
    from tools.adapters.langchain_adapter import (
        LangChainToolAdapter, create_langchain_adapter
    )
except ImportError:
    pass

try:
    from tools.adapters.mcp_adapter import (
        MCPToolAdapter, MCPDynamicTool
    )
except ImportError:
    pass

# Module version
__version__ = "0.1.0"


# Convenience functions
def register_tool(tool):
    """Register a tool with the global registry."""
    return registry.register_tool(tool)


def get_tool(name):
    """Get a tool by name from the global registry."""
    return registry.get_tool(name)


def get_all_tools():
    """Get all tools from the global registry."""
    return registry.get_all_tools()


def select_tools_for_task(task_description):
    """Select tools for a task using the global tool selector."""
    return tool_selector.select_tools_for_task(task_description)


def add_tools_to_agent(agent, tools=None, task_description=None, max_tools=None):
    """
    Add tools to an agent.
    
    Args:
        agent: Agent instance to add tools to
        tools: List of specific tools to add or None to select automatically
        task_description: Description of the task for automatic tool selection
        max_tools: Maximum number of tools to add
    """
    if tools is None and task_description:
        # Create a task from the description
        task = Task(
            description=task_description,
            goal="Complete the task successfully",
            type="general",
            domain=ToolDomain.GENERAL,
            required_capabilities=[],
            expected_outputs=[]
        )
        
        # Select tools for the task
        tools = tool_selector.select_tools_for_task(task, max_tools)
    
    # Add the tools to the agent
    if tools:
        for tool in tools:
            agent.add_tool(tool)


def init_tool_system(scan_packages=None):
    """
    Initialize the tool system.
    
    Args:
        scan_packages: List of package names to scan for tools
    """
    if scan_packages:
        for package_name in scan_packages:
            registry.scan_package_for_tools(package_name) 