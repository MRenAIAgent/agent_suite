"""
Demo script for the enhanced tool system.

This script showcases the key features of the enhanced tool system:
1. Tool registration from various sources
2. Enhanced metadata
3. Tool selection based on roles and tasks
"""
import asyncio
import logging
from typing import Dict, List, Optional, Any

from tools.types import (
    ToolCategory, ToolCapability, ToolDomain, ToolMetadata,
    ToolSource, Task, Role
)
from tools.base import EnhancedTool
from tools.registry import registry
from tools.selection import tool_selector

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Example 1: Define a simple EnhancedTool with metadata
class WeatherTool(EnhancedTool):
    """Get weather information for a specific location."""
    
    # Tool parameters
    location: str
    units: Optional[str] = "metric"
    
    def __init__(self, **data):
        # Initialize with enhanced metadata
        if "metadata" not in data:
            data["metadata"] = ToolMetadata(
                categories=[ToolCategory.UTILITY],
                domains=[ToolDomain.GENERAL],
                capabilities=[ToolCapability.FETCH],
                applicable_roles=["weather_reporter", "travel_planner"],
                task_patterns=["weather", "forecast", "temperature", "climate"],
                examples=[
                    "Get weather for New York",
                    "Check temperature in London with units=imperial"
                ],
                keywords=["weather", "forecast", "temperature"]
            )
        super().__init__(**data)
    
    async def arun(self, location: str, units: str = "metric") -> str:
        """Get weather information for the specified location."""
        # This is a mock implementation
        if units == "metric":
            return f"Weather for {location}: 25°C, Sunny"
        else:
            return f"Weather for {location}: 77°F, Sunny"


# Example 2: Tool with more complex metadata
class SearchTool(EnhancedTool):
    """Search the web for information."""
    
    # Tool parameters
    query: str
    max_results: Optional[int] = 5
    
    def __init__(self, **data):
        # Initialize with enhanced metadata
        if "metadata" not in data:
            data["metadata"] = ToolMetadata(
                categories=[ToolCategory.SEARCH, ToolCategory.WEB_BROWSING],
                domains=[ToolDomain.GENERAL],
                capabilities=[ToolCapability.SEARCH, ToolCapability.FETCH],
                applicable_roles=["researcher", "analyst", "assistant"],
                task_patterns=["search", "find information", "look up", "research"],
                examples=[
                    "Search for climate change news",
                    "Find information about python programming with max_results=10"
                ],
                is_cacheable=True,
                avg_execution_time_ms=1500,
                keywords=["search", "web", "information", "research"]
            )
        super().__init__(**data)
    
    async def arun(self, query: str, max_results: int = 5) -> str:
        """Search the web for the specified query."""
        # This is a mock implementation
        return f"Search results for '{query}' (max {max_results}):\n" + \
               "1. Example result 1\n" + \
               "2. Example result 2\n" + \
               "3. Example result 3"


# Example 3: Tool for a specific domain
class CodeAnalysisTool(EnhancedTool):
    """Analyze code for patterns and issues."""
    
    # Tool parameters
    code: str
    language: Optional[str] = "python"
    
    def __init__(self, **data):
        # Initialize with enhanced metadata
        if "metadata" not in data:
            data["metadata"] = ToolMetadata(
                categories=[ToolCategory.CODING],
                domains=[ToolDomain.PROGRAMMING],
                capabilities=[ToolCapability.ANALYZE],
                applicable_roles=["programmer", "code_reviewer"],
                task_patterns=["analyze code", "review code", "find bugs"],
                examples=[
                    "Analyze 'def hello(): print(\"Hello\")' with language=python",
                    "Check JavaScript code for issues"
                ],
                prerequisites=["Code must be syntactically valid"],
                keywords=["code", "analysis", "bugs", "programming"]
            )
        super().__init__(**data)
    
    async def arun(self, code: str, language: str = "python") -> str:
        """Analyze the provided code."""
        # This is a mock implementation
        return f"Analysis of {language} code:\n" + \
               "- No major issues found\n" + \
               "- Consider adding docstrings\n" + \
               "- Code follows standard conventions"


async def run_demo():
    """Run the enhanced tool system demo."""
    logger.info("Starting Enhanced Tool System Demo")
    
    # Step 1: Register the example tools
    logger.info("\n=== Step 1: Registering Tools ===")
    weather_tool = WeatherTool()
    search_tool = SearchTool()
    code_tool = CodeAnalysisTool()
    
    registry.register_tool(weather_tool)
    registry.register_tool(search_tool)
    registry.register_tool(code_tool)
    
    logger.info(f"Registered {len(registry.get_all_tools())} tools")
    
    # Step 2: Display tool information
    logger.info("\n=== Step 2: Tool Information ===")
    for tool in registry.get_all_tools():
        logger.info(f"\nTool: {tool.__class__.__name__}")
        logger.info(f"Description: {tool.__doc__}")
        logger.info(f"Categories: {[str(c) for c in tool.metadata.categories]}")
        logger.info(f"Capabilities: {[str(c) for c in tool.metadata.capabilities]}")
        logger.info(f"Domains: {[str(d) for d in tool.metadata.domains]}")
    
    # Step 3: Tool selection by role
    logger.info("\n=== Step 3: Tool Selection by Role ===")
    researcher_role = Role(
        name="researcher",
        description="Conducts research on various topics",
        responsibilities=["Find information", "Analyze data", "Report findings"],
        required_capabilities=[ToolCapability.SEARCH, ToolCapability.ANALYZE],
        domains=[ToolDomain.GENERAL]
    )
    
    programmer_role = Role(
        name="programmer",
        description="Writes and reviews code",
        responsibilities=["Write code", "Debug issues", "Review code"],
        required_capabilities=[ToolCapability.ANALYZE],
        domains=[ToolDomain.PROGRAMMING]
    )
    
    # Select tools for each role
    researcher_tools = tool_selector.select_tools_for_role(researcher_role)
    programmer_tools = tool_selector.select_tools_for_role(programmer_role)
    
    logger.info(f"\nResearcher tools: {[t.__class__.__name__ for t in researcher_tools]}")
    logger.info(f"Programmer tools: {[t.__class__.__name__ for t in programmer_tools]}")
    
    # Step 4: Tool selection by task
    logger.info("\n=== Step 4: Tool Selection by Task ===")
    weather_task = Task(
        description="Find the weather forecast for New York",
        goal="Get accurate weather information",
        type="information retrieval",
        domain=ToolDomain.GENERAL,
        required_capabilities=[ToolCapability.FETCH],
        expected_outputs=["Weather forecast"]
    )
    
    code_task = Task(
        description="Analyze Python code for bugs and style issues",
        goal="Improve code quality",
        type="code analysis",
        domain=ToolDomain.PROGRAMMING,
        required_capabilities=[ToolCapability.ANALYZE],
        expected_outputs=["Analysis report"]
    )
    
    # Select tools for each task
    weather_tools = tool_selector.select_tools_for_task(weather_task)
    code_tools = tool_selector.select_tools_for_task(code_task)
    
    logger.info(f"\nWeather task tools: {[t.__class__.__name__ for t in weather_tools]}")
    logger.info(f"Code task tools: {[t.__class__.__name__ for t in code_tools]}")
    
    # Step 5: Tool execution with usage statistics
    logger.info("\n=== Step 5: Tool Execution with Usage Stats ===")
    
    # Execute the weather tool
    result = await weather_tool.execute(location="London", units="imperial")
    logger.info(f"\nWeather tool result: {result}")
    
    # Execute the search tool
    result = await search_tool.execute(query="enhanced tool system python", max_results=3)
    logger.info(f"\nSearch tool result: {result}")
    
    # Show usage statistics
    logger.info("\nUsage statistics:")
    logger.info(f"Weather tool: {weather_tool.get_usage_stats().model_dump()}")
    logger.info(f"Search tool: {search_tool.get_usage_stats().model_dump()}")
    
    logger.info("\nDemo completed!")


# Run the demo
if __name__ == "__main__":
    asyncio.run(run_demo()) 