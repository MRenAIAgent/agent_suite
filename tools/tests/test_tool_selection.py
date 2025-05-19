"""
Test file for the tool selection functionality.

This module contains tests for the tool selection service and the role/task-based
tool selection capabilities.
"""
import asyncio
import unittest
from typing import List, Optional, Union

from pydantic import BaseModel, Field

from tools.tool_types import ToolCapability, ToolCategory, ToolDomain, ToolMetadata
from tools.base import EnhancedTool
from tools.registry import registry
from tools.selection import tool_selector


# Role and Task classes for testing tool selection
class Role(BaseModel):
    """Role definition for role-based tool selection testing."""
    name: str = Field(..., description="Name of the role")
    description: str = Field(..., description="Description of the role")
    responsibilities: List[str] = Field(
        default_factory=list,
        description="Key responsibilities of this role"
    )
    required_capabilities: List[Union[ToolCapability, str]] = Field(
        default_factory=list,
        description="Capabilities required by this role"
    )
    domains: List[Union[ToolDomain, str]] = Field(
        default_factory=list,
        description="Domains this role operates in"
    )
    preferred_tools: Optional[List[str]] = Field(
        default=None,
        description="Specific tools preferred for this role"
    )


class Task(BaseModel):
    """Task definition for task-based tool selection testing."""
    description: str = Field(..., description="Description of the task")
    goal: str = Field(..., description="Goal of the task")
    type: str = Field(..., description="Type of task (e.g., research, analysis)")
    domain: Union[ToolDomain, str] = Field(..., description="Domain of the task")
    required_capabilities: List[Union[ToolCapability, str]] = Field(
        default_factory=list,
        description="Capabilities required for this task"
    )
    expected_outputs: List[str] = Field(
        default_factory=list,
        description="Expected outputs from this task"
    )


# Example tools for testing
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


class ToolSelectionTests(unittest.TestCase):
    """Tests for the tool selection functionality."""
    
    def setUp(self):
        """Set up the test environment."""
        # Clear the registry
        registry.clear()
        
        # Register test tools
        self.weather_tool = WeatherTool(location="New York")
        self.search_tool = SearchTool(query="default search")
        self.code_tool = CodeAnalysisTool(code="def example(): pass")
        
        registry.register_tool(self.weather_tool)
        registry.register_tool(self.search_tool)
        registry.register_tool(self.code_tool)
    
    def tearDown(self):
        """Clean up after the test."""
        # Clear the registry
        registry.clear()
    
    def test_role_based_selection(self):
        """Test role-based tool selection."""
        # Define test roles
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
        
        weather_role = Role(
            name="weather_reporter",
            description="Reports on weather conditions",
            responsibilities=["Get weather data", "Report weather forecasts"],
            required_capabilities=[ToolCapability.FETCH],
            domains=[ToolDomain.GENERAL]
        )
        
        # Test researcher role
        researcher_tools = tool_selector.select_tools_for_role(researcher_role)
        self.assertIn(self.search_tool, researcher_tools)
        
        # Test programmer role
        programmer_tools = tool_selector.select_tools_for_role(programmer_role)
        self.assertIn(self.code_tool, programmer_tools)
        
        # Test weather reporter role
        weather_tools = tool_selector.select_tools_for_role(weather_role)
        self.assertIn(self.weather_tool, weather_tools)
    
    def test_task_based_selection(self):
        """Test task-based tool selection."""
        # Define test tasks
        weather_task = Task(
            description="Find the weather forecast for New York",
            goal="Get accurate weather information",
            type="information retrieval",
            domain=ToolDomain.GENERAL,
            required_capabilities=[ToolCapability.FETCH],
            expected_outputs=["Weather forecast"]
        )
        
        search_task = Task(
            description="Research climate change impacts",
            goal="Gather information on climate change",
            type="research",
            domain=ToolDomain.GENERAL,
            required_capabilities=[ToolCapability.SEARCH],
            expected_outputs=["Research findings"]
        )
        
        code_task = Task(
            description="Analyze Python code for bugs and style issues",
            goal="Improve code quality",
            type="code analysis",
            domain=ToolDomain.PROGRAMMING,
            required_capabilities=[ToolCapability.ANALYZE],
            expected_outputs=["Analysis report"]
        )
        
        # Test weather task
        weather_tools = tool_selector.select_tools_for_task(weather_task)
        self.assertIn(self.weather_tool, weather_tools)
        
        # Test search task
        search_tools = tool_selector.select_tools_for_task(search_task)
        self.assertIn(self.search_tool, search_tools)
        
        # Test code task
        code_tools = tool_selector.select_tools_for_task(code_task)
        self.assertIn(self.code_tool, code_tools)


if __name__ == "__main__":
    unittest.main() 