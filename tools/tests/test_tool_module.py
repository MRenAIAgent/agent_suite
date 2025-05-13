"""
Unit tests for the enhanced tool module.

This test suite comprehensively tests all major components of the enhanced tool system:
1. Tool types and enums
2. Base EnhancedTool class
3. Tool registry functionality
4. Legacy tool conversion
5. Tool selection service
"""
import os
import sys
import unittest
import asyncio
from unittest.mock import MagicMock, patch

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Import tool system components
from tools.tool_types import (
    ToolCategory, ToolCapability, ToolDomain, ToolMetadata, 
    ToolSource, ToolUsageStats, Task, Role
)
from tools.base import EnhancedTool
from tools.registry import registry, ToolRegistry
from tools.selection import ToolSelectionService
from tools.tool import Tool


class TestToolTypes(unittest.TestCase):
    """Test the tool type definitions."""
    
    def test_enums_exist(self):
        """Test that all enum types are properly defined."""
        # Test ToolCategory
        self.assertIsNotNone(ToolCategory.SEARCH)
        self.assertIsNotNone(ToolCategory.CODING)
        self.assertIsNotNone(ToolCategory.UTILITY)
        
        # Test ToolDomain
        self.assertIsNotNone(ToolDomain.GENERAL)
        self.assertIsNotNone(ToolDomain.PROGRAMMING)
        
        # Test ToolCapability
        self.assertIsNotNone(ToolCapability.SEARCH)
        self.assertIsNotNone(ToolCapability.READ)
        self.assertIsNotNone(ToolCapability.WRITE)
        
        # Test ToolSource
        self.assertIsNotNone(ToolSource.CUSTOM)
        self.assertIsNotNone(ToolSource.SYSTEM)
    
    def test_metadata_creation(self):
        """Test creating tool metadata."""
        metadata = ToolMetadata(
            name="test_tool",
            display_name="Test Tool",
            description="A test tool",
            categories=[ToolCategory.UTILITY],
            domains=[ToolDomain.GENERAL],
            capabilities=[ToolCapability.READ]
        )
        
        self.assertEqual(metadata.name, "test_tool")
        self.assertEqual(metadata.display_name, "Test Tool")
        self.assertEqual(metadata.description, "A test tool")
        self.assertEqual(metadata.categories, [ToolCategory.UTILITY])
        self.assertEqual(metadata.domains, [ToolDomain.GENERAL])
        self.assertEqual(metadata.capabilities, [ToolCapability.READ])
    
    def test_usage_stats(self):
        """Test tool usage statistics tracking."""
        stats = ToolUsageStats()
        
        # Initial state
        self.assertEqual(stats.success_count, 0)
        self.assertEqual(stats.failure_count, 0)
        
        # Update with success
        stats.update(True, 100.0)
        self.assertEqual(stats.success_count, 1)
        self.assertEqual(stats.failure_count, 0)
        self.assertEqual(stats.avg_execution_time, 100.0)
        
        # Update with failure
        stats.update(False, 200.0)
        self.assertEqual(stats.success_count, 1)
        self.assertEqual(stats.failure_count, 1)
        self.assertEqual(stats.avg_execution_time, 150.0)  # Average of 100 and 200


class TestEnhancedTool(unittest.IsolatedAsyncioTestCase):
    """Test the EnhancedTool base class."""
    
    def test_enhanced_tool_creation(self):
        """Test creating a basic enhanced tool."""
        # Create a simple enhanced tool for testing
        class TestTool(EnhancedTool):
            """A test tool for unit testing."""
            
            metadata: ToolMetadata = ToolMetadata(
                name="test_tool",
                display_name="Test Tool"
            )
            
            async def arun(self, **kwargs):
                return "Test result"
            
            def run(self, **kwargs):
                return "Test result"
        
        # Create an instance
        tool = TestTool()
        
        # Check basic properties
        self.assertEqual(tool.__class__.__name__, "TestTool")
        self.assertIsNotNone(tool.metadata)
        self.assertEqual(tool.source, ToolSource.CUSTOM)
    
    def test_function_call_conversion(self):
        """Test converting an enhanced tool to a function call format."""
        # Create a simple enhanced tool with parameters
        class ParamTool(EnhancedTool):
            """A test tool with parameters."""
            
            query: str
            max_results: int = 5
            metadata: ToolMetadata = ToolMetadata(
                name="param_tool",
                display_name="Parameter Tool"
            )
            
            async def arun(self, **kwargs):
                return f"Searching for {self.query}, max {self.max_results} results"
        
        # Create an instance
        tool = ParamTool(query="test")
        
        # Convert to function call
        func_def = tool.convert_to_function_call()
        
        # Check function call properties
        self.assertEqual(func_def["type"], "function")
        self.assertEqual(func_def["function"]["name"], "paramtool")
        self.assertIn("parameters", func_def["function"])
        self.assertIn("properties", func_def["function"]["parameters"])
        self.assertIn("query", func_def["function"]["parameters"]["properties"])
        self.assertIn("max_results", func_def["function"]["parameters"]["properties"])
    
    @patch('time.time')
    async def test_tool_execution_stats(self, mock_time):
        """Test that tool execution updates usage statistics."""
        # Mock time.time to return predictable values
        mock_time.side_effect = [100.0, 101.0]  # Start and end times
        
        # Create a simple tool
        class StatsTool(EnhancedTool):
            """A test tool for testing stats tracking."""
            
            metadata: ToolMetadata = ToolMetadata(
                name="stats_tool",
                display_name="Stats Tool"
            )
            
            async def arun(self, **kwargs):
                return "Test result"
        
        # Create an instance
        tool = StatsTool()
        
        # Execute the tool
        result = await tool.execute()
        
        # Check the result
        self.assertEqual(result, "Test result")
        
        # Check that stats were updated
        stats = tool.get_usage_stats()
        self.assertEqual(stats.success_count, 1)
        self.assertEqual(stats.failure_count, 0)
        self.assertEqual(stats.avg_execution_time, 1000.0)  # 1 second = 1000ms
    
    @patch('time.time')
    async def test_tool_execution_failure(self, mock_time):
        """Test that tool execution handles failures correctly."""
        # Mock time.time to return predictable values
        mock_time.side_effect = [100.0, 101.0]  # Start and end times
        
        # Create a tool that raises an exception
        class FailingTool(EnhancedTool):
            """A test tool that always fails."""
            
            metadata: ToolMetadata = ToolMetadata(
                name="failing_tool",
                display_name="Failing Tool"
            )
            
            async def arun(self, **kwargs):
                raise ValueError("Test error")
        
        # Create an instance
        tool = FailingTool()
        
        # Execute the tool and expect an exception
        with self.assertRaises(ValueError):
            await tool.execute()
        
        # Check that stats were updated
        stats = tool.get_usage_stats()
        self.assertEqual(stats.success_count, 0)
        self.assertEqual(stats.failure_count, 1)
        self.assertEqual(stats.avg_execution_time, 1000.0)  # 1 second = 1000ms


class TestToolRegistry(unittest.TestCase):
    """Test the tool registry functionality."""
    
    def setUp(self):
        """Set up for registry tests."""
        # Clear the registry before each test
        registry.clear()
        
        # Create some test tools
        class SearchTool(EnhancedTool):
            """A test search tool."""
            
            metadata: ToolMetadata = ToolMetadata(
                name="search_tool",
                categories=[ToolCategory.SEARCH],
                domains=[ToolDomain.GENERAL]
            )
            
            async def arun(self, **kwargs):
                return "Search results"
        
        class CodeTool(EnhancedTool):
            """A test coding tool."""
            
            metadata: ToolMetadata = ToolMetadata(
                name="code_tool",
                categories=[ToolCategory.CODING],
                domains=[ToolDomain.PROGRAMMING]
            )
            
            async def arun(self, **kwargs):
                return "Code snippet"
        
        # Save the classes for use in tests
        self.SearchTool = SearchTool
        self.CodeTool = CodeTool
    
    def test_register_tool(self):
        """Test registering a tool with the registry."""
        # Register a tool
        registry.clear()  # Make sure registry is empty
        tool = self.SearchTool()
        registry.register_tool(tool, override=True)  # Use override to ensure only one tool
        
        # Check that it was registered
        registered_tools = registry.get_all_tools()
        self.assertEqual(len(registered_tools), 1)
        self.assertEqual(registered_tools[0].__class__.__name__, "SearchTool")
    
    def test_get_tools_by_category(self):
        """Test retrieving tools by category."""
        # Register tools with override to avoid duplicates
        registry.clear()  # Make sure registry is empty
        registry.register_tool(self.SearchTool(), override=True)
        registry.register_tool(self.CodeTool(), override=True)
        
        # Get tools by category
        search_tools = registry.get_tools_by_category(ToolCategory.SEARCH)
        coding_tools = registry.get_tools_by_category(ToolCategory.CODING)
        
        # Check the results - should have exactly one tool per category
        self.assertEqual(len(search_tools), 1)
        self.assertEqual(len(coding_tools), 1)
        self.assertEqual(search_tools[0].__class__.__name__, "SearchTool")
        self.assertEqual(coding_tools[0].__class__.__name__, "CodeTool")
    
    def test_get_tools_by_domain(self):
        """Test retrieving tools by domain."""
        # Register tools with override to avoid duplicates
        registry.clear()  # Make sure registry is empty
        registry.register_tool(self.SearchTool(), override=True)
        registry.register_tool(self.CodeTool(), override=True)
        
        # Get tools by domain
        general_tools = registry.get_tools_by_domain(ToolDomain.GENERAL)
        programming_tools = registry.get_tools_by_domain(ToolDomain.PROGRAMMING)
        
        # Check the results - should have exactly one tool per domain
        self.assertEqual(len(general_tools), 1)
        self.assertEqual(len(programming_tools), 1)
        self.assertEqual(general_tools[0].__class__.__name__, "SearchTool")
        self.assertEqual(programming_tools[0].__class__.__name__, "CodeTool")
    
    def test_get_tool_by_name(self):
        """Test retrieving a tool by name."""
        # Register a tool
        registry.clear()  # Make sure registry is empty
        tool = self.SearchTool()
        registry.register_tool(tool, override=True)
        
        # Get the tool by name
        tool = registry.get_tool("SearchTool")
        
        # Check the result
        self.assertIsNotNone(tool)
        self.assertEqual(tool.__class__.__name__, "SearchTool")


class TestLegacyToolConversion(unittest.IsolatedAsyncioTestCase):
    """Test converting legacy tools to enhanced tools."""
    
    def setUp(self):
        """Set up for legacy tool tests."""
        # Clear the registry before each test
        registry.clear()
        
        # Create a simple legacy tool
        class LegacySearchTool(Tool):
            """A legacy search tool."""
            
            query: str
            
            async def arun(self, **kwargs):
                return f"Searching for {self.query}"
            
            def run(self, **kwargs):
                return f"Searching for {self.query}"
        
        # Save the class for use in tests
        self.LegacySearchTool = LegacySearchTool
    
    def test_legacy_tool_conversion(self):
        """Test converting a legacy tool to an enhanced tool."""
        # Create a legacy tool instance
        legacy_tool = self.LegacySearchTool(query="test")
        
        # Convert to enhanced tool
        enhanced_tool = legacy_tool.to_enhanced_tool()
        
        # Check the conversion
        self.assertIsInstance(enhanced_tool, EnhancedTool)
        self.assertEqual(enhanced_tool.original_tool, legacy_tool)
        
        # Check metadata
        self.assertEqual(enhanced_tool.metadata.display_name, "LegacySearchTool")
        self.assertIn(ToolCategory.UTILITY, enhanced_tool.metadata.categories)
    
    async def test_enhanced_wrapper_execution(self):
        """Test that the enhanced wrapper correctly delegates to the original tool."""
        # Create and convert a legacy tool
        legacy_tool = self.LegacySearchTool(query="test")
        enhanced_tool = legacy_tool.to_enhanced_tool()
        
        # Run the enhanced tool
        result = await enhanced_tool.arun()
        
        # Check that it delegated to the original tool
        self.assertEqual(result, "Searching for test")
        
        # Test with different parameters
        result = await enhanced_tool.arun(query="different")
        self.assertEqual(result, "Searching for different")


class TestToolSelectionService(unittest.TestCase):
    """Test the tool selection service."""
    
    def setUp(self):
        """Set up for tool selection tests."""
        # Clear the registry before each test
        registry.clear()
        
        # Create the tool selection service
        self.selector = ToolSelectionService()
        
        # Create some test tools with rich metadata
        class SearchTool(EnhancedTool):
            """A test search tool."""
            
            metadata: ToolMetadata = ToolMetadata(
                name="search_tool",
                categories=[ToolCategory.SEARCH],
                domains=[ToolDomain.GENERAL],
                capabilities=[ToolCapability.SEARCH],
                applicable_roles=["researcher", "analyst"],
                task_patterns=["search for", "find information"]
            )
            
            async def arun(self, **kwargs):
                return "Search results"
        
        class CodeTool(EnhancedTool):
            """A test coding tool."""
            
            metadata: ToolMetadata = ToolMetadata(
                name="code_tool",
                categories=[ToolCategory.CODING],
                domains=[ToolDomain.PROGRAMMING],
                capabilities=[ToolCapability.WRITE, ToolCapability.ANALYZE],
                applicable_roles=["developer", "programmer"],
                task_patterns=["write code", "debug", "refactor"]
            )
            
            async def arun(self, **kwargs):
                return "Code snippet"
        
        # Register the tools - using override to ensure clean registry
        registry.clear()
        registry.register_tool(SearchTool(), override=True)
        registry.register_tool(CodeTool(), override=True)
    
    def test_select_tools_for_role(self):
        """Test selecting tools based on a role."""
        # Create a researcher role
        researcher_role = Role(
            name="researcher",
            description="Conducts research and finds information",
            required_capabilities=[ToolCapability.SEARCH],
            domains=[ToolDomain.GENERAL]
        )
        
        # Select tools for the researcher role
        tools = self.selector.select_tools_for_role(researcher_role)
        
        # We should find at least one tool
        self.assertGreaterEqual(len(tools), 1)
        # The first tool should be the search tool
        self.assertEqual(tools[0].metadata.name, "search_tool")
    
    def test_select_tools_for_task(self):
        """Test selecting tools based on a task."""
        # Create a search task
        search_task = Task(
            description="search for information about AI",
            goal="Find information",
            type="research",
            domain=ToolDomain.GENERAL,
            required_capabilities=[ToolCapability.SEARCH]
        )
        
        # Select tools for the search task
        tools = self.selector.select_tools_for_task(search_task)
        
        # We should find at least one tool
        self.assertGreaterEqual(len(tools), 1)
        # The first tool should be the search tool
        self.assertEqual(tools[0].metadata.name, "search_tool")
        
        # Create a coding task
        coding_task = Task(
            description="write code to solve this problem",
            goal="Create code",
            type="coding",
            domain=ToolDomain.PROGRAMMING,
            required_capabilities=[ToolCapability.WRITE]
        )
        
        # Select tools for the coding task
        tools = self.selector.select_tools_for_task(coding_task)
        
        # We should find at least one tool
        self.assertGreaterEqual(len(tools), 1)
        # The first tool should be the code tool
        self.assertEqual(tools[0].metadata.name, "code_tool")


if __name__ == "__main__":
    unittest.main() 