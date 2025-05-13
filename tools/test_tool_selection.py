"""
Tool Selection Test

This script tests the various selection methods in the ToolSelector class,
demonstrating how each method selects tools based on different criteria.
"""
import asyncio
import os
import sys
from typing import List, Dict, Any

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tools.registry import registry
from tools.base import EnhancedTool
from tools.tool_selector import ToolSelector, SelectionMethod
from tools.calculator import CalculatorTool
from tools.adapters.integration import register_all_adapted_tools
from tools.tool_types import (
    ToolMetadata, 
    ToolCategory, 
    ToolCapability, 
    ToolDomain,
    ToolSource,
    Role,
    Task
)

# Set up logging
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WeatherTool(EnhancedTool):
    """Tool for getting weather information."""
    
    location: str
    
    metadata = ToolMetadata(
        name="WeatherTool",
        display_name="Weather Information",
        description="Gets weather information for a specific location",
        categories=[ToolCategory.UTILITY],
        capabilities=[ToolCapability.FETCH],
        domains=[ToolDomain.GENERAL],
        keywords=["weather", "temperature", "forecast", "rain", "climate", "humidity"],
        examples=["What's the weather in New York?", "Is it going to rain tomorrow?"],
        applicable_roles=["traveler", "researcher", "planner"]
    )
    
    source = ToolSource.CUSTOM
    
    async def arun(self, location: str) -> str:
        """Get weather information for a location."""
        # This is a mock implementation
        return f"Weather for {location}: Sunny, 75°F"


class StockPriceTool(EnhancedTool):
    """Tool for getting stock price information."""
    
    ticker: str
    
    metadata = ToolMetadata(
        name="StockPriceTool",
        display_name="Stock Prices",
        description="Gets current stock price information",
        categories=[ToolCategory.UTILITY],
        capabilities=[ToolCapability.FETCH],
        domains=[ToolDomain.FINANCE],
        keywords=["stock", "price", "market", "ticker", "finance", "investment", "NASDAQ", "NYSE"],
        examples=["What's the stock price of AAPL?", "How is Tesla stock doing today?"],
        applicable_roles=["investor", "financial analyst", "trader"]
    )
    
    source = ToolSource.CUSTOM
    
    async def arun(self, ticker: str) -> str:
        """Get stock price information."""
        # This is a mock implementation
        return f"Stock price for {ticker}: $150.25"


class DocumentSearchTool(EnhancedTool):
    """Tool for searching documents."""
    
    query: str
    
    metadata = ToolMetadata(
        name="DocumentSearchTool",
        display_name="Document Search",
        description="Searches through documents for specific information",
        categories=[ToolCategory.SEARCH, ToolCategory.DOCUMENT_PROCESSING],
        capabilities=[ToolCapability.SEARCH, ToolCapability.READ],
        domains=[ToolDomain.GENERAL],
        keywords=["search", "document", "find", "text", "information", "query"],
        examples=["Find documents about renewable energy", "Search for mentions of AI in my files"],
        applicable_roles=["researcher", "writer", "analyst"]
    )
    
    source = ToolSource.CUSTOM
    
    async def arun(self, query: str) -> str:
        """Search documents for information."""
        # This is a mock implementation
        return f"Found 3 documents matching '{query}'"


class ToolSelectionTest:
    """Test class for evaluating the different selection methods."""
    
    def __init__(self):
        """Initialize the test."""
        # Clear registry and register test tools
        registry.clear()
        self._register_test_tools()
        
        # Create selector
        self.selector = ToolSelector(registry)
        
        # Define test queries
        self.test_queries = {
            "weather": "What's the weather like in San Francisco today?",
            "stock": "What is the current stock price of Apple?",
            "calculation": "Calculate 15% of $85 and multiply by 1.2",
            "document": "Find documents related to climate change",
            "mixed": "I need to check the weather and calculate my investment returns"
        }
        
        # Define test role
        self.test_role = Role(
            name="financial_analyst",
            description="Financial analyst who researches markets and analyzes financial data",
            responsibilities=["Market research", "Financial analysis", "Investment recommendations"],
            required_capabilities=[ToolCapability.ANALYZE, ToolCapability.FETCH, ToolCapability.COMPUTE],
            domains=[ToolDomain.FINANCE, ToolDomain.BUSINESS]
        )
        
        # Define test task
        self.test_task = Task(
            description="Analyze stock performance and weather patterns to predict seasonal market trends",
            goal="Create a seasonal market trend prediction model",
            type="analysis",
            domain=ToolDomain.FINANCE,
            required_capabilities=[ToolCapability.ANALYZE, ToolCapability.FETCH, ToolCapability.COMPUTE],
            expected_outputs=["Analysis report", "Prediction model"]
        )
    
    def _register_test_tools(self):
        """Register test tools with the registry."""
        # Register standard tools
        register_all_adapted_tools()
        
        # Register our custom test tools
        registry.register_tool(WeatherTool())
        registry.register_tool(StockPriceTool())
        registry.register_tool(DocumentSearchTool())
        registry.register_tool(CalculatorTool())
        
        logger.info(f"Registered {len(registry.get_all_tools())} tools for testing")
    
    def test_keyword_selection(self):
        """Test keyword-based tool selection."""
        print("\n=== TESTING KEYWORD-BASED SELECTION ===")
        
        for name, query in self.test_queries.items():
            print(f"\nQuery: \"{query}\"")
            tools = self.selector.select_tools(
                query=query,
                method=SelectionMethod.KEYWORD,
                max_tools=2
            )
            
            self._print_selected_tools(tools)
    
    def test_category_selection(self):
        """Test category-based tool selection."""
        print("\n=== TESTING CATEGORY-BASED SELECTION ===")
        
        for name, query in self.test_queries.items():
            print(f"\nQuery: \"{query}\"")
            tools = self.selector.select_tools(
                query=query,
                method=SelectionMethod.CATEGORY,
                max_tools=2
            )
            
            self._print_selected_tools(tools)
    
    def test_capability_selection(self):
        """Test capability-based tool selection."""
        print("\n=== TESTING CAPABILITY-BASED SELECTION ===")
        
        for name, query in self.test_queries.items():
            print(f"\nQuery: \"{query}\"")
            tools = self.selector.select_tools(
                query=query,
                method=SelectionMethod.CAPABILITY,
                max_tools=2
            )
            
            self._print_selected_tools(tools)
    
    def test_domain_selection(self):
        """Test domain-based tool selection."""
        print("\n=== TESTING DOMAIN-BASED SELECTION ===")
        
        for name, query in self.test_queries.items():
            print(f"\nQuery: \"{query}\"")
            tools = self.selector.select_tools(
                query=query,
                method=SelectionMethod.DOMAIN,
                max_tools=2
            )
            
            self._print_selected_tools(tools)
    
    def test_role_selection(self):
        """Test role-based tool selection."""
        print("\n=== TESTING ROLE-BASED SELECTION ===")
        
        role = self.test_role
        print(f"\nRole: {role.name} - {role.description}")
        tools = self.selector.select_tools(
            query="",
            method=SelectionMethod.ROLE,
            role=role.name,
            max_tools=3
        )
        
        self._print_selected_tools(tools)
    
    def test_hybrid_selection(self):
        """Test hybrid tool selection."""
        print("\n=== TESTING HYBRID SELECTION ===")
        
        for name, query in self.test_queries.items():
            print(f"\nQuery: \"{query}\"")
            tools = self.selector.select_tools(
                query=query,
                method=SelectionMethod.HYBRID,
                role=self.test_role.name,
                max_tools=3
            )
            
            self._print_selected_tools(tools)
    
    def _print_selected_tools(self, tools: List[EnhancedTool]):
        """Print information about selected tools."""
        if not tools:
            print("  No tools selected")
            return
            
        print(f"  Selected {len(tools)} tools:")
        for i, tool in enumerate(tools):
            print(f"    {i+1}. {tool.metadata.name} ({tool.metadata.display_name})")
            print(f"       Categories: {[cat.value for cat in tool.metadata.categories]}")
            print(f"       Capabilities: {[cap.value for cap in tool.metadata.capabilities]}")
            print(f"       Source: {tool.source}")
    
    def run_all_tests(self):
        """Run all selection method tests."""
        print("\nTESTING TOOL SELECTION METHODS")
        print("-" * 50)
        
        self.test_keyword_selection()
        self.test_category_selection()
        self.test_capability_selection()
        self.test_domain_selection()
        self.test_role_selection()
        self.test_hybrid_selection()
        
        print("\nAll tests completed.\n")


async def main():
    """Run the tool selection tests."""
    test = ToolSelectionTest()
    test.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main()) 