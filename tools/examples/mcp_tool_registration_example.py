#!/usr/bin/env python3
"""
MCP Tool Registration Example

This script demonstrates how to register MCP tools with the enhanced tool system.
It shows how to:

1. Create MCP tool wrappers from MCP tool instances
2. Register MCP tools with appropriate namespaces
3. Use the tool selector to select MCP tools based on query
4. Display the enhanced tool metadata derived from MCP tools
"""
import os
import sys
import json
import argparse
import logging
from typing import Dict, Any, List
from pprint import pprint

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Import the necessary modules
from tools.registry import registry
from tools.mcp_wrapper import register_mcp_tool, register_mcp_tools, MCPToolWrapper
from tools.tool_selector import ToolSelector, SelectionMethod
from tools.tool_types import ToolCapability, ToolCategory, ToolDomain

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Mock MCP tools for demonstration purposes
class MockMCPTool:
    """Mock MCP tool for demonstration purposes."""
    
    def __init__(self, name, description):
        self.name = name
        self.description = description
    
    def run(self, **kwargs):
        """Run the mock MCP tool."""
        args_str = ", ".join(f"{k}={v}" for k, v in kwargs.items())
        return f"Running {self.name}: {self.description} with args: {args_str}"


def create_mock_mcp_tools() -> Dict[str, Dict[str, Any]]:
    """Create mock MCP tools for demonstration.
    
    Returns:
        Dictionary of MCP tools with their data and instances
    """
    # Weather tool
    weather_tool = MockMCPTool(
        name="weather",
        description="Get weather information for a location"
    )
    weather_data = {
        "description": "Get weather information for a location",
        "display_name": "Weather Information",
        "categories": ["search"],
        "capabilities": ["fetch", "read"],
        "domains": ["general"],
        "examples": [
            "What's the weather in London?",
            "Is it going to rain tomorrow in New York?"
        ],
        "keywords": ["weather", "forecast", "temperature", "climate"]
    }
    
    # GitHub search tool
    github_search = MockMCPTool(
        name="search_repositories",
        description="Search for GitHub repositories"
    )
    github_search_data = {
        "description": "Search for GitHub repositories",
        "display_name": "GitHub Repository Search",
        "categories": ["search"],
        "capabilities": ["search", "read"],
        "domains": ["programming"],
        "examples": [
            "Find Python repositories for data science",
            "Search for JavaScript frontend frameworks"
        ],
        "keywords": ["github", "repository", "code", "search"]
    }
    
    # Calculator tool
    calculator = MockMCPTool(
        name="calculator",
        description="Perform basic arithmetic operations"
    )
    calculator_data = {
        "description": "Perform basic arithmetic operations",
        "display_name": "Simple Calculator",
        "categories": ["utility"],
        "capabilities": ["compute"],
        "domains": ["general"],
        "examples": [
            "Calculate 125 + 375",
            "What is 500 * 0.15?"
        ],
        "keywords": ["calculate", "math", "arithmetic", "compute"]
    }
    
    # Stock price tool
    stock_price = MockMCPTool(
        name="stock_price",
        description="Get current stock price information"
    )
    stock_price_data = {
        "description": "Get current stock price information",
        "display_name": "Stock Price Lookup",
        "categories": ["search"],
        "capabilities": ["fetch", "read"],
        "domains": ["finance"],
        "examples": [
            "What's the current price of AAPL?",
            "How is TSLA stock performing today?"
        ],
        "keywords": ["stock", "price", "market", "finance", "investing"]
    }
    
    # Web browser tool
    browser = MockMCPTool(
        name="browserbase_navigate",
        description="Navigate to a URL in a web browser"
    )
    browser_data = {
        "description": "Navigate to a URL in a web browser",
        "display_name": "Web Browser Navigation",
        "categories": ["web_browsing"],
        "capabilities": ["read"],
        "domains": ["general"],
        "examples": [
            "Go to example.com",
            "Open the website at https://github.com"
        ],
        "keywords": ["browser", "web", "navigate", "open", "url"]
    }
    
    # Return as a dictionary with the MCP format
    return {
        "weather.get_weather": {
            "instance": weather_tool,
            "data": weather_data,
            "namespace": "weather"
        },
        "github.search_repositories": {
            "instance": github_search,
            "data": github_search_data,
            "namespace": "github"
        },
        "utility.calculator": {
            "instance": calculator,
            "data": calculator_data,
            "namespace": "utility"
        },
        "finance.stock_price": {
            "instance": stock_price,
            "data": stock_price_data,
            "namespace": "finance"
        },
        "browser.browserbase_navigate": {
            "instance": browser,
            "data": browser_data,
            "namespace": "browser"
        }
    }


def display_registered_tools():
    """Display all registered tools by namespace."""
    print("\n=== Registered Tools by Namespace ===")
    
    # Get namespaces from registry
    for namespace in registry.namespaces:
        tools = registry.get_tools_in_namespace(namespace)
        if not tools:
            continue
        
        print(f"\nNamespace: {namespace} ({len(tools)} tools)")
        for i, tool in enumerate(tools):
            print(f"{i+1}. {tool.metadata.display_name}")
            print(f"   Description: {tool.metadata.description}")
            print(f"   Categories: {[c.value for c in tool.metadata.categories]}")
            print(f"   Capabilities: {[c.value for c in tool.metadata.capabilities]}")
            print(f"   Domains: {[d.value for d in tool.metadata.domains]}")


def test_tool_selection():
    """Test selecting tools based on different queries."""
    # Create a tool selector
    selector = ToolSelector()
    
    # Define test queries
    test_queries = [
        "What's the weather like in San Francisco?",
        "Find GitHub repositories for machine learning",
        "Calculate 234 * 56",
        "What's the current stock price of MSFT?",
        "Open the website example.com"
    ]
    
    print("\n=== Tool Selection Tests ===")
    for query in test_queries:
        print(f"\nQuery: {query}")
        
        # Select tools using different methods
        keyword_tools = selector.select_tools(query, method=SelectionMethod.KEYWORD)
        category_tools = selector.select_tools(query, method=SelectionMethod.CATEGORY)
        hybrid_tools = selector.select_tools(query, method=SelectionMethod.HYBRID)
        
        # Display results for keyword selection
        print(f"Keyword selection ({len(keyword_tools)} tools):")
        for i, tool in enumerate(keyword_tools):
            print(f"  {i+1}. {tool.metadata.display_name} ({tool.metadata.name})")
        
        # Display results for hybrid selection
        print(f"Hybrid selection ({len(hybrid_tools)} tools):")
        for i, tool in enumerate(hybrid_tools):
            print(f"  {i+1}. {tool.metadata.display_name} ({tool.metadata.name})")


def run_mcp_tool_example():
    """Run the MCP tool registration example."""
    # Clear the registry before starting
    registry.clear()
    
    print("=== MCP Tool Registration Example ===")
    
    # Create mock MCP tools
    mcp_tools = create_mock_mcp_tools()
    print(f"Created {len(mcp_tools)} mock MCP tools")
    
    # Define namespace mapping (optional)
    namespace_mapping = {
        "weather": "weather",
        "github": "code",
        "utility": "utility",
        "finance": "finance",
        "browser": "web"
    }
    
    # Register all MCP tools
    wrappers = register_mcp_tools(mcp_tools, namespace_mapping)
    print(f"Registered {len(wrappers)} MCP tools with the registry")
    
    # Display all registered tools
    display_registered_tools()
    
    # Test tool selection
    test_tool_selection()
    
    # Test running a tool
    weather_wrapper = next(
        tool for tool in registry.get_tools_in_namespace("weather")
        if tool.metadata.name == "weather"
    )
    
    print("\n=== Running MCP Tool ===")
    print("Running weather tool...")
    result = weather_wrapper._mcp_tool_instance.run(location="London")
    print(f"Result: {result}")


if __name__ == "__main__":
    # Run the example
    run_mcp_tool_example() 