#!/usr/bin/env python3
"""
Enhanced React Agent Example

This script demonstrates a React agent that properly uses the enhanced tool 
registration system with namespaces and metadata. It shows how to:
1. Register tools from different sources with namespaces
2. Utilize tool metadata for better tool selection
3. Run the agent with the enhanced tools

Run this script to see how the agent uses the enhanced tool registry.
"""
import os
import sys
import asyncio
from typing import List, Optional, Dict, Any, ClassVar
import logging

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Import agent components
from agents.react_agent import ReActAgent
from llm.litellm.litellm import LiteLLM
from log.logging import LogManager

# Import enhanced tool system components
from tools.registry import registry
from tools.base import EnhancedTool
from tools.tool_types import (
    ToolCategory, 
    ToolCapability, 
    ToolDomain, 
    ToolMetadata, 
    ToolSource
)

# Import tools for registration
from tools.serper_api import SerperSearchTool
from tools.tool import Tool
from tools.tool_selector import ToolSelector, SelectionMethod

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CalculatorTool(EnhancedTool):
    """A simple calculator tool that can perform basic arithmetic operations."""
    
    # Tool parameters
    operation: str
    a: float
    b: float
    
    # Tool metadata as ClassVar to avoid Pydantic issues
    metadata: ClassVar[ToolMetadata] = ToolMetadata(
        name="calculator",
        display_name="Calculator",
        description="Perform basic arithmetic operations: add, subtract, multiply, divide",
        categories=[ToolCategory.UTILITY],
        domains=[ToolDomain.GENERAL],
        capabilities=[ToolCapability.ANALYZE],
        examples=["Calculate 5 + 3", "Calculate 10 * 2"]
    )
    
    async def arun(self, **kwargs) -> str:
        """Run the calculator with the given operation and values."""
        operation = kwargs.get("operation", self.operation)
        a = kwargs.get("a", self.a)
        b = kwargs.get("b", self.b)
        
        if operation == "add":
            return f"{a} + {b} = {a + b}"
        elif operation == "subtract":
            return f"{a} - {b} = {a - b}"
        elif operation == "multiply":
            return f"{a} * {b} = {a * b}"
        elif operation == "divide":
            if b == 0:
                return "Error: Division by zero is not allowed."
            return f"{a} / {b} = {a / b}"
        else:
            return f"Unknown operation: {operation}. Supported operations are: add, subtract, multiply, divide."


class WeatherTool(EnhancedTool):
    """A mock weather tool that provides weather information for a location."""
    
    # Tool parameters
    location: str
    
    # Tool metadata as ClassVar to avoid Pydantic issues
    metadata: ClassVar[ToolMetadata] = ToolMetadata(
        name="weather",
        display_name="Weather Information",
        description="Get weather information for a specific location",
        categories=[ToolCategory.SEARCH],
        domains=[ToolDomain.GENERAL],
        capabilities=[ToolCapability.FETCH],
        examples=["Get weather for New York", "Check temperature in London"]
    )
    
    async def arun(self, **kwargs) -> str:
        """Get weather information for the location."""
        location = kwargs.get("location", self.location)
        
        # This is a mock implementation - in a real tool, you would call a weather API
        weather_data = {
            "New York": {"condition": "Sunny", "temperature": "75°F", "humidity": "45%"},
            "London": {"condition": "Rainy", "temperature": "62°F", "humidity": "80%"},
            "Tokyo": {"condition": "Cloudy", "temperature": "70°F", "humidity": "60%"},
            "Sydney": {"condition": "Clear", "temperature": "82°F", "humidity": "55%"},
            "Paris": {"condition": "Partly Cloudy", "temperature": "68°F", "humidity": "50%"}
        }
        
        if location in weather_data:
            data = weather_data[location]
            return f"Weather in {location}: {data['condition']}, Temperature: {data['temperature']}, Humidity: {data['humidity']}"
        else:
            # Return a random response for any other location
            import random
            conditions = ["Sunny", "Cloudy", "Rainy", "Clear", "Partly Cloudy", "Overcast"]
            temp = random.randint(50, 90)
            humidity = random.randint(30, 95)
            return f"Weather in {location}: {random.choice(conditions)}, Temperature: {temp}°F, Humidity: {humidity}%"


class StockPriceTool(EnhancedTool):
    """A mock stock price tool that provides current stock prices and basic info."""
    
    # Tool parameters
    symbol: str
    
    # Tool metadata as ClassVar to avoid Pydantic issues
    metadata: ClassVar[ToolMetadata] = ToolMetadata(
        name="stock_price",
        display_name="Stock Price Lookup",
        description="Get current stock price and basic information for a company",
        categories=[ToolCategory.SEARCH],
        domains=[ToolDomain.FINANCE],
        capabilities=[ToolCapability.FETCH],
        examples=["Get stock price for AAPL", "Check current price of TSLA"]
    )
    
    async def arun(self, **kwargs) -> str:
        """Get stock price information for the symbol."""
        symbol = kwargs.get("symbol", self.symbol)
        
        # Mock stock data - in a real tool, you would call a financial API
        stock_data = {
            "AAPL": {"price": "185.92", "change": "+1.25", "volume": "62.5M", "market_cap": "2.89T"},
            "TSLA": {"price": "175.34", "change": "-2.18", "volume": "78.2M", "market_cap": "558.6B"},
            "MSFT": {"price": "402.65", "change": "+0.87", "volume": "25.4M", "market_cap": "3.1T"},
            "GOOG": {"price": "175.98", "change": "+0.45", "volume": "22.3M", "market_cap": "2.2T"},
            "AMZN": {"price": "180.75", "change": "-0.92", "volume": "35.7M", "market_cap": "1.87T"}
        }
        
        symbol = symbol.upper()
        if symbol in stock_data:
            data = stock_data[symbol]
            return (
                f"Stock information for {symbol}:\n"
                f"Price: ${data['price']}\n"
                f"Change: {data['change']}\n"
                f"Volume: {data['volume']}\n"
                f"Market Cap: ${data['market_cap']}"
            )
        else:
            return f"Could not find stock information for symbol: {symbol}"


# Create a simple base class for our tools
class SimpleTool:
    """A simple base class for tools that doesn't rely on Pydantic."""
    
    name = "base_tool"
    description = "Base tool description"
    
    def run(self, **kwargs):
        raise NotImplementedError("Subclasses must implement this")
    
    async def arun(self, **kwargs):
        return self.run(**kwargs)
    
    def convert_to_function_call(self):
        """Convert this tool to a function call format for LLM API.
        
        This is required by ReActAgent.
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": self._get_parameters(),
                    "required": []
                }
            }
        }
    
    def _get_parameters(self):
        """Get tool parameters in JSON Schema format."""
        # Default implementation with minimal parameters
        # Override in subclasses for more specific parameters
        return {
            "input": {
                "type": "string",
                "description": "Input for the tool"
            }
        }


# Create corresponding Tool instances for custom tools
class SimpleCalculatorTool(SimpleTool):
    """Simple calculator tool."""
    
    name = "calculator"
    description = "Perform basic arithmetic operations: add, subtract, multiply, divide"
    
    def __init__(self, operation="add", a=0, b=0):
        self.operation = operation
        self.a = a
        self.b = b
    
    def _get_parameters(self):
        """Get calculator parameters in JSON Schema format."""
        return {
            "operation": {
                "type": "string",
                "description": "Mathematical operation to perform",
                "enum": ["add", "subtract", "multiply", "divide"]
            },
            "a": {
                "type": "number",
                "description": "First number"
            },
            "b": {
                "type": "number",
                "description": "Second number"
            }
        }
    
    def run(self, **kwargs):
        operation = kwargs.get("operation", self.operation)
        a = kwargs.get("a", self.a)
        b = kwargs.get("b", self.b)
        
        if operation == "add":
            return f"{a} + {b} = {a + b}"
        elif operation == "subtract":
            return f"{a} - {b} = {a - b}"
        elif operation == "multiply":
            return f"{a} * {b} = {a * b}"
        elif operation == "divide":
            if b == 0:
                return "Error: Division by zero is not allowed."
            return f"{a} / {b} = {a / b}"
        else:
            return f"Unknown operation: {operation}. Supported operations are: add, subtract, multiply, divide."

class SimpleWeatherTool(SimpleTool):
    """Simple weather tool."""
    
    name = "weather"
    description = "Get weather information for a specific location"
    
    def __init__(self, location="New York"):
        self.location = location
    
    def _get_parameters(self):
        """Get weather tool parameters in JSON Schema format."""
        return {
            "location": {
                "type": "string",
                "description": "City or location to get weather for"
            }
        }
    
    def run(self, **kwargs):
        location = kwargs.get("location", self.location)
        
        # Mock implementation
        weather_data = {
            "New York": {"condition": "Sunny", "temperature": "75°F", "humidity": "45%"},
            "London": {"condition": "Rainy", "temperature": "62°F", "humidity": "80%"},
            "Tokyo": {"condition": "Cloudy", "temperature": "70°F", "humidity": "60%"}
        }
        
        if location in weather_data:
            data = weather_data[location]
            return f"Weather in {location}: {data['condition']}, Temperature: {data['temperature']}, Humidity: {data['humidity']}"
        else:
            import random
            conditions = ["Sunny", "Cloudy", "Rainy", "Clear", "Partly Cloudy"]
            temp = random.randint(50, 90)
            humidity = random.randint(30, 95)
            return f"Weather in {location}: {random.choice(conditions)}, Temperature: {temp}°F, Humidity: {humidity}%"

class SimpleStockPriceTool(SimpleTool):
    """Simple stock price tool."""
    
    name = "stock_price"
    description = "Get current stock price and basic information for a company"
    
    def __init__(self, symbol="AAPL"):
        self.symbol = symbol
    
    def _get_parameters(self):
        """Get stock price tool parameters in JSON Schema format."""
        return {
            "symbol": {
                "type": "string",
                "description": "Stock symbol/ticker (e.g., AAPL, TSLA, MSFT)"
            }
        }
    
    def run(self, **kwargs):
        symbol = kwargs.get("symbol", self.symbol)
        
        # Mock implementation
        stock_data = {
            "AAPL": {"price": "185.92", "change": "+1.25", "volume": "62.5M"},
            "TSLA": {"price": "175.34", "change": "-2.18", "volume": "78.2M"},
            "MSFT": {"price": "402.65", "change": "+0.87", "volume": "25.4M"}
        }
        
        symbol = symbol.upper()
        if symbol in stock_data:
            data = stock_data[symbol]
            return f"Stock information for {symbol}: Price: ${data['price']}, Change: {data['change']}, Volume: {data['volume']}"
        else:
            return f"Could not find stock information for symbol: {symbol}"


class EnhancedReactAgentExample:
    """Example implementation of React Agent with properly registered enhanced tools."""
    
    def __init__(
        self,
        model: str = "anthropic/claude-3-7-sonnet-20250219",
        role: str = "You are a helpful assistant that can use tools to answer questions about weather, stocks, and perform calculations.",
        task: str = "Help users get information about weather, stocks, and perform calculations.",
        guide: str = "Use the appropriate tools based on the user's query. For weather questions, use the weather tool. For stock information, use the stock price tool. For calculations, use the calculator tool.",
        max_iterations: int = 15,
        examples: Optional[List[str]] = None
    ):
        """Initialize the EnhancedReactAgentExample.
        
        Args:
            model: The LLM model to use
            role: The role of the agent
            task: The task of the agent
            guide: Guidelines for the agent
            max_iterations: Maximum number of iterations for the agent
            examples: Optional examples for the agent
        """
        self.llm = LiteLLM.create_llm()
        self.model = model
        self.role = role
        self.task = task
        self.guide = guide
        self.max_iterations = max_iterations
        
        # Initialize the original tools map
        self._original_tools_map = {}
        self.original_tools = []
        
        # Clear and initialize the tool registry
        registry.clear()
        
        # Store reference to registry
        self.registry = registry
        
        # Create tool selector
        self.tool_selector = ToolSelector(registry)
        
        # Register enhanced tools with proper namespaces
        self._register_tools()
        
        # Get tools for use with the agent
        self.tools = registry.get_all_tools()
        
        # Initialize the React agent with base tools
        # We'll dynamically select appropriate tools per query
        self.agent = self._create_agent(examples or [])
    
    def _create_agent(self, examples: List[str]) -> ReActAgent:
        """Create a ReActAgent instance with the basic tools.
        
        Args:
            examples: Example queries for the agent
            
        Returns:
            Configured ReActAgent instance
        """
        return ReActAgent(
            llm=self.llm,
            role=self.role,
            task=self.task,
            guide=self.guide,
            examples=examples,
            tools=self.original_tools,  # Use the original tools format for now
            log_manager=LogManager(),
            max_iterations=self.max_iterations
        )
    
    def _register_tools(self):
        """Register all enhanced tools with proper namespaces."""
        # Register SerperSearchTool in search namespace
        serper_tool = SerperSearchTool(query="")
        enhanced_serper = serper_tool.to_enhanced_tool()
        registry.register_tools_with_namespace("search", [enhanced_serper])
        
        # Register calculator tool in utility namespace
        calculator = CalculatorTool(operation="add", a=0, b=0)
        registry.register_tools_with_namespace("utility", [calculator])
        
        # Register weather tool in weather namespace
        weather = WeatherTool(location="New York")
        registry.register_tools_with_namespace("weather", [weather])
        
        # Register stock price tool in finance namespace
        stock = StockPriceTool(symbol="AAPL")
        registry.register_tools_with_namespace("finance", [stock])
        
        # Create original tools for use with ReActAgent
        # This is needed because ReActAgent currently expects original tool format
        
        # Add serper search tool
        serper_tool_org = SerperSearchTool(query="")
        self.original_tools.append(serper_tool_org)
        self._original_tools_map["search"] = serper_tool_org
        
        # Add calculator tool
        calc_tool_org = SimpleCalculatorTool()
        self.original_tools.append(calc_tool_org)
        self._original_tools_map["calculator"] = calc_tool_org
        
        # Add weather tool
        weather_tool_org = SimpleWeatherTool()
        self.original_tools.append(weather_tool_org)
        self._original_tools_map["weather"] = weather_tool_org
        
        # Add stock price tool
        stock_tool_org = SimpleStockPriceTool()
        self.original_tools.append(stock_tool_org)
        self._original_tools_map["stock_price"] = stock_tool_org
        
        logger.info(f"Registered {len(registry.get_all_tools())} tools with the registry")
    
    def _get_original_tools_for_enhanced(self, enhanced_tools: List[EnhancedTool]) -> List[SimpleTool]:
        """Convert enhanced tools to their original tool equivalents.
        
        Args:
            enhanced_tools: List of enhanced tools selected for the query
            
        Returns:
            List of original tool equivalents for ReActAgent
        """
        original_tools = []
        
        # Create a set of tool names we want
        tool_names = {tool.metadata.name for tool in enhanced_tools}
        
        # Add the original tool equivalents that match the enhanced tools
        for name, tool in self._original_tools_map.items():
            if name in tool_names:
                original_tools.append(tool)
        
        # Always include search as a fallback
        if self._original_tools_map["search"] not in original_tools:
            original_tools.append(self._original_tools_map["search"])
        
        return original_tools
    
    async def arun(self, user_input: str) -> str:
        """Run the agent asynchronously with dynamically selected tools.
        
        Args:
            user_input: The user's input
            
        Returns:
            The agent's response
        """
        # Select appropriate tools based on the query
        selected_tools = self.tool_selector.select_tools(
            query=user_input,
            method=SelectionMethod.HYBRID,
            role=self.role,
            task=self.task
        )
        
        # Log the selected tools
        logger.info(f"Selected {len(selected_tools)} tools for query: {user_input}")
        for i, tool in enumerate(selected_tools):
            logger.info(f"  {i+1}. {tool.metadata.name}: {tool.metadata.display_name}")
        
        # Convert to original tool format for the agent
        original_tools = self._get_original_tools_for_enhanced(selected_tools)
        
        # Update the agent's tools for this query
        self.agent.tools = original_tools
        
        # Run the agent with the dynamically selected tools
        return await self.agent.arun(user_input, self.model)
    
    def run(self, user_input: str) -> str:
        """Run the agent synchronously with dynamically selected tools.
        
        Args:
            user_input: The user's input
            
        Returns:
            The agent's response
        """
        return asyncio.run(self.arun(user_input))
    
    def get_registered_tools(self) -> Dict[str, List[EnhancedTool]]:
        """Get all enhanced tools registered in the registry, organized by namespace.
        
        Returns:
            Dictionary of namespace to list of tools
        """
        result = {}
        for namespace in registry.namespaces:
            tools = registry.get_tools_in_namespace(namespace)
            if tools:
                result[namespace] = tools
        return result


# Example usage
if __name__ == "__main__":
    # Create agent with enhanced tools
    agent = EnhancedReactAgentExample(max_iterations=15)
    
    # Display the registered tools by namespace
    tools_by_namespace = agent.get_registered_tools()
    print("\n=== Registered Enhanced Tools by Namespace ===")
    
    for namespace, tools in tools_by_namespace.items():
        print(f"\nNamespace: {namespace} ({len(tools)} tools)")
        for tool in tools:
            print(f"- {tool.metadata.display_name}: {tool.metadata.description}")
    
    # Example queries to test the agent
    test_queries = [
        "What is the weather like in Tokyo?",
        "What is the current stock price of Tesla?",
        "Calculate 125 divided by 5"
    ]
    
    # Run the agent on each query
    for query in test_queries:
        print(f"\n\n{'='*50}")
        print(f"QUERY: {query}")
        print(f"{'='*50}")
        
        response = agent.run(query)
        print(f"\nRESPONSE:\n{response}") 