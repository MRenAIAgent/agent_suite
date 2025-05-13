"""
Enhanced React Agent with Dynamic Tool Selection

This example demonstrates using the ToolSelector with a React agent to:
1. Dynamically select tools based on user queries
2. Integrate with React agent's thinking and execution flow
3. Show how tool selection improves agent capabilities
"""
import asyncio
import os
import sys
from typing import List, Dict, Any, Optional

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.react_agent import ReActAgent
from llm.litellm.litellm import LiteLLM
from log.logging import LogManager

# Import enhanced tool system components
from tools.tool_selector import ToolSelector, SelectionMethod
from tools.registry import registry
from tools.base import EnhancedTool

# Import tool adapters
from tools.adapters.integration import register_all_adapted_tools

# Import demo tools for testing
from tools.serpapi_tool import SerpApiSearchTool
from tools.animal_type import AnimalType
from tools.calculator import CalculatorTool

class EnhancedReactAgent:
    """React agent with enhanced dynamic tool selection capabilities."""
    
    def __init__(
        self,
        model: str = "anthropic/claude-3-7-sonnet-20250219",
        max_iterations: int = 15
    ):
        """Initialize the EnhancedReactAgent.
        
        Args:
            model: The LLM model to use
            max_iterations: Maximum number of iterations for the React agent
        """
        self.llm = LiteLLM.create_llm()
        self.model = model
        self.max_iterations = max_iterations
        self.log_manager = LogManager()
        
        # Initialize tool registry and selector
        self._initialize_tools()
        self.tool_selector = ToolSelector(registry)
        
        # Initialize the React agent with default tools
        # We'll update tools later based on the query
        self.agent = None
    
    def _initialize_tools(self):
        """Initialize the tool registry with available tools."""
        # Clear registry to start fresh
        registry.clear()
        
        # Register all adapted tools
        register_all_adapted_tools()
        
        # Register additional tools for demonstration
        calculator = CalculatorTool()
        registry.register_tool(calculator)
        
        # Ensure we have core search tool
        registry.register_tool(SerpApiSearchTool())
        registry.register_tool(AnimalType())
        
        print(f"Registered {len(registry.get_all_tools())} tools with the enhanced tool system")
    
    def _select_tools_for_query(self, query: str, max_tools: int = 5) -> List[Any]:
        """Select appropriate tools based on the user query.
        
        Args:
            query: User's query text
            max_tools: Maximum number of tools to select
            
        Returns:
            List of original tool instances for the React agent
        """
        # Use the tool selector to find relevant tools
        enhanced_tools = self.tool_selector.select_tools(
            query=query,
            method=SelectionMethod.HYBRID,
            max_tools=max_tools
        )
        
        # Convert enhanced tools to original formats
        # This is needed because ReactAgent expects the original tool formats
        original_tools = []
        for enhanced_tool in enhanced_tools:
            tool_name = enhanced_tool.metadata.name
            
            # Map enhanced tools to original tool instances
            if "SerpApiSearchTool" in tool_name:
                original_tools.append(SerpApiSearchTool())
            elif "AnimalType" in tool_name:
                original_tools.append(AnimalType())
            elif "CalculatorTool" in tool_name:
                original_tools.append(CalculatorTool())
            # Add more mappings as needed
        
        # Add a default search tool if no tools were selected
        if not original_tools:
            original_tools.append(SerpApiSearchTool())
        
        # Print selected tools for demonstration
        print(f"\nSelected {len(original_tools)} tools for query: '{query}'")
        for i, tool in enumerate(original_tools):
            print(f"  {i+1}. {tool.__class__.__name__}")
        
        return original_tools
    
    def _initialize_agent(self, query: str, tools: List[Any]):
        """Initialize or update the React agent with selected tools.
        
        Args:
            query: User's query text
            tools: List of tools to use
        """
        # Define React agent role and task
        role = "You are a helpful AI assistant that can use tools to accomplish tasks."
        task = f"Answer the user's question: {query}"
        guide = (
            "Think step by step and use the available tools when necessary. "
            "Always explain your reasoning and the significance of your findings."
        )
        
        # Create a new agent with the selected tools
        self.agent = ReActAgent(
            llm=self.llm,
            role=role,
            task=task,
            guide=guide,
            tools=tools,
            log_manager=self.log_manager,
            max_iterations=self.max_iterations
        )
    
    async def arun(self, query: str) -> str:
        """Run the enhanced React agent with dynamic tool selection.
        
        Args:
            query: User's query
            
        Returns:
            Agent's response
        """
        # Select tools based on the query
        tools = self._select_tools_for_query(query)
        
        # Initialize or update the agent with selected tools
        self._initialize_agent(query, tools)
        
        # Run the agent and return its response
        return await self.agent.arun(query, self.model)
    
    def run(self, query: str) -> str:
        """Run the agent synchronously.
        
        Args:
            query: User's query
            
        Returns:
            Agent's response
        """
        return asyncio.run(self.arun(query))


async def main():
    """Run a demonstration of the EnhancedReactAgent."""
    agent = EnhancedReactAgent()
    
    # Example queries demonstrating dynamic tool selection
    queries = [
        "What is the capital of France?",
        "Calculate 356 * 288 and explain the process",
        "Is a dolphin a fish or a mammal?",
        "What are the latest financial news about Apple?",
        "How do I implement binary search in Python?"
    ]
    
    for query in queries:
        print(f"\n\n=== QUERY: {query} ===")
        response = await agent.arun(query)
        print(f"\nRESPONSE: {response}")
        print("\n" + "="*50)


if __name__ == "__main__":
    asyncio.run(main()) 