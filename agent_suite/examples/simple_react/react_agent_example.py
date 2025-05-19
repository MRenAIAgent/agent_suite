import asyncio
import os
import sys
from typing import List, Optional, ClassVar

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from agent_suite.agent_suite.agents.react.agent import ReActAgent
from agent_suite.agent_suite.llm.litellm.litellm import LiteLLM
from agent_suite.agent_suite.tools.registry import registry
from agent_suite.agent_suite.tools.base import EnhancedTool
from agent_suite.agent_suite.tools.types import ToolCategory, ToolCapability, ToolDomain, ToolMetadata, ToolSource
from pydantic import Field

# Import mock tools for the example
class MockSearchTool:
    def __init__(self, query=""):
        self.name = "search"
        self.description = "Search the web for information about a topic"
        self.query = query

    def run(self, query):
        return f"Search results for: {query}"

    def to_enhanced_tool(self):
        metadata = ToolMetadata(
            name="search",
            display_name="Web Search",
            description="Search the web for information about a topic",
            categories=[ToolCategory.SEARCH],
            capabilities=[ToolCapability.SEARCH],
            domains=[ToolDomain.GENERAL],
            keywords=["search", "web", "information"]
        )
        
        return MockEnhancedSearchTool(metadata=metadata)

class MockEnhancedSearchTool(EnhancedTool):
    name: ClassVar[str] = "search"  # Use ClassVar for class attributes in Pydantic
    description: ClassVar[str] = "Search the web for information about a topic"
    
    async def arun(self, query: str) -> str:
        return f"Mock search results for: {query}"
        
    def run(self, query: str) -> str:
        return asyncio.run(self.arun(query))


class ReactAgentExample:
    """Example implementation of ReactAgent with enhanced tool system."""
    
    def __init__(
        self,
        model: str = "anthropic/claude-3-7-sonnet-20250219",
        role: str = "You are a helpful assistant that can answer questions about a variety of topics.",
        task: str = "Answer the user's question as accurately and completely as possible.",
        guide: str = "Use the available tools when necessary to gather accurate information.",
        max_iterations: int = 20,
        examples: Optional[List[str]] = None
    ):
        """Initialize the ReactAgentExample with enhanced tool system.
        
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
        
        # Clear and initialize the tool registry
        registry.clear()
        
        # Register enhanced tools
        self._register_tools()
        
        # Get tools from registry for the agent
        self.tools = self._get_tools_from_registry()
        
        # Initialize the ReactAgent with tools from registry
        self.agent = ReActAgent(
            llm=self.llm,
            role=role,
            task=task,
            guide=guide,
            tools=self.tools,
            max_iterations=max_iterations
        )
    
    def _register_tools(self):
        """Register all enhanced tools with the registry."""
        # Create and register mock search tool
        search_tool = MockSearchTool()
        enhanced_search = search_tool.to_enhanced_tool()
        registry.register_tool(enhanced_search)
        
        print(f"Registered {len(registry.get_all_tools())} tools with the registry")
    
    def _get_tools_from_registry(self):
        """Get tools from registry for use with the agent.
        
        Returns:
            List of tools for the agent
        """
        # Get enhanced tools from registry
        enhanced_tools = registry.get_all_tools()
        
        # For ReActAgent, we need to return the actual tool instances
        return enhanced_tools
    
    async def arun(self, user_input: str) -> str:
        """Run the agent asynchronously.
        
        Args:
            user_input: The user's input
            
        Returns:
            The agent's response
        """
        return await self.agent.arun(user_input, self.model)
    
    def run(self, user_input: str) -> str:
        """Run the agent synchronously.
        
        Args:
            user_input: The user's input
            
        Returns:
            The agent's response
        """
        return asyncio.run(self.arun(user_input))
    
    def get_registered_tools(self) -> List[EnhancedTool]:
        """Get all enhanced tools registered in the registry.
        
        Returns:
            List of registered enhanced tools
        """
        return registry.get_all_tools()


# Example usage
if __name__ == "__main__":
    # Create agent with enhanced tools
    agent = ReactAgentExample(max_iterations=30)
    
    # Display the registered tools
    enhanced_tools = agent.get_registered_tools()
    print(f"\nRegistered {len(enhanced_tools)} enhanced tools:")
    for tool in enhanced_tools:
        print(f"- {tool.metadata.display_name}: {tool.metadata.description}")
    
    # Example queries to test the agent
    test_queries = [
        "What is the capital of France?",
        "What are the main features of Python 3.10?"
    ]
    
    # Run the agent on each query
    for query in test_queries:
        print(f"\n\n=== Query: {query} ===")
        response = agent.run(query)
        print(f"\nResponse:\n{response}")