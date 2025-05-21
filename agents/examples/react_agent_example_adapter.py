import asyncio
import os
import sys
from typing import List, Optional
from datetime import datetime

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from agents.react_agent import ReActAgent
from llm.litellm.litellm import LiteLLM
from log.logging import LogManager

# Import enhanced tool system components
from tools.registry import registry
from tools.base import EnhancedTool
from tools.tool_types import ToolCategory, ToolCapability, ToolDomain, ToolMetadata, ToolSource

# Import tools that will be registered
from tools.serper_api import SerperSearchTool
from langchain_community.tools import WikipediaQueryRun, YouTubeSearchTool
from langchain_community.utilities import WikipediaAPIWrapper
from tools.adapter.langchain_tool import convert_langchain_tools

# Import new memory system components
from agent_suite.memory.context.in_memory import InMemoryContextMemory
from agent_suite.memory.factory import MemoryFactory
from agent_suite.memory.adapter import ContextMemoryAdapter
from agents.memory_manager import MemoryManager as OldMemoryManager


class ReactAgentExampleWithNewMemory:
    """Example implementation of ReactAgent with enhanced tool system and new memory system."""
    
    def __init__(
        self,
        model: str = "anthropic/claude-3-7-sonnet-20250219",
        role: str = "You are a seasoned financial analyst specializing in stock market trends and investment strategies. You know the materials and information you need to collect to complete the task.",
        task: str = "Guide users through a comprehensive analysis of Tesla's stock performance, including actionable insights and strategic recommendations.",
        guide: str = "Utilize various analytical tools to gather in-depth information on Tesla's financial health, market trends, news articles, social media sentiment, quarterly and annual reports, like 10-K, 10-Q, etc. Provide a step-by-step breakdown of the analysis process, ensuring clarity and thoroughness in your explanations.",
        max_iterations: int = 20,
        examples: Optional[List[str]] = None
    ):
        """Initialize the ReactAgentExample with enhanced tool system and new memory.
        
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
        
        # Create a new memory system
        self.context_memory = InMemoryContextMemory()
        
        # Add some initial context to demonstrate it works
        self.context_memory.add("system_message", 
                              "You are a financial analyst assistant using the new memory system.", 
                              {"timestamp": datetime.now().isoformat()})
        
        # Create an adapter to convert the new context memory to the old IntermediateStorage format
        adapted_storage = ContextMemoryAdapter.adapt_context_to_intermediate(self.context_memory)
        
        # Create the old memory manager with our adapted storage
        memory_manager = OldCustomMemoryManager(adapted_storage)
        
        # Initialize the ReactAgent with tools from registry and adapted memory
        self.agent = ReActAgent(
            llm=self.llm,
            role=role,
            task=task,
            guide=guide,
            examples=examples or [],
            tools=self.tools,
            log_manager=LogManager(),
            memory_manager=memory_manager,  # Using the adapted memory
            max_iterations=max_iterations
        )
    
    def _register_tools(self):
        """Register all enhanced tools with the registry."""
        # Register SerperSearchTool as enhanced tool
        serper_tool = SerperSearchTool(query="")
        enhanced_serper = serper_tool.to_enhanced_tool()
        registry.register_tool(enhanced_serper)
        
        # Initialize and register LangChain tools
        # Convert LangChain tools to our format first
        wikipedia_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
        youtube_tool = YouTubeSearchTool()
        langchain_tools = convert_langchain_tools([wikipedia_tool, youtube_tool])
        
        # Register converted LangChain tools as enhanced tools
        for tool in langchain_tools:
            enhanced_tool = tool.to_enhanced_tool()
            registry.register_tool(enhanced_tool)
        
        print(f"Registered {len(registry.get_all_tools())} tools with the registry")
    
    def _get_tools_from_registry(self):
        """Get tools from registry for use with the agent.
        
        Returns:
            List of tools for the agent
        """
        # For now, we still need to return the original tools for ReActAgent
        # This would ideally be improved to directly use enhanced tools
        enhanced_tools = registry.get_all_tools()
        original_tools = []
        
        for tool in enhanced_tools:
            # Create original tool version
            if tool.metadata.name == "search":
                original_tools.append(SerperSearchTool(query=""))
            # Add other tool conversions here as needed
        
        # For testing purposes, if we don't have any tools, add SerperSearchTool
        if not original_tools:
            original_tools.append(SerperSearchTool(query=""))
            
        return original_tools
    
    async def arun(self, user_input: str) -> str:
        """Run the agent asynchronously.
        
        Args:
            user_input: The user's input
            
        Returns:
            The agent's response
        """
        # Add the user input to our new context memory
        self.context_memory.add("user_query", user_input, 
                              {"timestamp": datetime.now().isoformat()})
        
        # Run the agent with the adapted memory
        response = await self.agent.arun(user_input, self.model)
        
        # Print the contents of the new context memory to verify it's being used
        print("\n=== New Memory System Contents ===")
        print(self.context_memory.get_formatted())
        
        return response
    
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


class OldCustomMemoryManager(OldMemoryManager):
    """Custom memory manager that uses our provided storage."""
    
    def __init__(self, storage):
        """Initialize with the provided storage."""
        super().__init__()
        self.store = storage  # Override the default store with our adapted storage
    

# Example usage
if __name__ == "__main__":
    print("=== Testing React Agent with New Memory System Using Adapter ===\n")
    
    # Create agent with enhanced tools and new memory system
    agent = ReactAgentExampleWithNewMemory(max_iterations=10)
    
    # Display the registered tools
    enhanced_tools = agent.get_registered_tools()
    print(f"\nRegistered {len(enhanced_tools)} enhanced tools:")
    for tool in enhanced_tools:
        print(f"- {tool.metadata.display_name}: {tool.metadata.description}")
    
    # Example queries to test the agent
    test_queries = [
        "Quick overview of Tesla's recent performance"
    ]
    
    # Run the agent on each query
    for query in test_queries:
        print(f"\n\n=== Query: {query} ===")
        response = agent.run(query)
        print(f"\nResponse:\n{response}") 