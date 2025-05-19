import asyncio
import os
import sys
from typing import List, Optional

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


class ReactAgentExample:
    """Example implementation of ReactAgent with enhanced tool system."""
    
    def __init__(
        self,
        model: str = "anthropic/claude-3-7-sonnet-20250219",
        role: str = "You are a helpful AI assistant tasked with answering questions precisely and concisely.",
        task: str = "Answer user questions using the tools provided when necessary.",
        guide: str = "Provide direct, accurate answers without unnecessary elaboration.",
        max_iterations: int = 10,
        examples: Optional[List[str]] = None
    ):
        """Initialize the ReactAgentExample with enhanced tool system."""
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
            examples=examples or [],
            tools=self.tools,
            log_manager=LogManager(),
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
        """Get tools from registry for use with the agent."""
        # For now, we still need to return the original tools for ReActAgent
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
        """Run the agent asynchronously."""
        return await self.agent.arun(user_input, self.model)
    
    def run(self, user_input: str) -> str:
        """Run the agent synchronously."""
        return asyncio.run(self.arun(user_input))
    
    def get_registered_tools(self) -> List[EnhancedTool]:
        """Get all enhanced tools registered in the registry."""
        return registry.get_all_tools()


# Example usage
if __name__ == "__main__":
    # Create agent with enhanced tools
    agent = ReactAgentExample(max_iterations=10)
    
    # Display the registered tools
    enhanced_tools = agent.get_registered_tools()
    print(f"\nRegistered {len(enhanced_tools)} enhanced tools:")
    for tool in enhanced_tools:
        print(f"- {tool.metadata.display_name}: {tool.metadata.description}")
    
    # Example queries to test the agent
    test_queries = [
        "What is the current CEO of Tesla?"
    ]
    
    # Run the agent on each query
    for query in test_queries:
        print(f"\n\n=== Query: {query} ===")
        response = agent.run(query)
        print(f"\nFull Response:\n{response}") 