import asyncio
import os
import sys
from typing import List, Optional, Dict

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
        role: str = "You are a seasoned financial analyst specializing in stock market trends and investment strategies. You know the materials and information you need to collect to complete the task.",
        task: str = "Guide users through a comprehensive analysis of Tesla's stock performance, including actionable insights and strategic recommendations.",
        guide: str = "Utilize various analytical tools to gather in-depth information on Tesla's financial health, market trends, news articles, social media sentiment, quarterly and annual reports, like 10-K, 10-Q, etc. Provide a step-by-step breakdown of the analysis process, ensuring clarity and thoroughness in your explanations.",
        max_iterations: int = 20,
        examples: Optional[List[str]] = None,
        debug: bool = False
    ):
        """Initialize the ReactAgentExample with enhanced tool system.
        
        Args:
            model: The LLM model to use
            role: The role of the agent
            task: The task of the agent
            guide: Guidelines for the agent
            max_iterations: Maximum number of iterations for the agent
            examples: Optional examples for the agent
            debug: Enable debug output
        """
        self.llm = LiteLLM.create_llm()
        self.model = model
        self.debug = debug
        
        # Register tools and get the tools for ReActAgent
        self.tools = self._setup_tools()
        
        # Initialize the ReactAgent with tools
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
    
    def _setup_tools(self):
        """Register tools with the registry and get the tools for the agent."""
        # Start fresh with the registry
        registry.clear()
        
        # Reset any static registration state
        for cls in EnhancedTool.__subclasses__():
            if hasattr(cls, '_registered'):
                cls._registered = False
            if hasattr(cls, '_instances'):
                cls._instances.clear()
        
        if self.debug:
            print("=== Setting Up Tools ===")
        
        # 1. Create the original tool instances
        serper_tool = SerperSearchTool(query="")
        wikipedia_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
        youtube_tool = YouTubeSearchTool()
        
        # 2. Register each tool with a unique namespace to avoid collisions
        
        # Google Search
        enhanced_serper = serper_tool.to_enhanced_tool()
        enhanced_serper.metadata.display_name = "Google Search"
        registry.register_tools_with_namespace("search", [enhanced_serper])
        
        # Wikipedia
        adapted_wiki = convert_langchain_tools([wikipedia_tool])[0]
        enhanced_wiki = adapted_wiki.to_enhanced_tool()
        enhanced_wiki.metadata.display_name = "Wikipedia Search"
        registry.register_tools_with_namespace("wiki", [enhanced_wiki])
        
        # YouTube
        adapted_youtube = convert_langchain_tools([youtube_tool])[0]
        enhanced_youtube = adapted_youtube.to_enhanced_tool()
        enhanced_youtube.metadata.display_name = "YouTube Search"
        registry.register_tools_with_namespace("video", [enhanced_youtube])
        
        # Return the tools for the React Agent to use
        return [serper_tool]
    
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


# Example usage
if __name__ == "__main__":
    # Create agent with tools
    agent = ReactAgentExample(max_iterations=10, debug=True)
    
    # Display the tools by namespace
    print("\n=== Available Tools ===")
    for namespace in ["search", "wiki", "video"]:
        tools = registry.get_tools_in_namespace(namespace)
        for tool in tools:
            print(f"- {tool.metadata.display_name}: {tool.metadata.description}")
    
    # Example query to test the agent
    query = "Give me a brief analysis of Tesla's stock performance"
    print(f"\n\n=== Query: {query} ===")
    
    response = agent.run(query)
    print(f"\nResponse:\n{response}")