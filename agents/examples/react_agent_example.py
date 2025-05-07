import asyncio
import os
import sys
from typing import List, Optional

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from agents.react_agent import ReActAgent
from llm.litellm.litellm import LiteLLM
from log.logging import LogManager
from tools.adapter.langchain_tool import convert_langchain_tools
from tools.serper_api import SerperSearchTool

# Import LangChain tools
from langchain_community.tools import WikipediaQueryRun, YouTubeSearchTool
from langchain_community.utilities import WikipediaAPIWrapper


class ReactAgentExample:
    """Example implementation of ReactAgent with multiple tools."""
    
    def __init__(
        self,
        # model: str = "openai/gpt-4o",
        # model: str = "anthropic/claude-3-5-sonnet-20240620",
        model: str = "anthropic/claude-3-7-sonnet-20250219",
        role: str = "You are a seasoned financial analyst specializing in stock market trends and investment strategies. You know the materials and information you need to collect to complete the task.",
        task: str = "Guide users through a comprehensive analysis of Tesla's stock performance, including actionable insights and strategic recommendations.",
        guide: str = "Utilize various analytical tools to gather in-depth information on Tesla's financial health, market trends, news articles, social media sentiment, quarterly and annual reports, like 10-K, 10-Q, etc. Provide a step-by-step breakdown of the analysis process, ensuring clarity and thoroughness in your explanations.",
        max_iterations: int = 20,
        examples: Optional[List[str]] = None
    ):
        """Initialize the ReactAgentExample.
        
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
        
        # Initialize tools
        self.tools = self._initialize_tools()
        
        # Initialize the ReactAgent
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
    
    def _initialize_tools(self):
        """Initialize the tools for the agent.
        
        Returns:
            List of tools for the agent
        """
        # Initialize SerperSearchTool
        serper_tool = SerperSearchTool(query="")
        
        # Initialize LangChain tools
        wikipedia_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
        youtube_tool = YouTubeSearchTool()
        
        # Convert LangChain tools to our format

        langchain_tools = convert_langchain_tools([wikipedia_tool, youtube_tool])
        
        # Combine all tools
        # return [serper_tool] + langchain_tools
        return [serper_tool]
    
    async def arun(self, user_input: str, model_override: Optional[str] = None) -> str:
        """Run the agent asynchronously.
        
        Args:
            user_input: The user's input
            model_override: Optional model override (ignored, using predefined model)
            
        Returns:
            The agent's response
        """
        # We ignore model_override and use self.model since the agent is already configured
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
    agent = ReactAgentExample(max_iterations=30)
    
    # Example queries to test the agent
    test_queries = [
        # "Generate comprehensive report for the most popular agentic agents patterns from 2023 to 2025.",
        # "estimate the world cup winner in 2026"
        "Comprehensive report on tesla stock prediction in 2025"
    ]
    
    # Run the agent on each query
    for query in test_queries:
        print(f"\n\n=== Query: {query} ===")
        response = agent.run(query)
        print(f"\nResponse:\n{response}")