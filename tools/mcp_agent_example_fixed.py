#!/usr/bin/env python3
"""
MCP Agent Example (Fixed)

This script demonstrates an end-to-end integration of the MCP SDK with a React agent.
The agent discovers available tools from an MCP server, registers them with the
enhanced tool system, and uses them to answer user queries.

This version uses the fixed MCP SDK wrapper to address Pydantic compatibility issues.
"""
import asyncio
import os
import sys
import logging
from typing import List, Dict, Any, Optional

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.react_agent import ReActAgent
from llm.litellm.litellm import LiteLLM
from log.logging import LogManager
from tools.mcp_sdk_wrapper_fixed import MCPSDKToolProvider, MCPSDKToolWrapper
from tools.tool_selector import ToolSelector, SelectionMethod
from tools.registry import registry
from tools.base import EnhancedTool
from tools.mock_mcp_server import MockMCPServer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class MCPEnhancedReactAgent:
    """React agent with MCP tool integration using the fixed wrapper."""
    
    def __init__(
        self,
        mcp_server_url: str,
        model: str = "anthropic/claude-3-7-sonnet-20250219",
        max_iterations: int = 15,
        mcp_namespace: str = "mcp",
        api_key: Optional[str] = None
    ):
        """Initialize the MCP-enhanced React agent.
        
        Args:
            mcp_server_url: URL of the MCP server
            model: LLM model to use
            max_iterations: Maximum number of ReAct iterations
            mcp_namespace: Namespace for MCP tools
            api_key: Optional API key for MCP server
        """
        self.mcp_server_url = mcp_server_url
        self.model = model
        self.max_iterations = max_iterations
        self.mcp_namespace = mcp_namespace
        self.api_key = api_key
        
        # Initialize components
        self.llm = LiteLLM.create_llm()
        self.log_manager = LogManager()
        self.tool_provider = MCPSDKToolProvider(
            server_url=mcp_server_url,
            namespace=mcp_namespace,
            api_key=api_key
        )
        self.tool_selector = ToolSelector(registry)
        
        # Will be set after tool discovery
        self.mcp_tools = []
        self.agent = None
        self.initialized = False
    
    async def initialize(self):
        """Initialize the agent by discovering and registering MCP tools."""
        if self.initialized:
            return
        
        logger.info(f"Initializing MCP agent with server: {self.mcp_server_url}")
        
        # Clear the registry to avoid conflicts
        registry.clear()
        
        # Discover and register MCP tools
        self.mcp_tools = await self.tool_provider.discover_and_register_tools()
        logger.info(f"Registered {len(self.mcp_tools)} MCP tools")
        
        self.initialized = True
    
    def _select_tools_for_query(self, query: str, max_tools: int = 5) -> List[MCPSDKToolWrapper]:
        """Select appropriate tools based on the user query.
        
        Args:
            query: User's query text
            max_tools: Maximum number of tools to select
            
        Returns:
            List of selected MCPSDKToolWrapper instances
        """
        # Use hybrid selection method for best results
        tools = self.tool_selector.select_tools(
            query=query,
            method=SelectionMethod.HYBRID,
            max_tools=max_tools
        )
        
        # Display selected tools for demonstration
        logger.info(f"Selected {len(tools)} tools for query: '{query}'")
        for i, tool in enumerate(tools):
            logger.info(f"  {i+1}. {tool.metadata.name} ({tool.metadata.description})")
        
        # Convert to MCPSDKToolWrapper instances
        mcp_tools = []
        for tool in tools:
            if isinstance(tool, MCPSDKToolWrapper):
                mcp_tools.append(tool)
        
        return mcp_tools
    
    def _initialize_react_agent(self, query: str, tools: List[MCPSDKToolWrapper]):
        """Initialize the React agent with the selected tools.
        
        Args:
            query: User's query
            tools: Selected tools for this query
        """
        # Define role and task for the agent
        role = "You are a helpful AI assistant that can use tools from an MCP server to answer questions."
        task = f"Answer the user's question: {query}"
        guide = (
            "Think step by step and use the available tools when necessary. "
            "The tools are provided by an MCP server and have specific capabilities. "
            "Always explain your reasoning and the significance of your findings."
        )
        
        # Extract the actual tool instances
        tool_instances = [tool._mcp_tool_instance for tool in tools]
        
        # Create the ReAct agent
        self.agent = ReActAgent(
            llm=self.llm,
            role=role,
            task=task,
            guide=guide,
            tools=tool_instances,
            log_manager=self.log_manager,
            max_iterations=self.max_iterations
        )
    
    async def arun(self, query: str) -> str:
        """Run the agent to answer the query.
        
        Args:
            query: User's query
            
        Returns:
            Agent's response
        """
        # Make sure we're initialized
        if not self.initialized:
            await self.initialize()
        
        # Select tools based on the query
        tools = self._select_tools_for_query(query)
        
        # Check if we have tools
        if not tools:
            return f"I'm sorry, but I don't have any suitable tools to answer your query: {query}"
        
        # Initialize the React agent with selected tools
        self._initialize_react_agent(query, tools)
        
        # Run the agent
        return await self.agent.arun(query, self.model)
    
    def run(self, query: str) -> str:
        """Run the agent synchronously.
        
        Args:
            query: User's query
            
        Returns:
            Agent's response
        """
        return asyncio.run(self.arun(query))
    
    async def close(self):
        """Close the agent and its resources."""
        await self.tool_provider.close()


async def main():
    """Run the MCP agent example."""
    # Start the mock MCP server
    server = MockMCPServer(port=8000)
    server.start()
    
    try:
        # Create and initialize the agent
        agent = MCPEnhancedReactAgent(
            mcp_server_url=server.base_url,
            mcp_namespace="mock_mcp"
        )
        
        # Initialize the agent to discover tools
        await agent.initialize()
        
        # Run a test query
        query = "Calculate 125 * 8 - 32"
        print(f"\n\n{'='*50}")
        print(f"QUERY: {query}")
        print(f"{'='*50}")
        
        # Select tools for the query
        tools = agent._select_tools_for_query(query)
        print(f"\nSelected {len(tools)} tools:")
        for i, tool in enumerate(tools):
            print(f"  {i+1}. {tool.metadata.name}: {tool.metadata.description}")
        
        if tools:
            # Test direct tool execution
            calculator_tool = next(
                (tool for tool in tools if "calculator" in tool.metadata.name.lower()),
                None
            )
            
            if calculator_tool:
                print("\nExecuting calculator tool directly...")
                result = await calculator_tool.arun(expression=query)
                print(f"Result: {result}")
        
        print("\nAgent is now processing the query...")
        response = await agent.arun(query)
        print(f"\nAGENT RESPONSE: {response}")
        
        # Close the agent
        await agent.close()
    
    finally:
        # Stop the server
        server.stop()


if __name__ == "__main__":
    asyncio.run(main()) 