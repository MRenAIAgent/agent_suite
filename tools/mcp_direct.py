"""
MCP Direct Integration

This module provides a direct integration with MCP (Model Completion Protocol) servers,
bypassing the enhanced tool system entirely to avoid Pydantic compatibility issues.
"""
import asyncio
import aiohttp
import json
import logging
from typing import Dict, Any, List, Optional, Union, Set

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MCPClient:
    """Client for connecting to MCP servers."""
    
    def __init__(self, base_url: str, api_key: Optional[str] = None):
        """Initialize the MCP client.
        
        Args:
            base_url: The base URL of the MCP server
            api_key: Optional API key for authentication
        """
        self.base_url = base_url
        self.api_key = api_key
        self.session = None
        self.tools = {}
        self._initialized = False
    
    async def initialize(self):
        """Initialize the client and discover tools."""
        if self._initialized:
            return
        
        # Create session if needed
        if self.session is None:
            self.session = aiohttp.ClientSession()
        
        # Discover tools
        await self.discover_tools()
        self._initialized = True
    
    async def discover_tools(self) -> List[Dict[str, Any]]:
        """Discover tools from the MCP server."""
        if self.session is None:
            self.session = aiohttp.ClientSession()
        
        tools_url = f"{self.base_url}/tools"
        headers = {}
        
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        try:
            async with self.session.get(tools_url, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    tools = data.get("tools", [])
                    
                    # Cache tools by name
                    for tool in tools:
                        self.tools[tool.get("name")] = tool
                    
                    logger.info(f"Discovered {len(tools)} tools from MCP server")
                    return tools
                else:
                    error_text = await response.text()
                    logger.error(f"Failed to discover tools: HTTP {response.status} - {error_text}")
                    return []
        except Exception as e:
            logger.error(f"Error discovering tools: {e}")
            return []
    
    async def execute_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool on the MCP server.
        
        Args:
            tool_name: Name of the tool to execute
            parameters: Parameters for the tool
            
        Returns:
            Results from the tool execution
        """
        # Initialize if needed
        if not self._initialized:
            await self.initialize()
        
        # Check if the tool exists
        if tool_name not in self.tools:
            raise ValueError(f"Tool not found: {tool_name}")
        
        # Prepare request
        payload = {
            "name": tool_name,
            "parameters": parameters
        }
        
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        # Execute the tool
        try:
            execute_url = f"{self.base_url}/execute"
            
            async with self.session.post(
                execute_url,
                headers=headers,
                json=payload
            ) as response:
                response_data = await response.json()
                
                if response.status == 200:
                    result = response_data.get("result", {})
                    logger.info(f"Successfully executed tool {tool_name}")
                    return result
                else:
                    error = response_data.get("error", f"HTTP Error: {response.status}")
                    logger.error(f"Error executing tool {tool_name}: {error}")
                    raise RuntimeError(f"Error executing tool: {error}")
        except Exception as e:
            logger.error(f"Error executing tool {tool_name}: {e}")
            raise RuntimeError(f"Error executing tool {tool_name}: {str(e)}")
    
    def get_tool_data(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """Get data for a specific tool."""
        return self.tools.get(tool_name)
    
    def get_all_tools(self) -> List[Dict[str, Any]]:
        """Get list of all available tools."""
        return list(self.tools.values())
    
    async def close(self):
        """Close the client session."""
        if self.session:
            await self.session.close()
            self.session = None


class MCPTool:
    """Wrapper for an MCP tool that can be executed directly."""
    
    def __init__(self, client: MCPClient, tool_name: str, tool_data: Dict[str, Any]):
        """Initialize the MCP tool.
        
        Args:
            client: The MCP client
            tool_name: Name of the tool
            tool_data: Tool data from the MCP server
        """
        self.client = client
        self.name = tool_name
        self.data = tool_data
        self.description = tool_data.get("description", "")
        self.parameters = tool_data.get("parameters", {})
    
    async def arun(self, **kwargs) -> Dict[str, Any]:
        """Execute the tool asynchronously.
        
        Args:
            **kwargs: Parameters for the tool
            
        Returns:
            Results from the tool execution
        """
        try:
            return await self.client.execute_tool(self.name, kwargs)
        except Exception as e:
            logger.error(f"Error executing tool {self.name}: {e}")
            return {"error": str(e)}
    
    def run(self, **kwargs) -> Dict[str, Any]:
        """Execute the tool synchronously.
        
        Args:
            **kwargs: Parameters for the tool
            
        Returns:
            Results from the tool execution
        """
        return asyncio.run(self.arun(**kwargs))
    
    def get_description(self) -> str:
        """Get the tool description."""
        return self.description
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get the tool parameters schema."""
        return self.parameters
    
    def __str__(self) -> str:
        """String representation of the tool."""
        return f"{self.name}: {self.description}"


class MCPToolProvider:
    """Provider for MCP tools from a server."""
    
    def __init__(self, server_url: str, api_key: Optional[str] = None):
        """Initialize the MCP tool provider.
        
        Args:
            server_url: URL of the MCP server
            api_key: Optional API key for authentication
        """
        self.client = MCPClient(base_url=server_url, api_key=api_key)
        self.tools = {}
    
    async def initialize(self):
        """Initialize the provider and discover tools."""
        await self.client.initialize()
        await self.discover_tools()
    
    async def discover_tools(self) -> Dict[str, MCPTool]:
        """Discover and create tools from the MCP server.
        
        Returns:
            Dictionary of tool name to MCPTool
        """
        # Make sure client is initialized
        if not self.client._initialized:
            await self.client.initialize()
        
        # Get tool data
        tool_data_list = self.client.get_all_tools()
        
        # Create tool instances
        for tool_data in tool_data_list:
            tool_name = tool_data.get("name", "")
            if not tool_name:
                continue
                
            tool = MCPTool(
                client=self.client,
                tool_name=tool_name,
                tool_data=tool_data
            )
            
            self.tools[tool_name] = tool
        
        logger.info(f"Created {len(self.tools)} MCP tools")
        return self.tools
    
    def get_tool(self, tool_name: str) -> Optional[MCPTool]:
        """Get a specific tool by name.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            The tool instance or None if not found
        """
        return self.tools.get(tool_name)
    
    def get_all_tools(self) -> List[MCPTool]:
        """Get all available tools.
        
        Returns:
            List of MCPTool instances
        """
        return list(self.tools.values())
    
    def select_tools_for_query(self, query: str) -> List[MCPTool]:
        """Select appropriate tools for a query using basic keyword matching.
        
        Args:
            query: The user query
            
        Returns:
            List of relevant tools
        """
        query_lower = query.lower()
        
        # Simple scoring function for tools
        def score_tool(tool: MCPTool) -> float:
            score = 0.0
            name = tool.name.lower()
            desc = tool.description.lower()
            
            # Check for calculator
            if ("calculate" in query_lower or "compute" in query_lower or 
                    any(op in query_lower for op in ["+", "-", "*", "/"])):
                if "calculator" in name or "compute" in name:
                    score += 5.0
            
            # Check for weather
            if "weather" in query_lower:
                if "weather" in name or "forecast" in name:
                    score += 5.0
            
            # Check for stocks
            if "stock" in query_lower or "price" in query_lower:
                if "stock" in name or "price" in name:
                    score += 5.0
            
            # Check for web search
            if "search" in query_lower or "information" in query_lower:
                if "search" in name or "web" in name:
                    score += 5.0
            
            # General name/description match bonus
            for word in query_lower.split():
                if len(word) > 3:
                    if word in name:
                        score += 2.0
                    if word in desc:
                        score += 1.0
            
            return score
        
        # Score all tools and select the most relevant ones
        scored_tools = [(tool, score_tool(tool)) for tool in self.tools.values()]
        relevant_tools = [tool for tool, score in sorted(scored_tools, key=lambda x: x[1], reverse=True) if score > 0]
        
        # If no tools matched, return a default selection
        if not relevant_tools and self.tools:
            # Get the first tool that matches any of these categories
            for category in ["calculator", "search", "weather", "stock"]:
                for tool in self.tools.values():
                    if category in tool.name.lower():
                        relevant_tools.append(tool)
                        break
                if relevant_tools:
                    break
            
            # Still no matches, return the first tool
            if not relevant_tools:
                relevant_tools = [next(iter(self.tools.values()))]
        
        return relevant_tools[:3]  # Limit to 3 tools
    
    async def close(self):
        """Close the provider and its client."""
        await self.client.close()


class MCPReactAgent:
    """A direct React agent for MCP tools."""
    
    def __init__(self, server_url: str, api_key: Optional[str] = None):
        """Initialize the direct React agent.
        
        Args:
            server_url: URL of the MCP server
            api_key: Optional API key for authentication
        """
        self.provider = MCPToolProvider(server_url=server_url, api_key=api_key)
        self.initialized = False
    
    async def initialize(self):
        """Initialize the agent and discover tools."""
        if not self.initialized:
            await self.provider.initialize()
            self.initialized = True
    
    async def answer_query(self, query: str) -> str:
        """Answer a user query using appropriate MCP tools.
        
        Args:
            query: The user's query
            
        Returns:
            Response to the query
        """
        # Initialize if needed
        if not self.initialized:
            await self.initialize()
        
        # Select tools for the query
        tools = self.provider.select_tools_for_query(query)
        
        if not tools:
            return f"I'm sorry, but I don't have any suitable tools to answer your query: {query}"
        
        # For this demo, just execute the first relevant tool
        tool = tools[0]
        logger.info(f"Using tool: {tool.name} to answer query: {query}")
        
        try:
            # Handle different tool types
            if "calculator" in tool.name.lower() and any(op in query for op in ["+", "-", "*", "/"]):
                # Extract expression from query
                expression = query
                for prefix in ["calculate", "compute", "what is", "what's"]:
                    if prefix in query.lower():
                        expression = query.lower().split(prefix, 1)[1].strip()
                        break
                
                result = await tool.arun(expression=expression)
                answer = f"The result of '{expression}' is {result.get('result', 'unknown')}"
            
            elif "weather" in tool.name.lower():
                # Extract location from query
                location = "San Francisco"  # Default
                if "in" in query and "weather" in query:
                    parts = query.split("in")
                    if len(parts) > 1:
                        location = parts[1].strip().rstrip("?.,!")
                
                result = await tool.arun(location=location)
                answer = f"Weather information: {result.get('forecast', 'No weather data available')}"
            
            elif "stock" in tool.name.lower():
                # Extract symbol from query
                symbol = "AAPL"  # Default
                if "of" in query and "stock" in query:
                    parts = query.split("of")
                    if len(parts) > 1:
                        symbol_part = parts[1].strip().rstrip("?.,!")
                        # Extract ticker symbol (likely in all caps)
                        words = symbol_part.split()
                        for word in words:
                            if word.isupper() and len(word) < 5:
                                symbol = word
                                break
                
                result = await tool.arun(symbol=symbol)
                price = result.get("price", "unknown")
                change = result.get("change", "")
                answer = f"The stock price for {symbol} is ${price} ({change})"
            
            elif "search" in tool.name.lower():
                # Use whole query as search term
                search_query = query
                for prefix in ["search for", "search", "find", "look up"]:
                    if prefix in query.lower():
                        search_query = query.lower().split(prefix, 1)[1].strip()
                        break
                
                result = await tool.arun(query=search_query)
                answer = "Search results:\n"
                for idx, r in enumerate(result.get("results", [])[:3]):
                    answer += f"{idx+1}. {r.get('title', '')}: {r.get('snippet', '')}\n"
            
            else:
                # Generic tool execution
                result = await tool.arun()
                answer = f"Tool result: {json.dumps(result, indent=2)}"
            
            return answer
            
        except Exception as e:
            logger.error(f"Error executing tool {tool.name}: {e}")
            return f"I encountered an error while using the {tool.name} tool: {str(e)}"
    
    async def close(self):
        """Close the agent and release resources."""
        await self.provider.close()


async def test_mcp_direct():
    """Test the direct MCP integration."""
    # Import the mock server
    from tools.mock_mcp_server import MockMCPServer
    
    # Start a mock server
    server = MockMCPServer(port=8000)
    server.start()
    
    try:
        print("\n=== Testing Direct MCP Integration ===")
        
        # Create agent
        agent = MCPReactAgent(server_url=server.base_url)
        
        # Initialize
        await agent.initialize()
        
        # Test queries
        test_queries = [
            "Calculate 125 * 8 - 32",
            "What's the weather like in Seattle?",
            "What's the current stock price of AAPL?",
            "Search for information about artificial intelligence"
        ]
        
        for query in test_queries:
            print(f"\nQuery: {query}")
            
            # Get answer
            answer = await agent.answer_query(query)
            
            print(f"Answer: {answer}")
        
        # Close agent
        await agent.close()
    
    finally:
        server.stop()


if __name__ == "__main__":
    asyncio.run(test_mcp_direct()) 