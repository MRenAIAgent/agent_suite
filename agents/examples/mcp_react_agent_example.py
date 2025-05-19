import asyncio
import os
import sys
import json
import logging
from typing import List, Optional, Dict, Any, Callable

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Use ReactAgentFC instead of ReActAgent
from agents.react_agent_fc.react_agent_fc import ReactAgentFC
from llm.litellm.litellm import LiteLLM
from log.logging import LogManager

# Import enhanced tool system components
from tools.registry import registry
from tools.base import EnhancedTool
from tools.tool_types import ToolCategory, ToolCapability, ToolDomain, ToolMetadata, ToolSource

# Use a try/except block to handle missing MCP packages
try:
    # Import MCP client tools
    from agents.mcp_client.github_client import GitHubMCPClient
    from agents.mcp_client.mcp_sdk_client import MCPSDKClient
    from agents.mcp_client.client import MCPServerConfig, MCPTransportType
    from agents.mcp_client.sequential_thinking import SequentialThinkPattern
    HAS_MCP_SDK = True
except ImportError:
    logging.warning("MCP SDK or client modules not installed. The MCP tools will be mocked.")
    HAS_MCP_SDK = False

# Import for tool creation
from pydantic import BaseModel, Field, ConfigDict

# Configure logging
logger = logging.getLogger(__name__)


# Simplified tool wrapper
class SimpleTool:
    """A simple tool wrapper for use with ReactAgentFC."""
    
    def __init__(self, name: str, description: str, function: Callable):
        """
        Initialize a simple tool.
        
        Args:
            name: The name of the tool
            description: A description of what the tool does
            function: The function to call when the tool is invoked
        """
        self.name = name
        self.description = description
        self.function = function
        self.run = function  # Add run method for compatibility
        self.arun = self._async_run  # Add arun method for compatibility
        self.source = ToolSource.CUSTOM  # Add source attribute for registry compatibility
        
        # Add metadata for registry compatibility
        self.metadata = ToolMetadata(
            name=name,
            display_name=name,
            description=description,
            categories=[ToolCategory.UTILITY],
            capabilities=[ToolCapability.READ],
            domains=[ToolDomain.GENERAL],
            keywords=[name.lower()]
        )
        
    async def _async_run(self, **kwargs):
        """Async wrapper for the function."""
        # Use the synchronous function in an async context
        return self.function(**kwargs)
        
    def convert_to_function_call(self) -> Dict:
        """Convert to a function call definition compatible with OpenAI function calling."""
        # Create a schema that works with keyword-based queries
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query"
                        }
                    },
                    "required": ["query"]
                }
            }
        }


class MCPTool(EnhancedTool):
    """Base class for all MCP-based tools."""
    
    # Allow arbitrary types in Pydantic model
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    client: Optional[Any] = None
    tool_name: str = Field(default="", description="Name of the MCP tool to call")
    
    async def arun(self, **kwargs) -> Any:
        """
        Execute the MCP tool.
        
        Args:
            **kwargs: Arguments for the tool
            
        Returns:
            Result of the tool execution
        """
        if not HAS_MCP_SDK:
            logger.warning(f"MCP SDK not installed. Mocking response for {self.tool_name}")
            # Return a more robust mock response that works with the expected format
            return {
                "warning": "This is a mock response because the MCP SDK is not installed.",
                "tool": self.tool_name,
                "args": kwargs,
                # Add these properties to ensure compatibility with various handlers
                "result": f"Mock result for {self.tool_name} with args {str(kwargs)}",
                "content": {"text": f"Mock content for {self.tool_name}"}
            }
            
        if self.client is None:
            raise ValueError("MCP client is not initialized")
            
        if not self.client._initialized:
            try:
                await self.client.connect()
            except Exception as e:
                logger.error(f"Failed to connect to MCP server: {e}")
                return {"error": f"Failed to connect to MCP server: {str(e)}"}
            
        try:
            result = await self.client.call_tool(self.tool_name, kwargs)
            return self._parse_result(result)
        except Exception as e:
            logger.error(f"Error calling MCP tool {self.tool_name}: {e}")
            return {"error": f"Error calling MCP tool {self.tool_name}: {str(e)}"}
    
    def _parse_result(self, result: Any) -> Any:
        """Parse the result from a tool call."""
        # Handle different result types
        if hasattr(result, "content") and hasattr(result.content, "text"):
            # Try to parse JSON
            try:
                return json.loads(result.content.text)
            except json.JSONDecodeError:
                return result.content.text
        # Return the result as is if it doesn't match known formats
        return result


class GitHubSearchRepositories(MCPTool):
    """Search for GitHub repositories."""
    
    query: str = Field(default="", description="The search query for repositories")
    page: int = Field(default=1, description="Page number for pagination")
    per_page: int = Field(default=10, description="Results per page")
    
    async def arun(self, query: str = "", page: int = 1, per_page: int = 10) -> Dict[str, Any]:
        """
        Search for GitHub repositories.
        
        Args:
            query: The search query for repositories
            page: Page number for pagination
            per_page: Results per page
            
        Returns:
            Search results
        """
        if not HAS_MCP_SDK:
            logger.warning("MCP SDK not installed. Returning mock response.")
            
            # Generate a realistic mock response
            is_python = "python" in query.lower() or "language:python" in query.lower()
            is_stars = "stars" in query.lower() or "sort:stars" in query.lower()
            
            mock_repositories = []
            
            if is_python:
                mock_repositories.extend([
                    {
                        "name": "python/cpython", 
                        "full_name": "python/cpython",
                        "description": "The Python programming language", 
                        "stars": 56000,
                        "url": "https://github.com/python/cpython",
                        "language": "Python",
                        "forks": 25000,
                        "created_at": "2017-02-10T12:31:04Z",
                        "updated_at": "2023-04-30T10:15:22Z"
                    },
                    {
                        "name": "vinta/awesome-python",
                        "full_name": "vinta/awesome-python",
                        "description": "A curated list of awesome Python frameworks, libraries, software and resources", 
                        "stars": 170000,
                        "url": "https://github.com/vinta/awesome-python",
                        "language": "Python",
                        "forks": 30000,
                        "created_at": "2014-06-27T21:00:06Z",
                        "updated_at": "2023-05-01T14:23:44Z"
                    },
                    {
                        "name": "django/django",
                        "full_name": "django/django",
                        "description": "The Web framework for perfectionists with deadlines", 
                        "stars": 72000,
                        "url": "https://github.com/django/django",
                        "language": "Python",
                        "forks": 30000,
                        "created_at": "2012-04-28T02:47:18Z",
                        "updated_at": "2023-04-29T22:13:59Z"
                    },
                    {
                        "name": "pallets/flask",
                        "full_name": "pallets/flask",
                        "description": "The Python micro framework for building web applications", 
                        "stars": 64000,
                        "url": "https://github.com/pallets/flask",
                        "language": "Python",
                        "forks": 16000,
                        "created_at": "2010-04-06T11:11:59Z",
                        "updated_at": "2023-05-01T08:44:01Z"
                    },
                    {
                        "name": "tensorflow/tensorflow",
                        "full_name": "tensorflow/tensorflow",
                        "description": "An Open Source Machine Learning Framework for Everyone", 
                        "stars": 174000,
                        "url": "https://github.com/tensorflow/tensorflow",
                        "language": "C++",
                        "forks": 88000,
                        "created_at": "2015-11-07T01:19:20Z",
                        "updated_at": "2023-05-01T17:01:37Z"
                    }
                ])
            else:
                mock_repositories.extend([
                    {
                        "name": "freeCodeCamp/freeCodeCamp",
                        "full_name": "freeCodeCamp/freeCodeCamp",
                        "description": "freeCodeCamp.org's open-source codebase and curriculum", 
                        "stars": 370000,
                        "url": "https://github.com/freeCodeCamp/freeCodeCamp",
                        "language": "TypeScript",
                        "forks": 33000,
                        "created_at": "2014-12-24T17:49:19Z",
                        "updated_at": "2023-05-01T16:41:05Z"
                    },
                    {
                        "name": "EbookFoundation/free-programming-books",
                        "full_name": "EbookFoundation/free-programming-books",
                        "description": "Freely available programming books", 
                        "stars": 280000,
                        "url": "https://github.com/EbookFoundation/free-programming-books",
                        "language": null,
                        "forks": 54000,
                        "created_at": "2013-10-11T06:25:46Z",
                        "updated_at": "2023-05-01T15:21:27Z"
                    }
                ])
            
            # If specifically looking for stars, sort by stars
            if is_stars:
                mock_repositories.sort(key=lambda x: x["stars"], reverse=True)
            
            # Apply pagination
            start_idx = (page - 1) * per_page
            end_idx = start_idx + per_page
            paginated_repos = mock_repositories[start_idx:end_idx]
            
            return {
                "total_count": len(mock_repositories),
                "incomplete_results": False,
                "query": query,
                "page": page,
                "per_page": per_page,
                "repositories": paginated_repos
            }
        
        if self.client is None:
            try:
                self.client = GitHubMCPClient()
                await self.client.connect()
                self.tool_name = "mcp_github_search_repositories"
            except Exception as e:
                logger.error(f"Failed to initialize GitHub client: {e}")
                return {"error": f"Failed to initialize GitHub client: {str(e)}"}
            
        try:
            result = await self.client.call_tool(self.tool_name, {
                "query": query,
                "page": page,
                "per_page": per_page
            })
            return self._parse_result(result)
        except Exception as e:
            logger.error(f"Error searching GitHub repositories: {e}")
            return {
                "error": f"Error searching GitHub repositories: {str(e)}",
                "query": query,
                "repositories": []
            }


class GitHubSearchCode(MCPTool):
    """Search for code across GitHub repositories."""
    
    q: str = Field(default="", description="Search query using GitHub code search syntax")
    
    async def arun(self, q: str = "") -> Dict[str, Any]:
        """
        Search for code across GitHub repositories.
        
        Args:
            q: Search query using GitHub code search syntax
            
        Returns:
            Search results
        """
        if not HAS_MCP_SDK:
            logger.warning("MCP SDK not installed. Returning mock response.")
            return {
                "warning": "This is a mock response because the MCP SDK is not installed.",
                "tool": "github_code_search",
                "query": q,
                "code_items": [
                    {"repo": "django/django", "path": "django/views.py", "content": "def render(request, template_name, context=None):"},
                    {"repo": "flask/flask", "path": "flask/app.py", "content": "def route(self, rule, **options):"}
                ]
            }
            
        if self.client is None:
            try:
                self.client = GitHubMCPClient()
                await self.client.connect()
                self.tool_name = "mcp_github_search_code"
            except Exception as e:
                logger.error(f"Failed to initialize GitHub client: {e}")
                return {"error": f"Failed to initialize GitHub client: {str(e)}"}
            
        try:
            result = await self.client.call_tool(self.tool_name, {
                "q": q
            })
            return self._parse_result(result)
        except Exception as e:
            logger.error(f"Error searching GitHub code: {e}")
            return {
                "error": f"Error searching GitHub code: {str(e)}",
                "query": q,
                "code_items": []
            }


class GitHubSearchIssues(MCPTool):
    """Search for issues in GitHub repositories."""
    
    q: str = Field(default="", description="Search query using GitHub issues search syntax")
    
    async def arun(self, q: str = "") -> Dict[str, Any]:
        """
        Search for issues in GitHub repositories.
        
        Args:
            q: Search query using GitHub issues search syntax
            
        Returns:
            Search results
        """
        if not HAS_MCP_SDK:
            logger.warning("MCP SDK not installed. Returning mock response.")
            return {
                "warning": "This is a mock response because the MCP SDK is not installed.",
                "tool": "github_issue_search",
                "query": q,
                "issues": [
                    {"repo": "django/django", "title": "Add support for async views", "number": 1234, "state": "open"},
                    {"repo": "flask/flask", "title": "Improve documentation for blueprints", "number": 567, "state": "closed"}
                ]
            }
            
        if self.client is None:
            try:
                self.client = GitHubMCPClient()
                await self.client.connect()
                self.tool_name = "mcp_github_search_issues"
            except Exception as e:
                logger.error(f"Failed to initialize GitHub client: {e}")
                return {"error": f"Failed to initialize GitHub client: {str(e)}"}
            
        try:
            result = await self.client.call_tool(self.tool_name, {
                "q": q
            })
            return self._parse_result(result)
        except Exception as e:
            logger.error(f"Error searching GitHub issues: {e}")
            return {
                "error": f"Error searching GitHub issues: {str(e)}",
                "query": q,
                "issues": []
            }


class SequentialThinkingTool(MCPTool):
    """Use sequential thinking to solve complex problems step by step."""
    
    thought: str = Field(default="", description="Your current thinking step")
    thought_number: int = Field(default=1, description="Current thought number")
    total_thoughts: int = Field(default=5, description="Estimated total thoughts needed")
    next_thought_needed: bool = Field(default=True, description="Whether another thought step is needed")
    
    async def arun(self, thought: str = "", thought_number: int = 1, total_thoughts: int = 5, next_thought_needed: bool = True) -> Dict[str, Any]:
        """
        Execute sequential thinking for complex problem solving.
        
        Args:
            thought: Current thinking step
            thought_number: Current thought number
            total_thoughts: Estimated total thoughts needed
            next_thought_needed: Whether another thought step is needed
            
        Returns:
            Result of the sequential thinking step
        """
        if not HAS_MCP_SDK:
            logger.warning("MCP SDK not installed. Returning mock response.")
            next_thought = "Based on the previous thought, I should continue by analyzing the specific features of each framework."
            if thought_number >= total_thoughts:
                next_thought_needed = False
                
            return {
                "warning": "This is a mock response because the MCP SDK is not installed.",
                "tool": "sequential_thinking",
                "thought": thought,
                "thoughtNumber": thought_number,
                "totalThoughts": total_thoughts,
                "nextThoughtNeeded": next_thought_needed,
                "nextThought": next_thought if next_thought_needed else ""
            }
            
        if self.client is None:
            try:
                # Create MCP client with stdio transport
                server_config = MCPServerConfig(
                    name="sequential_thinking",
                    transport_type=MCPTransportType.STDIO,
                    command="python",
                    args=["-m", "mcp_sequential_thinking"]
                )
                self.client = MCPSDKClient(server_config)
                await self.client.connect()
                self.tool_name = "mcp_sequential-thinking_sequentialthinking"
            except Exception as e:
                logger.error(f"Failed to initialize Sequential Thinking client: {e}")
                return {"error": f"Failed to initialize Sequential Thinking client: {str(e)}"}
            
        try:
            result = await self.client.call_tool(self.tool_name, {
                "thought": thought,
                "thoughtNumber": thought_number,
                "totalThoughts": total_thoughts,
                "nextThoughtNeeded": next_thought_needed
            })
            return self._parse_result(result)
        except Exception as e:
            logger.error(f"Error during sequential thinking: {e}")
            return {
                "error": f"Error during sequential thinking: {str(e)}",
                "thought": thought,
                "thoughtNumber": thought_number,
                "totalThoughts": total_thoughts,
                "nextThoughtNeeded": next_thought_needed
            }


# Create synchronous wrapper functions for MCP tools
def github_repo_search(query: str) -> Dict[str, Any]:
    """Search for GitHub repositories."""
    tool = GitHubSearchRepositories()
    return asyncio.run(tool.arun(query=query))

def github_code_search(query: str) -> Dict[str, Any]:
    """Search for code across GitHub repositories."""
    tool = GitHubSearchCode()
    return asyncio.run(tool.arun(q=query))
    
def github_issue_search(query: str) -> Dict[str, Any]:
    """Search for issues in GitHub repositories."""
    tool = GitHubSearchIssues()
    return asyncio.run(tool.arun(q=query))

def sequential_thinking(query: str) -> Dict[str, Any]:
    """Use sequential thinking to solve complex problems step by step."""
    tool = SequentialThinkingTool()
    return asyncio.run(tool.arun(thought=query))


class MCPReactAgentExample:
    """Example implementation of a React agent with MCP tools."""
    
    def __init__(
        self,
        model: str = "gpt-3.5-turbo",
        role: str = "You are a technical researcher who helps developers solve programming problems using a variety of research and analytical tools.",
        task: str = "Help users research and analyze GitHub repositories and programming topics using a range of specialized tools.",
        guide: str = "Use GitHub search tools to find relevant repositories, issues, and code. When facing complex problems, use sequential thinking to break down the analysis into structured steps.",
        max_iterations: int = 20,
        examples: Optional[List[str]] = None
    ):
        """Initialize the MCP React agent.
        
        Args:
            model: Model to use for the agent
            role: Role description for the agent
            task: Task description for the agent
            guide: Guide for the agent's approach
            max_iterations: Maximum number of iterations to run
            examples: Optional list of examples to include in the prompt
        """
        # Setup logging
        self.log_manager = LogManager()
        
        # Initialize LLM
        self.llm = LiteLLM()
        
        # Register tools with the registry
        self._register_tools()
        
        # Create the React agent with a more compatible function calling setup
        try:
            self.agent = ReactAgentFC(
                llm=self.llm,
                role=role,
                task=task,
                guide=guide,
                examples=examples,
                tools=self._get_tools_from_registry(),
                max_iterations=max_iterations
            )
            # Set the log manager on the agent
            self.agent.log_manager = self.log_manager
            self.model = model
        except Exception as e:
            logging.error(f"Failed to initialize ReactAgentFC: {e}")
            raise
    
    def _register_tools(self):
        """Register MCP tools with the tool registry."""
        # Clear registry first to avoid duplicate tools
        registry.clear()
        
        try:
            # Create GitHub search tools
            github_repo_search_tool = GitHubSearchRepositories()
            registry.register_tool(github_repo_search_tool)
            
            github_code_search_tool = GitHubSearchCode()
            registry.register_tool(github_code_search_tool)
            
            github_issue_search_tool = GitHubSearchIssues()
            registry.register_tool(github_issue_search_tool)
            
            sequential_thinking_tool = SequentialThinkingTool()
            registry.register_tool(sequential_thinking_tool)
            
            # Create simplified tools for easier function calling
            simple_tools = [
                SimpleTool(
                    name="github_repo_search",
                    description="Search for GitHub repositories by various criteria",
                    function=github_repo_search
                ),
                SimpleTool(
                    name="github_code_search",
                    description="Search for code across GitHub repositories using GitHub's code search syntax",
                    function=github_code_search
                ),
                SimpleTool(
                    name="github_issue_search",
                    description="Search for issues in GitHub repositories using GitHub's issue search syntax",
                    function=github_issue_search
                ),
                SimpleTool(
                    name="sequential_thinking",
                    description="Break down complex problems into a sequence of reasoning steps",
                    function=sequential_thinking
                )
            ]
            
            # Register simplified tools
            for tool in simple_tools:
                registry.register_tool(tool)
                
        except Exception as e:
            logging.error(f"Error registering tools: {e}")
            raise
            
    def _get_tools_from_registry(self):
        """Get tools from the registry.
        
        Returns:
            List of tools from the registry
        """
        # Convert all registry tools to standard Tool format expected by ReactAgentFC
        tools = []
        for tool in registry.get_all_tools():
            # For SimpleTool objects, they already have the right format
            if isinstance(tool, SimpleTool):
                tools.append(tool)
        
        return tools
    
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
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create agent with MCP tools
    agent = MCPReactAgentExample(max_iterations=30)
    
    # Display the registered tools
    enhanced_tools = agent.get_registered_tools()
    print(f"\nRegistered {len(enhanced_tools)} MCP tools:")
    for tool in enhanced_tools:
        print(f"- {tool.metadata.display_name}: {tool.metadata.description}")
    
    # Example queries to test the agent
    test_queries = [
        "Find popular Python web frameworks on GitHub and analyze their features"
    ]
    
    # Run the agent on each query
    for query in test_queries:
        print(f"\n\n=== Query: {query} ===")
        response = agent.run(query)
        print(f"\nResponse:\n{response}") 