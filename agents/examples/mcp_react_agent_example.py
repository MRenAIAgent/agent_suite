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
        
    async def _async_run(self, **kwargs):
        """Async wrapper for the function."""
        # Use the synchronous function in an async context
        return self.function(**kwargs)
        
    def convert_to_function_call(self) -> Dict:
        """Convert to a function call definition compatible with OpenAI function calling."""
        # Simplified schema for basic tools
        return {
            "type": "function",
            "function": {
                "name": self.name.lower(),
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
            return {
                "warning": "This is a mock response because the MCP SDK is not installed.",
                "tool": self.tool_name,
                "args": kwargs
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
            return {
                "warning": "This is a mock response because the MCP SDK is not installed.",
                "tool": "github_repo_search",
                "query": query,
                "page": page,
                "per_page": per_page,
                "repositories": [
                    {"name": "django", "description": "The Web framework for perfectionists with deadlines.", "stars": 69000},
                    {"name": "flask", "description": "The Python micro framework for building web applications.", "stars": 62000},
                    {"name": "fastapi", "description": "FastAPI framework, high performance, easy to learn, fast to code, ready for production", "stars": 59000}
                ]
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
    """Example implementation of ReactAgent with MCP tool system."""
    
    def __init__(
        self,
        model: str = "anthropic/claude-3-7-sonnet-20250219",
        role: str = "You are a technical researcher who helps developers solve programming problems using a variety of research and analytical tools.",
        task: str = "Help users research and analyze GitHub repositories and programming topics using a range of specialized tools.",
        guide: str = "Use GitHub search tools to find relevant repositories, issues, and code. When facing complex problems, use sequential thinking to break down the analysis into structured steps.",
        max_iterations: int = 20,
        examples: Optional[List[str]] = None
    ):
        """Initialize the MCPReactAgentExample with MCP tools.
        
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
        self.log_manager = LogManager()
        
        # Clear and initialize the tool registry
        registry.clear()
        
        # Register MCP tools
        self._register_tools()
        
        # Create simple tool wrappers for function calling
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
        
        # Initialize the ReactAgentFC with tools
        self.agent = ReactAgentFC(
            llm=self.llm,
            role=role,
            task=task,
            guide=guide,
            examples=examples or [],
            tools=simple_tools,
            max_iterations=max_iterations
        )
        
        # Add log manager
        self.agent.log_manager = self.log_manager
    
    def _register_tools(self):
        """Register all MCP tools with the registry."""
        # Register GitHub search tools
        github_repo_search = GitHubSearchRepositories(
            metadata=ToolMetadata(
                name="github_repo_search",
                display_name="GitHub Repository Search",
                description="Search for GitHub repositories by various criteria",
                categories=[ToolCategory.SEARCH, ToolCategory.UTILITY],
                domains=[ToolDomain.PROGRAMMING, ToolDomain.GENERAL],
                capabilities=[ToolCapability.SEARCH, ToolCapability.FETCH]
            ),
            source=ToolSource.MCP
        )
        registry.register_tool(github_repo_search)
        
        github_code_search = GitHubSearchCode(
            metadata=ToolMetadata(
                name="github_code_search",
                display_name="GitHub Code Search",
                description="Search for code across GitHub repositories using GitHub's code search syntax",
                categories=[ToolCategory.SEARCH, ToolCategory.CODING],
                domains=[ToolDomain.PROGRAMMING, ToolDomain.GENERAL],
                capabilities=[ToolCapability.SEARCH, ToolCapability.FETCH]
            ),
            source=ToolSource.MCP
        )
        registry.register_tool(github_code_search)
        
        github_issue_search = GitHubSearchIssues(
            metadata=ToolMetadata(
                name="github_issue_search",
                display_name="GitHub Issue Search",
                description="Search for issues across GitHub repositories using GitHub's issue search syntax",
                categories=[ToolCategory.SEARCH, ToolCategory.UTILITY],
                domains=[ToolDomain.PROGRAMMING, ToolDomain.GENERAL],
                capabilities=[ToolCapability.SEARCH, ToolCapability.FETCH]
            ),
            source=ToolSource.MCP
        )
        registry.register_tool(github_issue_search)
        
        # Register Sequential Thinking tool
        sequential_thinking = SequentialThinkingTool(
            metadata=ToolMetadata(
                name="sequential_thinking",
                display_name="Sequential Thinking",
                description="Break down complex problems into a sequence of reasoning steps",
                categories=[ToolCategory.UTILITY, ToolCategory.CUSTOM],
                domains=[ToolDomain.GENERAL, ToolDomain.EDUCATION],
                capabilities=[ToolCapability.ANALYZE, ToolCapability.COMPUTE]
            ),
            source=ToolSource.MCP
        )
        registry.register_tool(sequential_thinking)
        
        print(f"Registered {len(registry.get_all_tools())} MCP tools with the registry")
    
    def _get_tools_from_registry(self):
        """Get tools from registry for use with the agent.
        
        Returns:
            List of tools for the agent
        """
        enhanced_tools = registry.get_all_tools()
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