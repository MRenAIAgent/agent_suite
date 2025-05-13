# MCP Client Implementation

This module provides a client implementation for the Model Context Protocol (MCP), which is a standardized way to connect large language models with tools and data.

## Overview

The MCP client implementation includes:

- Base MCP client interface
- Full implementation using the Python MCP SDK
- Sequential thinking integration for complex reasoning tasks
- Specialized clients for specific MCP servers (GitHub, etc.)
- Tools for agents to interact with MCP servers

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

For the GitHub MCP client, you also need to install:

```bash
pip install mcp-github-cli
```

## Components

### Client Interface

- `MCPClient`: Base abstract class defining the MCP client interface
- `MCPSDKClient`: Concrete implementation using the Python MCP SDK
- `SequentialThinkingMCPClient`: Enhanced client with sequential thinking capabilities
- `GitHubMCPClient`: Specialized client for GitHub operations

### Configuration

- `MCPServerConfig`: Configuration class for connecting to MCP servers
- `MCPTransportType`: Enum of supported transport types (STDIO, SSE, etc.)

### Tools

The following agent tools are provided:

- `MCPListTools`: Tool for listing available tools from an MCP server
- `MCPCallTool`: Tool for calling a tool on an MCP server
- `MCPSequentialThinking`: Tool for solving problems using sequential thinking with an MCP server

## Usage

### Basic Usage

```python
import asyncio
from agents.mcp_client.client import MCPServerConfig, MCPTransportType
from agents.mcp_client.mcp_sdk_client import MCPSDKClient

async def main():
    # Create server configuration
    server_config = MCPServerConfig(
        name="test_server",
        transport_type=MCPTransportType.STDIO,
        command="python",
        args=["-m", "mcp.tools.echo_server"]
    )
    
    # Create and connect client
    client = MCPSDKClient(server_config)
    await client.connect()
    
    # List available tools
    tools = await client.list_tools()
    print(f"Available tools: {tools}")
    
    # Call a tool
    result = await client.call_tool("echo", {"message": "Hello, MCP!"})
    print(f"Result: {result}")
    
    # Clean up
    await client.disconnect()

if __name__ == "__main__":
    asyncio.run(main())
```

### Sequential Thinking

```python
import asyncio
from agents.mcp_client.client import MCPServerConfig, MCPTransportType
from agents.mcp_client.sequential_thinking_client import SequentialThinkingMCPClient

async def main():
    # Create server configuration
    server_config = MCPServerConfig(
        name="test_server",
        transport_type=MCPTransportType.STDIO,
        command="python",
        args=["-m", "mcp.tools.echo_server"]
    )
    
    # Create and connect client
    client = SequentialThinkingMCPClient(server_config)
    await client.connect()
    
    # Solve a complex problem using sequential thinking
    problem = "What are the benefits and drawbacks of implementing a microservices architecture?"
    result = await client.solve_problem_sequentially(problem, max_thoughts=5)
    
    # Print the result
    print(f"Problem: {result['problem']}")
    print(f"Conclusion: {result['conclusion']}")
    
    # Clean up
    await client.disconnect()

if __name__ == "__main__":
    asyncio.run(main())
```

### GitHub Client

```python
import asyncio
from agents.mcp_client.github_client import GitHubMCPClient

async def main():
    # Create and connect GitHub client
    client = GitHubMCPClient()
    await client.connect()
    
    # Search for repositories
    repos = await client.search_repositories("language:python stars:>1000", limit=5)
    print(f"Top Python repositories: {repos}")
    
    # Get repository info
    repo_info = await client.get_repository_info("modelcontextprotocol/python-sdk")
    print(f"Repository info: {repo_info}")
    
    # Get repository contents
    contents = await client.get_repository_contents("modelcontextprotocol/python-sdk", "README.md")
    print(f"README content: {contents}")
    
    # List branches
    branches = await client.list_repository_branches("modelcontextprotocol/python-sdk")
    print(f"Branches: {branches}")
    
    # Get open issues
    issues = await client.get_issues("modelcontextprotocol/python-sdk", limit=5, state="OPEN")
    print(f"Open issues: {issues}")
    
    # Clean up
    await client.disconnect()

if __name__ == "__main__":
    asyncio.run(main())
```

## Testing

Run the test scripts to verify the MCP client functionality:

```bash
# Test the base MCP client
python -m agents.mcp_client.test_mcp_client

# Test the GitHub MCP client
python -m agents.mcp_client.test_github_client
```

## Integration with Agents

To integrate MCP capabilities with an agent:

```python
from agents.agent import Agent
from tools.mcp_tool import MCPListTools, MCPCallTool, MCPSequentialThinking

# Create an agent
agent = Agent(...)

# Add MCP tools
agent.add_tool(MCPListTools())
agent.add_tool(MCPCallTool())
agent.add_tool(MCPSequentialThinking())

# Use the agent
response = agent.run("What tools are available on the GitHub MCP server?", "gpt-4")
```

## Do We Need to Implement a Client for Each MCP Server?

No, you don't need to implement a custom client for each MCP server. The architecture is designed with three levels of client abstraction:

1. **Base MCP Client (`MCPClient`)**: The abstract base class that defines the standard interface for all MCP clients. This provides methods for connecting to servers, listing tools/resources, and calling tools.

2. **SDK Implementation (`MCPSDKClient`)**: A concrete implementation of the base client that uses the MCP SDK. This is a general-purpose client that can connect to and interact with any MCP server.

3. **Specialized Clients**: Optional domain-specific clients like `GitHubMCPClient` that extend the SDK client with helper methods for specific use cases.

For most MCP servers, using the general `MCPSDKClient` is sufficient. You need to implement a specialized client only when:

1. You want to provide a more user-friendly API for specific domains
2. You need to optimize performance for specific patterns of usage
3. You want to hide complexity from users for common operations

The `GitHubMCPClient` is an example of a specialized client that provides convenient methods for GitHub operations, but under the hood, it uses the same SDK client to communicate with the server.

This tiered approach gives you flexibility to choose the right level of abstraction for your needs. 