# MCP (Model Completion Protocol) Integration

This document explains the different approaches we've implemented for integrating with MCP servers.

## What is MCP?

MCP (Model Completion Protocol) is a protocol for standardizing tool usage across different AI models and providers. It allows tools to be discovered and executed in a consistent way, regardless of the underlying implementation.

## Integration Approaches

We've implemented multiple approaches to MCP integration:

### 1. Enhanced Tool System Integration (`mcp_wrapper.py`)

This approach integrates MCP tools with our enhanced tool system, allowing them to be discovered, registered, and selected alongside other tools. It uses the `MCPToolWrapper` class to adapt MCP tools to the `EnhancedTool` interface.

**Note**: This approach may have issues with Pydantic compatibility depending on your environment.

### 2. Direct MCP Integration (`mcp_direct.py`)

This is a simpler, more direct approach that bypasses the enhanced tool system entirely. It provides a standalone client for connecting to MCP servers, discovering tools, and executing them. This approach is more reliable when you encounter Pydantic compatibility issues.

### 3. Fixed SDK Integration (`mcp_sdk_wrapper_fixed.py`)

This is a fixed version of the enhanced tool system integration that addresses certain Pydantic compatibility issues by using a different inheritance approach.

## Example Implementations

1. **Mock MCP Server** (`mock_mcp_server.py`): A simple mock MCP server for testing purposes.

2. **Direct MCP Agent** (`mcp_direct.py`): A basic agent that connects to an MCP server and uses tools directly.

3. **MCP LLM Agent** (`mcp_llm_agent_fixed.py`): A more advanced agent that uses an LLM to select and use MCP tools based on user queries.

## How to Use

### Running the Mock Server

```python
from tools.mock_mcp_server import MockMCPServer

# Start a mock server on port 8000
server = MockMCPServer(port=8000)
server.start()

# Do something with the server...

# Stop the server when done
server.stop()
```

### Direct MCP Integration

```python
from tools.mcp_direct import MCPToolProvider

# Create a provider that connects to an MCP server
provider = MCPToolProvider(server_url="http://localhost:8000")

# Initialize the provider (discovers tools)
await provider.initialize()

# Get tools for a specific query
tools = provider.select_tools_for_query("Calculate 125 * 8")

# Execute a specific tool
calculator = provider.get_tool("calculator_compute")
result = await calculator.arun(expression="125 * 8")
print(result)  # {'result': 1000}

# Close the provider when done
await provider.close()
```

### MCP LLM Agent

```python
from tools.mcp_llm_agent_fixed import MCPLLMAgent

# Create an agent that uses an LLM to interact with MCP tools
agent = MCPLLMAgent(
    mcp_server_url="http://localhost:8000",
    model="anthropic/claude-3-7-sonnet-20250219"
)

# Initialize the agent
await agent.initialize()

# Answer a query using the appropriate MCP tools
answer = await agent.answer_query("What's the weather like in Seattle?")
print(answer)

# Close the agent when done
await agent.close()
```

## Core Components

1. **MCPClient**: A client for connecting to MCP servers, discovering tools, and executing them.

2. **MCPTool**: A wrapper around an MCP tool that can be executed directly.

3. **MCPToolProvider**: A provider that manages a collection of MCP tools from a server.

4. **MCPLLMAgent**: An agent that uses an LLM to interact with MCP tools.

## Troubleshooting

If you encounter Pydantic compatibility issues with the enhanced tool system integration (`mcp_wrapper.py`), try using the direct integration (`mcp_direct.py`) instead.

Common errors include:
- `"MCPToolWrapper" object has no attribute "__pydantic_private__"`
- `ValueError: "MCPToolWrapper" object has no field "mcp_tool_name"`

These are typically caused by version mismatches between Pydantic versions. 