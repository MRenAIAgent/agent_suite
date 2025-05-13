# MCP Integration Example

This example demonstrates integrating the agent_suite with an MCP (Model Completion Protocol) server.
The example shows how to:

1. Connect to an MCP server
2. Discover available tools
3. Register tools with the enhanced tool system
4. Use tools with a ReAct agent to answer queries

## Components

- `mock_mcp_server.py`: Simulates an MCP server with tools
- `mcp_sdk_client.py`: Client for connecting to MCP servers
- `mcp_sdk_wrapper.py`: Integrates MCP tools with the enhanced tool system
- `mcp_agent_example.py`: End-to-end example of an agent using MCP tools

## Running the Example

1. Install the required dependencies:

```bash
pip install -r mcp_example_requirements.txt
```

2. Run the MCP agent example:

```bash
python tools/mcp_agent_example.py
```

This will:
- Start a mock MCP server
- Create an agent that connects to the server
- Discover available tools
- Run test queries that demonstrate different tools

## Connecting to a Real MCP Server

To connect to a real MCP server instead of the mock server:

1. Modify the `mcp_agent_example.py` file:

```python
# Comment out the mock server
# server = MockMCPServer(port=8000)
# server.start()

# Connect to your real MCP server
agent = MCPEnhancedReactAgent(
    mcp_server_url="https://your-mcp-server.example.com",
    api_key="your-api-key",  # if required
    mcp_namespace="mcp"
)
```

2. Run the modified example:

```bash
python tools/mcp_agent_example.py
```

## Customizing the MCP Integration

You can customize the MCP integration by:

- Modifying the category, capability, and domain extraction logic in `MCPSDKToolProvider`
- Adding authentication to the MCP client
- Implementing custom tool selection criteria
- Extending the React agent with additional features

## Architecture

The MCP integration uses the following architecture:

1. **Mock MCP Server**: Simulates an MCP endpoint
2. **MCP SDK Client**: Connects to MCP servers
3. **MCP SDK Tool Provider**: Discovers and registers tools
4. **Enhanced Tool System**: Provides metadata and selection capabilities
5. **React Agent**: Uses selected tools to answer queries 