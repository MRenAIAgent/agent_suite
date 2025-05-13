# Agent Suite

A Python project with AI agent tools including MCP client implementations and an enhanced tool system.

## Code Quality

This project uses:
- **Ruff**: A fast Python linter and formatter
  - Custom line length of 88 characters
  - Checks for unused imports and variables
  - Import sorting

## Enhanced Tool System

The project includes a comprehensive tool system with registration-based tools, enhanced metadata, and intelligent tool selection based on roles and tasks.

### Features:

- **Registration-Based Tools**: Centralized registry for managing tools from various sources
- **Enhanced Metadata**: Rich metadata for better tool selection and organization
- **Tool Selection Services**: Intelligent selection of tools based on roles and tasks
- **Adapter System**: Integration with multiple tool sources (LangChain, MCP)
- **Usage Statistics**: Tracking of tool usage for better selection

### Key Components:

1. **Tool Registry**: Central system for registering, discovering, and managing tools
2. **Enhanced Tool Base**: Tool base class with rich metadata and registration support
3. **Tool Selection Service**: Service for selecting appropriate tools based on roles and tasks
4. **Tool Adapters**: Adapters for integrating tools from various sources

### Usage Example:

```python
import asyncio
from tools import (
    registry, EnhancedTool, ToolMetadata, ToolCategory, 
    ToolCapability, ToolDomain, add_tools_to_agent
)
from tools.adapters import MCPToolAdapter, LangChainToolAdapter
from agents import ReActAgent

# Define a custom tool with enhanced metadata
class WeatherTool(EnhancedTool):
    """Get weather information for a specific location."""
    
    location: str
    units: str = "metric"
    
    def __init__(self, **data):
        # Initialize with enhanced metadata
        if "metadata" not in data:
            data["metadata"] = ToolMetadata(
                categories=[ToolCategory.UTILITY],
                domains=[ToolDomain.GENERAL],
                capabilities=[ToolCapability.FETCH],
                task_patterns=["weather", "forecast", "temperature"]
            )
        super().__init__(**data)
    
    async def arun(self, location: str, units: str = "metric") -> str:
        # Implementation here
        return f"Weather for {location}: 25°C, Sunny"

async def main():
    # Register a custom tool
    weather_tool = WeatherTool()
    registry.register_tool(weather_tool)
    
    # Connect to MCP server and register its tools
    mcp_adapter = MCPToolAdapter()
    await mcp_adapter.connect_to_server(
        mcp_url="https://mcp.example.com/your-endpoint",
        provider_type="github",
        api_key="your-api-key"
    )
    
    # Register LangChain tools
    try:
        from langchain.agents import Tool as LCTool
        from langchain.tools import DuckDuckGoSearchRun
        
        search_tool = LCTool(
            name="web_search",
            func=DuckDuckGoSearchRun().run,
            description="Search the web for information"
        )
        
        lc_adapter = LangChainToolAdapter()
        lc_adapter.register_langchain_tool(search_tool)
    except ImportError:
        pass
    
    # Create an agent with task-specific tools
    agent = ReActAgent(...)
    add_tools_to_agent(
        agent,
        task_description="Find the weather forecast and research climate patterns",
        max_tools=5
    )
    
    # Agent now has the most appropriate tools for the task

if __name__ == "__main__":
    asyncio.run(main())
```

### Demo Script:

The project includes a demo script that showcases the enhanced tool system features:

```bash
# Run the Enhanced Tool System demo
python tools/demo.py
```

## MCP Client Implementations

### Multi-Provider MCP Client

The project includes a Multi-Provider MCP client implementation that allows AI agents to interact with various MCP servers from different providers. This unified client can connect to Zapier, Zendesk, GitHub, Notion, and custom MCP servers.

#### Features:
- Connect to multiple MCP servers simultaneously
- Support for different transport types (SSE, STDIO)
- Provider-specific configuration handling
- Unified tool execution interface
- Client instance caching for better performance
- Interactive setup for each provider

#### Usage Example:

```python
import asyncio
from tools.multi_provider_mcp_client import (
    MultiProviderMCPClient,
    MCPProviderType,
    create_zapier_mcp_client,
    create_zendesk_mcp_client
)

async def main():
    # Create and connect to a Zapier MCP server
    zapier_client = await create_zapier_mcp_client(
        mcp_url="https://mcp.zapier.com/your-endpoint",
        api_key="optional-api-key" 
    )
    
    # Create and connect to a Zendesk MCP server
    zendesk_client = await create_zendesk_mcp_client(
        mcp_url="https://your-zendesk-mcp-endpoint.com",
        api_key="your-zendesk-api-key"
    )
    
    # List available tools from each provider
    zapier_tools = await zapier_client.list_available_tools()
    zendesk_tools = await zendesk_client.list_available_tools()
    
    # Execute a tool on Zapier
    zapier_result = await zapier_client.execute_tool(
        "gmail_send_email",
        {
            "to": "recipient@example.com",
            "subject": "Email from Zapier MCP",
            "body": "This email was sent via Zapier MCP!"
        }
    )
    
    # Execute a tool on Zendesk
    zendesk_result = await zendesk_client.execute_tool(
        "zendesk_create_ticket",
        {
            "subject": "New support request",
            "description": "Customer needs help with their account"
        }
    )
    
    # Disconnect from both services
    await zapier_client.disconnect()
    await zendesk_client.disconnect()

if __name__ == "__main__":
    asyncio.run(main())
```

#### Demo Script:

The project includes a multi-provider demo script to showcase connecting to and using different MCP providers:

```bash
# Run the Multi-Provider MCP demo
python tools/multi_provider_mcp_demo.py
```

### Zapier MCP Client

The project also includes a specialized Zapier MCP client implementation that allows AI agents to interact with Zapier's MCP servers, providing access to 7,000+ apps and 30,000+ actions without complex API integrations.

#### Features:
- Connect to Zapier MCP servers using MCP's SSE transport
- List all available Zapier actions
- Execute actions with parameters
- Support for authentication
- Client caching for improved performance
- Interactive setup and testing

#### Demo Script:

```bash
# Run the Zapier MCP demo
python tools/zapier_mcp_demo.py
```

## Getting Started

```bash
# Install dependencies
pip install -r requirements.txt

# Run linter
ruff check .

# Run the enhanced tool system demo
python tools/demo.py
```

## Project Structure

- `tools/`: Tool implementations and core tool system
  - `types.py`: Core type definitions for the enhanced tool system
  - `base.py`: Enhanced tool base classes
  - `registry.py`: Tool registry for tool management
  - `selection.py`: Tool selection service
  - `adapters/`: Adapters for integrating various tool sources
    - `mcp_adapter.py`: Adapter for MCP tools
    - `langchain_adapter.py`: Adapter for LangChain tools
  - `demo.py`: Demo script for the enhanced tool system
- `tools/multi_provider_mcp_client.py`: Multi-provider MCP client implementation
- `tools/multi_provider_mcp_tool.py`: Tools for interacting with multiple MCP servers
- `tools/multi_provider_mcp_demo.py`: Demo script for multi-provider MCP connections
- `tools/zapier_mcp_client.py`: Zapier-specific MCP client implementation
- `tools/zapier_mcp_tool.py`: Tools for interacting with Zapier MCP
- `tools/zapier_mcp_demo.py`: Demo script for the Zapier MCP client
