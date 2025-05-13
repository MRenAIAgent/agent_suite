# Enhanced Tool System

The enhanced tool system provides a comprehensive framework for tool management in agent-based systems, featuring registration, rich metadata, and intelligent tool selection.

## Key Features

1. **Tool Registration**: Central registry for all tools with namespacing to avoid collisions
2. **Enhanced Metadata**: Rich metadata for better tool selection and discovery
3. **Intelligent Tool Selection**: Context-aware tool selection based on roles, tasks, semantic matching, and query content
4. **MCP Integration**: Wrapper system for Model Completion Protocol (MCP) tools from various providers
5. **Dynamic Tool Selection**: Runtime selection of relevant tools based on user queries

## Architecture

The enhanced tool system consists of these key components:

- **`base.py`**: Defines the EnhancedTool base class
- **`tool_types.py`**: Defines core types for the enhanced tool system
- **`registry.py`**: Implements the central tool registry
- **`selection.py`**: Provides intelligent tool selection services
- **`tool_selector.py`**: Intelligent tool selection with various selection methods
- **`mcp_wrapper.py`**: MCP tool wrapper and registration system
- **`adapters/`**: Contains adapters for various tool sources

## Intelligent Tool Selection

The enhanced tool system provides multiple methods for selecting appropriate tools:

1. **Keyword-based Selection**: Matches tools based on keyword patterns in queries
2. **Category-based Selection**: Selects tools from appropriate categories (SEARCH, UTILITY, etc.)
3. **Capability-based Selection**: Matches tools with required capabilities (READ, WRITE, COMPUTE, etc.)
4. **Domain-based Selection**: Finds tools relevant to specific domains (FINANCE, HEALTHCARE, etc.)
5. **Role-based Selection**: Selects tools appropriate for a given agent role
6. **Hybrid Selection**: Combines all methods with weighted scoring for optimal selection

Example usage:

```python
from tools.tool_selector import ToolSelector, SelectionMethod

# Create a selector
selector = ToolSelector()

# Select tools based on a user query
tools = selector.select_tools(
    query="Calculate 15% of $85 and convert to euros",
    method=SelectionMethod.HYBRID,
    max_tools=3
)
```

## MCP Tool Integration

The enhanced tool system provides seamless integration with MCP (Model Completion Protocol) tools:

1. **MCPToolWrapper**: Wraps MCP tools to use with the enhanced tool system
2. **Metadata Extraction**: Automatically extracts metadata from MCP tool definitions
3. **Bulk Registration**: Register multiple MCP tools at once with namespacing

Example of MCP tool registration:

```python
from tools.mcp_wrapper import register_mcp_tool, register_mcp_tools

# Register a single MCP tool
wrapped_tool = register_mcp_tool(
    mcp_tool_name="github_search_repositories",
    mcp_namespace="github",
    mcp_tool_data=tool_data,
    mcp_tool_instance=tool_instance
)

# Register multiple MCP tools at once
mcp_tools = register_mcp_tools(
    mcp_tools=tools_data,
    registry_namespace_mapping={"tool_name": "namespace"}
)
```

## Integration with Existing Tools

The enhanced tool system provides seamless integration with existing tools:

1. **Automatic Conversion**: Legacy tools can be automatically converted to enhanced tools
2. **Manual Adapters**: Custom adapters for specific tool types
3. **Backward Compatibility**: Enhanced tools can be used wherever legacy tools are accepted

### Using with React Agent

The React agent has been updated to work with the enhanced tool system:

```python
from agents.examples.react_agent_example import ReactAgentExample

# Create a React agent with enhanced tools
agent = ReactAgentExample(max_iterations=15)

# Get the query response
response = agent.run("What is the current stock price of Tesla?")
```

## Examples

Example scripts demonstrating the enhanced tool system:

- `tools/enhanced_react_agent_example.py`: React agent with dynamic tool selection
- `tools/mcp_registration_example.py`: MCP tool registration and usage example
- `tools/test_enhanced_tools.py`: Basic tests for the enhanced tool system
- `tools/calculator.py`: Example custom tool with enhanced metadata

### Enhanced React Agent Example

This example demonstrates a React agent that dynamically selects tools based on the user's query:

```python
# Create the enhanced agent
agent = EnhancedReactAgent()

# Run with different queries to see tool selection in action
response = agent.run("Calculate 356 * 288 and explain the process")
response = agent.run("Is a dolphin a fish or a mammal?")
```

### MCP Tool Registration Example

This example shows how to register and use MCP tools:

```python
# Initialize the example
example = MCPRegistrationExample()

# Register individual and bulk MCP tools
example.register_individual_tool()
example.register_tool_collection()

# Demonstrate selection with the registered tools
example.demonstrate_tool_selection()
```

## Adding New Tools

To add a new tool to the enhanced system:

1. Create a new class that inherits from `EnhancedTool`
2. Define rich metadata in the constructor
3. Register the tool with the registry

Example:

```python
from tools.base import EnhancedTool
from tools.tool_types import ToolMetadata, ToolCategory, ToolCapability, ToolDomain, ToolSource
from tools.registry import registry

class MyNewTool(EnhancedTool):
    """A new enhanced tool."""
    
    def __init__(self, **data):
        """Initialize with metadata."""
        if "metadata" not in data:
            data["metadata"] = ToolMetadata(
                name="my_tool",
                display_name="My Tool",
                description="A useful new tool",
                categories=[ToolCategory.GENERAL],
                domains=[ToolDomain.GENERAL],
                capabilities=[ToolCapability.GENERAL]
            )
        super().__init__(**data)
    
    async def arun(self, **kwargs):
        """Run the tool asynchronously."""
        # Implementation here
        pass
    
    def run(self, **kwargs):
        """Run the tool synchronously."""
        # Implementation here
        pass

# Register the tool
tool = MyNewTool()
registry.register_tool(tool)
```

## Converting Existing Tools

To convert an existing tool to use the enhanced system:

1. Add a `to_enhanced_tool` method to your tool class
2. Return an EnhancedTool instance with appropriate metadata

Example:

```python
def to_enhanced_tool(self) -> EnhancedTool:
    """Convert this tool to an enhanced tool."""
    class EnhancedWrapper(EnhancedTool):
        # Implementation here
        pass
    
    return EnhancedWrapper()
```

Or use the automatic conversion:

```python
from tools.adapters.integration import auto_convert_legacy_tools

# Auto-convert and register all legacy tools
converted_tools = auto_convert_legacy_tools()
``` 