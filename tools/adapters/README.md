# Tool Adapters

This directory contains adapters for integrating various tool sources into the enhanced tool system.

## Overview

The adapters package provides:

1. **Legacy Tool Adapters** - Convert original tools to use the enhanced tool system 
2. **External Tool Adapters** - Integrate tools from external sources (LangChain, MCP, etc.)
3. **Integration Utilities** - Functions for bulk registration and auto-conversion

## Legacy Tool Adapters

These adapters convert the original tools to use the enhanced tool system:

- `mcp_tool_adapter.py` - Adapters for MCP tools
- `serpapi_tool_adapter.py` - Adapter for SerpAPI search tool
- `animal_type_adapter.py` - Adapter for the AnimalType tool

## External Tool Adapters

These adapters integrate tools from external sources:

- `langchain_adapter.py` - Adapter for LangChain tools
- `mcp_adapter.py` - Adapter for MCP servers

## Integration Utilities

The `integration.py` module provides utilities for:

- Registering all adapted legacy tools at once
- Automatically converting any original tools to enhanced tools

## Usage Examples

### Option 1: Register specific legacy tool adapters

```python
from tools.adapters.mcp_tool_adapter import register_mcp_tools
from tools.adapters.serpapi_tool_adapter import register_serpapi_tools

# Register specific tool adapters
register_mcp_tools()
register_serpapi_tools()
```

### Option 2: Register all adapted tools at once

```python
from tools.adapters.integration import register_all_adapted_tools

# Register all adapted tools
tools = register_all_adapted_tools()
print(f"Registered {len(tools)} tools")
```

### Option 3: Automatically convert any original tools

```python
from tools.adapters.integration import auto_convert_legacy_tools

# Auto-convert any original tools
converted_tools = auto_convert_legacy_tools()
print(f"Auto-converted {len(converted_tools)} tools")
```

## Demo Script

The package includes a demo script that shows how to use the legacy tool adapters:

```bash
# Run the legacy tools demo
python tools/legacy_tools_demo.py
``` 