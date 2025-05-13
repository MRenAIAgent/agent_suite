"""
Tool adapters for integrating various tool sources into the enhanced tool system.

This package provides adapters for converting tools from various sources
(LangChain, MCP, etc.) to the enhanced tool format and registering them
with the tool registry.
"""
# External tools adapters
try:
    from tools.adapters.langchain_adapter import (
        LangChainToolAdapter, create_langchain_adapter
    )
except ImportError:
    pass

try:
    from tools.adapters.mcp_adapter import (
        MCPToolAdapter, MCPDynamicTool
    )
except ImportError:
    pass

# Original tool adapters (legacy tools)
try:
    from tools.adapters.mcp_tool_adapter import (
        EnhancedMCPToolBase, EnhancedMCPListTools, 
        EnhancedMCPCallTool, EnhancedMCPSequentialThinking,
        register_mcp_tools
    )
except ImportError:
    pass

try:
    from tools.adapters.serpapi_tool_adapter import (
        EnhancedSerpApiSearchTool, register_serpapi_tools
    )
except ImportError:
    pass

try:
    from tools.adapters.animal_type_adapter import (
        EnhancedAnimalTypeTool, register_animal_tools
    )
except ImportError:
    pass

# Integration functions
try:
    from tools.adapters.integration import (
        register_all_adapted_tools, auto_convert_legacy_tools
    )
except ImportError:
    pass 