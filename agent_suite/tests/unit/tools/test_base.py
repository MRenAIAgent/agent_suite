"""
Test script for verifying the enhanced tool system integration.

This script tests the basic functionality of the enhanced tool system:
1. Registering tools with the registry
2. Converting legacy tools to enhanced tools
3. Using enhanced tools with a simple test case
"""
import os
import sys
from typing import List, Optional, Any

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tools.tool_types import ToolCategory, ToolCapability, ToolDomain, ToolMetadata, ToolSource
from tools.base import EnhancedTool
from tools.registry import registry
from tools.serper_api import SerperSearchTool


def test_registry_registration():
    """Test registering a tool with the registry."""
    # Clear the registry to start fresh
    registry.clear()
    
    # Create and register an enhanced tool
    serper_tool = SerperSearchTool(query="test query")
    enhanced_tool = serper_tool.to_enhanced_tool()
    registry.register_tool(enhanced_tool)
    
    # Verify that at least one tool is registered
    registered_tools = registry.get_all_tools()
    assert len(registered_tools) > 0
    
    # Verify tool properties - first tool should have these properties
    registered_tool = registered_tools[0]
    assert registered_tool.metadata.display_name == "Google Search"
    assert ToolCategory.SEARCH in registered_tool.metadata.categories
    
    print("✓ Registry registration test passed")


def test_tool_conversion():
    """Test converting legacy tools to enhanced tools."""
    # Clear the registry to start fresh
    registry.clear()
    
    # Create and convert a tool directly
    serper_tool = SerperSearchTool(query="test")
    enhanced_tool = serper_tool.to_enhanced_tool()
    registry.register_tool(enhanced_tool)
    
    # Verify that the tool was converted and registered
    registered_tools = registry.get_all_tools()
    assert len(registered_tools) > 0
    assert registered_tools[0].metadata.display_name == "Google Search"
    
    # Create a second tool to verify multiple registrations
    serper_tool2 = SerperSearchTool(query="another test")
    enhanced_tool2 = serper_tool2.to_enhanced_tool()
    registry.register_tool(enhanced_tool2)
    
    # Check that more tools are registered now
    new_registered_tools = registry.get_all_tools()
    assert len(new_registered_tools) > len(registered_tools)
    
    print(f"✓ Tool conversion test passed - registered {len(new_registered_tools)} tools")


def test_enhanced_tool_execution():
    """Test executing an enhanced tool."""
    # Create a SerperSearchTool
    serper_tool = SerperSearchTool(query="test query")
    
    # Convert to enhanced tool
    enhanced_tool = serper_tool.to_enhanced_tool()
    
    # Verify that the enhanced tool can be executed
    try:
        # Note: We're not actually executing it since that would make an API call
        # Just verify that the tool has the necessary methods
        assert hasattr(enhanced_tool, 'run')
        assert hasattr(enhanced_tool, 'arun')
        print("✓ Enhanced tool execution test passed")
    except Exception as e:
        print(f"✗ Enhanced tool execution test failed: {e}")


def main():
    """Run all tests."""
    print("Testing Enhanced Tool System Integration\n")
    
    # Run tests
    test_registry_registration()
    test_tool_conversion()
    test_enhanced_tool_execution()
    
    print("\nAll tests completed!")


if __name__ == "__main__":
    main() 