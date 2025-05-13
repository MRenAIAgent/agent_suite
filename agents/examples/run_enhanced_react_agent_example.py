#!/usr/bin/env python3
"""
Run the Enhanced React Agent Example with Tool Selection

This script runs the Enhanced React Agent example that demonstrates
the new tool registration system with namespaces, rich metadata, and
dynamic tool selection based on user queries.
"""
import os
import sys

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from agents.examples.enhanced_react_agent_example import EnhancedReactAgentExample
from tools.tool_types import ToolCapability, ToolDomain
from tools.tool_selector import SelectionMethod


def display_tools_by_capability(agent):
    """Display tools filtered by capability."""
    print("\n=== Tools Filtered by Capability ===")
    
    # Get all tools
    all_tools = agent.registry.get_all_tools()
    
    # Filter for tools with FETCH capability
    fetch_tools = [tool for tool in all_tools if ToolCapability.FETCH in tool.metadata.capabilities]
    print(f"\nTools with FETCH capability ({len(fetch_tools)}):")
    for tool in fetch_tools:
        print(f"- {tool.metadata.display_name} (Namespace: {get_tool_namespace(agent.registry, tool)})")
    
    # Filter for tools with ANALYZE capability
    analyze_tools = [tool for tool in all_tools if ToolCapability.ANALYZE in tool.metadata.capabilities]
    print(f"\nTools with ANALYZE capability ({len(analyze_tools)}):")
    for tool in analyze_tools:
        print(f"- {tool.metadata.display_name} (Namespace: {get_tool_namespace(agent.registry, tool)})")


def display_tools_by_domain(agent):
    """Display tools filtered by domain."""
    print("\n=== Tools Filtered by Domain ===")
    
    # Get all tools
    all_tools = agent.registry.get_all_tools()
    
    # Filter for tools with GENERAL domain
    general_tools = [tool for tool in all_tools if ToolDomain.GENERAL in tool.metadata.domains]
    print(f"\nTools with GENERAL domain ({len(general_tools)}):")
    for tool in general_tools:
        print(f"- {tool.metadata.display_name} (Namespace: {get_tool_namespace(agent.registry, tool)})")
    
    # Filter for tools with FINANCE domain
    finance_tools = [tool for tool in all_tools if ToolDomain.FINANCE in tool.metadata.domains]
    print(f"\nTools with FINANCE domain ({len(finance_tools)}):")
    for tool in finance_tools:
        print(f"- {tool.metadata.display_name} (Namespace: {get_tool_namespace(agent.registry, tool)})")


def get_tool_namespace(registry, tool):
    """Get the namespace of a tool."""
    for namespace, tools in registry.namespaces.items():
        for tool_name in tools:
            if registry.tools[tool_name] == tool:
                return namespace
    return "unknown"


def display_selected_tools(agent, query):
    """Display tools selected for a specific query."""
    print(f"\n=== Tools Selected for Query: '{query}' ===")
    
    # Use the tool selector to select tools for the query
    selected_tools = agent.tool_selector.select_tools(
        query=query,
        method=SelectionMethod.HYBRID,
        role=agent.role,
        task=agent.task
    )
    
    # Display the selected tools
    for i, tool in enumerate(selected_tools):
        print(f"{i+1}. {tool.metadata.display_name} ({tool.metadata.name})")
        print(f"   Namespace: {get_tool_namespace(agent.registry, tool)}")
        print(f"   Categories: {[c.value for c in tool.metadata.categories]}")
        print(f"   Capabilities: {[c.value for c in tool.metadata.capabilities]}")


def main():
    """Run the Enhanced React Agent Example with Tool Selection."""
    print("=== Enhanced React Agent with Dynamic Tool Selection ===")
    
    # Create the agent
    agent = EnhancedReactAgentExample(max_iterations=15)
    
    # Show registered tools by namespace
    tools_by_namespace = agent.get_registered_tools()
    print("\nRegistered tools by namespace:")
    
    for namespace, tools in tools_by_namespace.items():
        print(f"\nNamespace: {namespace} ({len(tools)} tools)")
        for tool in tools:
            print(f"- {tool.metadata.display_name}: {tool.metadata.description}")
            print(f"  Categories: {[c.value for c in tool.metadata.categories]}")
            print(f"  Capabilities: {[c.value for c in tool.metadata.capabilities]}")
    
    # Display tools by capability and domain
    display_tools_by_capability(agent)
    display_tools_by_domain(agent)
    
    # Define test queries
    test_queries = [
        "What is the weather like in Tokyo?",
        "What is the current stock price of Tesla?",
        "Calculate 125 divided by 5",
        "Who is the current CEO of Apple?",
        "What's the meaning of life?"
    ]
    
    # Demonstrate tool selection for each query
    for query in test_queries:
        display_selected_tools(agent, query)
    
    # Execute each query
    for query in test_queries:
        print(f"\n\n{'='*50}")
        print(f"QUERY: {query}")
        print(f"{'='*50}")
        
        response = agent.run(query)
        print(f"\nRESPONSE:\n{response}")


if __name__ == "__main__":
    main() 