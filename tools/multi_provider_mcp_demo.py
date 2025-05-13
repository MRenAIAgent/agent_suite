"""
Demo script for using the Multi-Provider MCP client and tools.

This script shows how to:
1. Set up and connect to different MCP servers (Zapier, Zendesk)
2. List available tools from each server
3. Execute tools from different providers
4. Manage multiple provider connections simultaneously
"""
import asyncio
import json
import os
import sys
from typing import Dict, Any, Optional, List

# Add the parent directory to the path so we can import our tools
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.multi_provider_mcp_client import (
    MultiProviderMCPClient,
    MCPProviderType,
    create_zapier_mcp_client,
    create_zendesk_mcp_client
)
from tools.multi_provider_mcp_tool import MCPListTools, MCPExecuteTool, MCPSetupProvider


async def demo_setup_provider(provider_type: str) -> Optional[str]:
    """
    Interactive setup of an MCP server connection.
    
    Args:
        provider_type: Type of MCP provider (zapier, zendesk, etc.)
        
    Returns:
        MCP URL if successful, None otherwise
    """
    print("=" * 80)
    print(f"{provider_type.capitalize()} MCP Client Setup Demo")
    print("=" * 80)
    
    setup_tool = MCPSetupProvider()
    result = await setup_tool.arun(provider_type=provider_type)
    
    print(result)
    
    # Extract the URL from the success message
    if f"Successfully connected to {provider_type} MCP server at " in result:
        prefix = f"Successfully connected to {provider_type} MCP server at "
        url_start = result.find(prefix) + len(prefix)
        url_end = result.find("\n", url_start)
        return result[url_start:url_end]
    
    return None


async def demo_list_tools(mcp_url: str, provider_type: str) -> List[Dict[str, Any]]:
    """
    Demo listing available tools from an MCP server.
    
    Args:
        mcp_url: URL of the MCP server endpoint
        provider_type: Type of MCP provider
    
    Returns:
        List of available tools
    """
    print("\n" + "=" * 80)
    print(f"Listing Available Tools from {provider_type.capitalize()} MCP Server")
    print("=" * 80)
    
    list_tools_tool = MCPListTools()
    tools_json = await list_tools_tool.arun(
        mcp_url=mcp_url,
        provider_type=provider_type
    )
    
    tools = json.loads(tools_json)
    
    print(f"Found {len(tools)} available tools:")
    
    # Group tools by category or function if possible
    if provider_type == "zapier":
        # For Zapier, group by app
        app_tools = {}
        for tool in tools:
            app = tool.get("name", "Unknown").split("_")[0] if "_" in tool.get("name", "") else "Other"
            if app not in app_tools:
                app_tools[app] = []
            app_tools[app].append(tool)
        
        for app, app_tool_list in app_tools.items():
            print(f"\n🔹 {app.capitalize()} ({len(app_tool_list)} actions)")
            for tool in app_tool_list:
                print(f"  • {tool['name']}: {tool['description']}")
    else:
        # For other providers, just list the tools
        for tool in tools:
            print(f"• {tool['name']}: {tool['description']}")
    
    return tools


async def demo_execute_tool(mcp_url: str, provider_type: str, tools: List[Dict[str, Any]]) -> None:
    """
    Demo executing a tool on an MCP server.
    
    Args:
        mcp_url: URL of the MCP server endpoint
        provider_type: Type of MCP provider
        tools: List of available tools
    """
    print("\n" + "=" * 80)
    print(f"Executing a Tool on {provider_type.capitalize()} MCP Server")
    print("=" * 80)
    
    # Let the user select a tool to execute
    print("Select a tool to execute:")
    
    # Find simple tools with few parameters for the demo
    simple_tools = []
    for tool in tools:
        if len(tool.get("parameters", [])) <= 2:
            simple_tools.append(tool)
    
    # Sort by number of parameters (ascending)
    simple_tools.sort(key=lambda x: len(x.get("parameters", [])))
    
    # Display a subset of simple tools
    display_tools = simple_tools[:min(10, len(simple_tools))]
    
    for i, tool in enumerate(display_tools):
        params = ", ".join([p.get("name", "unknown") for p in tool.get("parameters", [])])
        print(f"{i+1}. {tool['name']} (Parameters: {params})")
    
    try:
        choice = int(input("\nEnter the number of the tool to execute (or 0 to skip): "))
        if choice == 0:
            print("Skipping tool execution.")
            return
        
        selected_tool = display_tools[choice - 1]
        
        # Collect parameters for the tool
        parameters = {}
        for param in selected_tool.get("parameters", []):
            param_name = param.get("name", "unknown")
            param_desc = param.get("description", "No description")
            param_required = param.get("required", False)
            
            prompt = f"Enter value for '{param_name}'"
            if param_desc:
                prompt += f" ({param_desc})"
            if param_required:
                prompt += " [required]"
            prompt += ": "
            
            value = input(prompt)
            if value:
                parameters[param_name] = value
        
        # Execute the tool
        print(f"\nExecuting {selected_tool['name']}...")
        execute_tool = MCPExecuteTool()
        result = await execute_tool.arun(
            mcp_url=mcp_url,
            provider_type=provider_type,
            tool_name=selected_tool["name"],
            parameters=parameters
        )
        
        print("\nResult:")
        print(result)
        
    except (ValueError, IndexError) as e:
        print(f"Error: {e}")
        return


async def demo_multi_provider() -> None:
    """
    Demo using multiple MCP providers simultaneously.
    """
    print("\n" + "=" * 80)
    print("Multi-Provider MCP Demo")
    print("=" * 80)
    
    # Dictionary to store provider URLs
    provider_urls = {}
    
    # Ask which providers to set up
    providers = ["zapier", "zendesk"]
    active_providers = []
    
    for provider in providers:
        setup = input(f"Do you want to set up {provider.capitalize()} MCP? (y/n): ").lower()
        if setup == "y":
            active_providers.append(provider)
    
    if not active_providers:
        print("No providers selected. Exiting.")
        return
    
    # Set up each selected provider
    for provider in active_providers:
        # Check if we already have a URL from command line args
        if len(sys.argv) > 1 and provider == "zapier":
            provider_urls["zapier"] = sys.argv[1]
            print(f"Using provided Zapier MCP URL: {provider_urls['zapier']}")
        elif len(sys.argv) > 2 and provider == "zendesk":
            provider_urls["zendesk"] = sys.argv[2]
            print(f"Using provided Zendesk MCP URL: {provider_urls['zendesk']}")
        else:
            # Run interactive setup
            url = await demo_setup_provider(provider)
            if url:
                provider_urls[provider] = url
    
    if not provider_urls:
        print("Failed to set up any providers. Exiting.")
        return
    
    # List tools for each provider
    all_tools = {}
    for provider, url in provider_urls.items():
        tools = await demo_list_tools(url, provider)
        all_tools[provider] = tools
    
    # Execute a tool from each provider
    for provider, url in provider_urls.items():
        await demo_execute_tool(url, provider, all_tools[provider])
    
    print("\nMulti-Provider Demo completed!")


async def main():
    """Main demo function."""
    # Run the multi-provider demo
    await demo_multi_provider()


if __name__ == "__main__":
    asyncio.run(main()) 