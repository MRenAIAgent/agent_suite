"""
Demo script for using the Zapier MCP client and tools.

This script shows how to:
1. Set up and connect to a Zapier MCP server
2. List available Zapier actions
3. Execute a Zapier action
"""
import asyncio
import json
import os
import sys
from typing import Dict, Any, Optional

# Add the parent directory to the path so we can import our tools
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.zapier_mcp_client import ZapierMCPClient
from tools.zapier_mcp_tool import ZapierListActions, ZapierExecuteAction, ZapierSetupMCP


async def demo_setup() -> Optional[str]:
    """
    Interactive setup of Zapier MCP connection.
    
    Returns:
        Zapier MCP URL if successful, None otherwise
    """
    print("=" * 80)
    print("Zapier MCP Client Setup Demo")
    print("=" * 80)
    
    setup_tool = ZapierSetupMCP()
    result = await setup_tool.arun()
    
    print(result)
    
    # Extract the URL from the success message
    if "Successfully connected to Zapier MCP server at" in result:
        url_start = result.find("at ") + 3
        url_end = result.find("\n", url_start)
        return result[url_start:url_end]
    
    return None


async def demo_list_actions(zapier_mcp_url: str) -> None:
    """
    Demo listing available Zapier actions.
    
    Args:
        zapier_mcp_url: URL of the Zapier MCP server endpoint
    """
    print("\n" + "=" * 80)
    print("Listing Available Zapier Actions")
    print("=" * 80)
    
    list_actions_tool = ZapierListActions()
    actions_json = await list_actions_tool.arun(zapier_mcp_url=zapier_mcp_url)
    
    actions = json.loads(actions_json)
    
    print(f"Found {len(actions)} available actions:")
    
    # Print a summary of the actions grouped by app
    app_actions = {}
    for action in actions:
        app = action.get("app", "Unknown")
        if app not in app_actions:
            app_actions[app] = []
        app_actions[app].append(action)
    
    for app, app_action_list in app_actions.items():
        print(f"\n🔹 {app} ({len(app_action_list)} actions)")
        for action in app_action_list:
            print(f"  • {action['name']}: {action['description']}")
    
    return actions


async def demo_execute_action(zapier_mcp_url: str, actions: list) -> None:
    """
    Demo executing a Zapier action.
    
    Args:
        zapier_mcp_url: URL of the Zapier MCP server endpoint
        actions: List of available actions
    """
    print("\n" + "=" * 80)
    print("Executing a Zapier Action")
    print("=" * 80)
    
    # Let the user select an action to execute
    print("Select an action to execute:")
    
    # Find simple actions with few parameters for the demo
    simple_actions = []
    for action in actions:
        if len(action.get("parameters", [])) <= 2:
            simple_actions.append(action)
    
    # Sort by number of parameters (ascending)
    simple_actions.sort(key=lambda x: len(x.get("parameters", [])))
    
    # Display a subset of simple actions
    display_actions = simple_actions[:min(10, len(simple_actions))]
    
    for i, action in enumerate(display_actions):
        params = ", ".join([p.get("name", "unknown") for p in action.get("parameters", [])])
        print(f"{i+1}. {action['app']} - {action['name']} (Parameters: {params})")
    
    try:
        choice = int(input("\nEnter the number of the action to execute (or 0 to skip): "))
        if choice == 0:
            print("Skipping action execution.")
            return
        
        selected_action = display_actions[choice - 1]
        
        # Collect parameters for the action
        parameters = {}
        for param in selected_action.get("parameters", []):
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
        
        # Execute the action
        print(f"\nExecuting {selected_action['app']} - {selected_action['name']}...")
        execute_tool = ZapierExecuteAction()
        result = await execute_tool.arun(
            zapier_mcp_url=zapier_mcp_url,
            action_name=selected_action["name"],
            parameters=parameters
        )
        
        print("\nResult:")
        print(result)
        
    except (ValueError, IndexError) as e:
        print(f"Error: {e}")
        return


async def main():
    """Main demo function."""
    # Allow specifying a Zapier MCP URL as a command-line argument
    zapier_mcp_url = None
    if len(sys.argv) > 1:
        zapier_mcp_url = sys.argv[1]
        print(f"Using provided Zapier MCP URL: {zapier_mcp_url}")
    
    # If no URL provided, run the setup
    if not zapier_mcp_url:
        zapier_mcp_url = await demo_setup()
        if not zapier_mcp_url:
            print("Setup failed. Exiting.")
            return
    
    # List available actions
    actions = await demo_list_actions(zapier_mcp_url)
    
    # Execute an action
    await demo_execute_action(zapier_mcp_url, actions)
    
    print("\nDemo completed!")


if __name__ == "__main__":
    asyncio.run(main()) 