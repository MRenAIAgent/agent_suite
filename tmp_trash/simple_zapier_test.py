#!/usr/bin/env python3
"""
Simple test script for connecting to Zapier MCP.
This script tries to use the installed MCP package to connect to Zapier.
"""
import asyncio
import sys
import mcp
from pprint import pprint

# Print the available MCP components
print("Available MCP components:")
pprint(dir(mcp))
print("\n")

async def test_zapier_connection(zapier_mcp_url=None):
    """
    Test connection to Zapier MCP server.
    
    Args:
        zapier_mcp_url: Zapier MCP URL (will prompt if not provided)
    """
    if not zapier_mcp_url:
        print("To use Zapier MCP, you need to generate an MCP endpoint from https://zapier.com/mcp")
        print("Follow these steps:")
        print("1. Go to https://zapier.com/mcp and sign in")
        print("2. Click 'Get started'")
        print("3. Generate your MCP endpoint")
        print("4. Copy the URL and paste it below")
        
        zapier_mcp_url = input("Enter your Zapier MCP URL: ").strip()
    
    print(f"Testing connection to {zapier_mcp_url}...")
    
    try:
        # Using the installed MCP version to create a connection
        if hasattr(mcp, 'SseServerParameters') and hasattr(mcp, 'ClientSession'):
            # Use newer MCP SDK style
            from mcp.client.sse import sse_client
            
            server_params = mcp.SseServerParameters(url=zapier_mcp_url)
            read_stream, write_stream = await sse_client(server_params)
            
            client_session = mcp.ClientSession(read_stream, write_stream)
            await client_session.initialize()
            
            # List available tools
            result = await client_session.list_tools()
            print(f"Success! Found {len(result.tools)} tools.")
            
            # Print the first few tools
            for i, tool in enumerate(result.tools[:5]):
                print(f"Tool {i+1}: {tool.get('name')} - {tool.get('description', 'No description')}")
            
            # Disconnect
            await client_session.__aexit__(None, None, None)
            
            return True
        else:
            # Try alternative approach with the version you have installed
            print("The installed MCP version doesn't have the expected interfaces.")
            print("Available interfaces in your MCP version:")
            for name in dir(mcp):
                if not name.startswith('__'):
                    print(f"- {name}")
            return False
    except Exception as e:
        print(f"Error connecting to Zapier MCP: {e}")
        return False

if __name__ == "__main__":
    # Get URL from command line if provided
    zapier_url = sys.argv[1] if len(sys.argv) > 1 else None
    
    asyncio.run(test_zapier_connection(zapier_url)) 