#!/usr/bin/env python3
"""Check if the SSE module exists in the MCP package."""
import importlib
import sys

print("Checking for SSE module in MCP...")
try:
    importlib.import_module('mcp.client.sse')
    print("Success: SSE module exists")
except ImportError as e:
    print(f"Error: SSE module not found - {e}")

print("\nAvailable modules in mcp.client:")
import mcp.client
for item in dir(mcp.client):
    if not item.startswith('__'):
        print(f"- {item}")

print("\nChecking for SseServerParameters...")
try:
    from mcp import SseServerParameters
    print("Success: SseServerParameters exists")
except ImportError as e:
    print(f"Error: SseServerParameters not found - {e}")

print("\nLooking for files in MCP package...")
import os
import mcp
mcp_path = os.path.dirname(mcp.__file__)
print(f"MCP path: {mcp_path}")
for root, dirs, files in os.walk(mcp_path):
    relative_path = os.path.relpath(root, mcp_path)
    if relative_path == ".":
        relative_path = ""
    else:
        relative_path += "/"
    
    for file in files:
        if file.endswith('.py') and not file.startswith('__'):
            print(f"- {relative_path}{file}") 