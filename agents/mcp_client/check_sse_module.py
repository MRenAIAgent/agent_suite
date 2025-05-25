#!/usr/bin/env python3
"""Check what's in the SSE module of MCP package."""
import mcp.client.sse

print("Contents of mcp.client.sse module:")
for item in dir(mcp.client.sse):
    if not item.startswith('__'):
        print(f"- {item}")

print("\nTrying to check if there's some sort of SseServerParameters in the module...")
for item in dir(mcp.client.sse):
    if "sse" in item.lower() or "server" in item.lower() or "param" in item.lower():
        print(f"Potential match: {item}")
        try:
            obj = getattr(mcp.client.sse, item)
            print(f"  Type: {type(obj)}")
            if hasattr(obj, "__doc__") and obj.__doc__:
                print(f"  Doc: {obj.__doc__.strip()}")
        except Exception as e:
            print(f"  Error accessing: {e}") 