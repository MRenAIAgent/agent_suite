#!/usr/bin/env python3
"""Check the signature of the sse_client function in the MCP package."""
import inspect
import mcp.client.sse

# Get the sse_client function
sse_client = mcp.client.sse.sse_client

# Print the function's signature
print("SSE Client Function Signature:")
print(inspect.signature(sse_client))

# Print the function's docstring
print("\nSSE Client Function Docstring:")
print(sse_client.__doc__)

# Check if it's an async context manager
print("\nIs it an async context manager?")
if hasattr(sse_client, "__aenter__") or "asynccontextmanager" in str(sse_client):
    print("Yes, appears to be an async context manager")
else:
    print("No, does not appear to be an async context manager")

# Check the source code if possible
print("\nTrying to get source code:")
try:
    source = inspect.getsource(sse_client)
    print(source)
except Exception as e:
    print(f"Could not get source: {e}")

# Check module attributes that might help
print("\nModule attributes that might help:")
for name in dir(mcp.client.sse):
    if name.startswith("_"):
        continue
    attr = getattr(mcp.client.sse, name)
    if callable(attr) and "context" in str(attr) or "context" in name.lower():
        print(f"- {name}: {attr}")
        if hasattr(attr, "__doc__") and attr.__doc__:
            print(f"  Doc: {attr.__doc__.strip()}") 