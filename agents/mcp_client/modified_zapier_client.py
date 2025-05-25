#!/usr/bin/env python3
"""
Modified Zapier MCP client implementation that works with the installed MCP package version.
"""
import asyncio
import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional, Tuple, Union
import contextlib

# Import the available MCP components
import mcp
from mcp.client import session
from mcp.client.sse import sse_client

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModifiedZapierMCPClient:
    """
    Modified Zapier MCP client that works with the installed MCP package version.
    """
    
    def __init__(self, zapier_mcp_url: str, api_key: Optional[str] = None):
        """
        Initialize the Zapier MCP client.
        
        Args:
            zapier_mcp_url: URL of the Zapier MCP server endpoint
            api_key: Optional API key for authentication
        """
        self.zapier_mcp_url = zapier_mcp_url
        self.api_key = api_key
        self._client_session = None
        self._read_stream = None
        self._write_stream = None
        self._initialized = False
        self._sse_task_group = None
        
    async def connect(self) -> bool:
        """
        Connect to the Zapier MCP server.
        
        Returns:
            True if connection was successful, False otherwise
        """
        try:
            # Use the sse_client function as an async context manager
            logger.info(f"Connecting to Zapier MCP at {self.zapier_mcp_url}")
            
            # Add API key as header if provided
            headers = {}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            
            # Use a context object to store the streams outside the async with block
            class StreamContainer:
                read_stream = None
                write_stream = None
            
            container = StreamContainer()
            
            # Create a background task that will keep the context manager alive
            async def connect_sse():
                async with sse_client(self.zapier_mcp_url, headers=headers) as (read_stream, write_stream):
                    container.read_stream = read_stream
                    container.write_stream = write_stream
                    
                    # Create a client session
                    self._client_session = mcp.ClientSession(read_stream, write_stream)
                    
                    # Initialize the session
                    await self._client_session.initialize()
                    
                    self._initialized = True
                    logger.info("Successfully connected to Zapier MCP")
                    
                    # Keep the connection alive until disconnect is called
                    while self._initialized:
                        await asyncio.sleep(1)
            
            # Start the connection task
            self._sse_task = asyncio.create_task(connect_sse())
            
            # Wait a bit for the connection to be established
            for _ in range(10):  # Wait up to 10 seconds
                await asyncio.sleep(1)
                if self._initialized:
                    self._read_stream = container.read_stream
                    self._write_stream = container.write_stream
                    return True
            
            # If we got here, the connection timed out
            self._sse_task.cancel()
            return False
            
        except Exception as e:
            logger.error(f"Failed to connect to Zapier MCP: {e}")
            return False
    
    async def disconnect(self) -> bool:
        """
        Disconnect from the Zapier MCP server.
        
        Returns:
            True if disconnect was successful, False otherwise
        """
        try:
            self._initialized = False
            
            if self._sse_task:
                self._sse_task.cancel()
                try:
                    await self._sse_task
                except asyncio.CancelledError:
                    pass
                
            self._client_session = None
            self._read_stream = None
            self._write_stream = None
            
            logger.info("Disconnected from Zapier MCP")
            return True
        except Exception as e:
            logger.error(f"Failed to disconnect from Zapier MCP: {e}")
            return False
    
    async def list_tools(self) -> List[Dict[str, Any]]:
        """
        List available tools from the Zapier MCP server.
        
        Returns:
            List of available tools
        """
        if not self._initialized:
            raise RuntimeError("Client not connected to Zapier MCP server")
            
        try:
            result = await self._client_session.list_tools()
            return result.tools
        except Exception as e:
            logger.error(f"Failed to list tools: {e}")
            return []
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """
        Call a tool on the Zapier MCP server.
        
        Args:
            tool_name: Name of the tool to call
            arguments: Arguments for the tool
            
        Returns:
            Result of the tool call
        """
        if not self._initialized:
            raise RuntimeError("Client not connected to Zapier MCP server")
            
        try:
            result = await self._client_session.call_tool(tool_name, arguments)
            return result
        except Exception as e:
            logger.error(f"Failed to call tool '{tool_name}': {e}")
            raise
    
    @classmethod
    async def create_and_test_connection(cls, zapier_mcp_url: str, api_key: Optional[str] = None) -> Tuple[bool, Union["ModifiedZapierMCPClient", str]]:
        """
        Create a client and test the connection to Zapier MCP.
        
        Args:
            zapier_mcp_url: URL of the Zapier MCP server endpoint
            api_key: Optional API key for authentication
            
        Returns:
            Tuple of (success, client_or_error_message)
        """
        try:
            client = cls(zapier_mcp_url, api_key)
            success = await client.connect()
            
            if success:
                # Test by listing tools to ensure we can communicate
                tools = await client.list_tools()
                if tools:
                    return True, client
                else:
                    await client.disconnect()
                    return False, "Connected to server but no tools were returned"
            else:
                return False, "Failed to connect to Zapier MCP server"
        except Exception as e:
            logger.error(f"Error connecting to Zapier MCP: {e}")
            return False, f"Error: {str(e)}"


# Alternative simpler approach using a direct async with context
async def connect_to_zapier(zapier_mcp_url: str, api_key: Optional[str] = None):
    """
    Connect to Zapier MCP using a direct context manager approach.
    
    Args:
        zapier_mcp_url: URL of the Zapier MCP server endpoint
        api_key: Optional API key for authentication
        
    Returns:
        Tuple of (success, tools or error message)
    """
    try:
        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        
        async with sse_client(zapier_mcp_url, headers=headers) as (read_stream, write_stream):
            # Create a client session
            client_session = mcp.ClientSession(read_stream, write_stream)
            
            # Initialize the session
            await client_session.initialize()
            
            # List available tools
            result = await client_session.list_tools()
            return True, result.tools
    except Exception as e:
        logger.error(f"Error connecting to Zapier MCP: {e}")
        return False, f"Error: {str(e)}"


async def test_connection(zapier_mcp_url: str = None):
    """
    Test connection to Zapier MCP.
    
    Args:
        zapier_mcp_url: Optional Zapier MCP URL (will prompt if not provided)
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
    print("Using the simple direct context manager approach...")
    
    # Try the simpler approach first
    success, result = await connect_to_zapier(zapier_mcp_url)
    
    if success:
        tools = result
        print("Successfully connected to Zapier MCP!")
        print(f"Found {len(tools)} tools available.")
        
        # Group tools by app
        app_tools = {}
        for tool in tools:
            # Extract app name from tool name (usually first part before underscore)
            tool_name = tool.get("name", "unknown")
            app_name = tool_name.split("_")[0] if "_" in tool_name else "unknown"
            
            if app_name not in app_tools:
                app_tools[app_name] = []
            app_tools[app_name].append(tool)
        
        # Display tools grouped by app
        for app, app_tool_list in app_tools.items():
            print(f"\n🔹 {app.capitalize()} ({len(app_tool_list)} tools)")
            for i, tool in enumerate(app_tool_list[:3]):  # Show only first 3 tools per app
                print(f"  • {tool.get('name')}: {tool.get('description', 'No description')}")
            if len(app_tool_list) > 3:
                print(f"  • ... and {len(app_tool_list) - 3} more")
        
        return True
    else:
        error_msg = result
        print(f"Failed to connect: {error_msg}")
        
        # Try with the class-based approach as fallback
        print("\nTrying alternative connection method...")
        
        success, result = await ModifiedZapierMCPClient.create_and_test_connection(zapier_mcp_url)
        
        if success:
            client = result
            print("Successfully connected to Zapier MCP using alternative method!")
            
            # List tools
            tools = await client.list_tools()
            print(f"Found {len(tools)} tools available.")
            
            # Disconnect
            await client.disconnect()
            return True
        else:
            error_msg = result
            print(f"Failed to connect with alternative method: {error_msg}")
            return False


if __name__ == "__main__":
    # Get URL from command line if provided
    zapier_url = sys.argv[1] if len(sys.argv) > 1 else None
    
    asyncio.run(test_connection(zapier_url)) 