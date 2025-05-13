#!/usr/bin/env python3
"""
Simple test script for the SerperSearchTool without the enhanced tool system.
"""
import asyncio
import os
import sys

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from tools.serper_api import SerperSearchTool


async def test_serper_search():
    """Test the SerperSearchTool."""
    # Create a SerperSearchTool
    serper_tool = SerperSearchTool(query="Tesla stock price")
    
    try:
        # Execute the tool
        result = await serper_tool.arun("Tesla stock price")
        
        # Print the result
        print(f"Search result: {result}")
        return True
    except Exception as e:
        print(f"Error searching: {e}")
        return False


def main():
    """Run the simple SerperSearchTool test."""
    print("Testing SerperSearchTool...")
    success = asyncio.run(test_serper_search())
    
    if success:
        print("\nTest completed successfully!")
    else:
        print("\nTest failed.")


if __name__ == "__main__":
    main() 