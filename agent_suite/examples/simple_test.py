#!/usr/bin/env python3
"""
Simple Test for Restructured Code

This script tests that the restructured code can be imported correctly.
"""
import os
import sys
import logging

# Add the parent directory to the Python path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def main():
    """Test that the restructured code can be imported correctly."""
    try:
        # Import from agent_suite package
        from agent_suite.integrations.mcp.mock_server import MockMCPServer
        from agent_suite.integrations.mcp.client import MCPSDKClient
        from agent_suite.integrations.mcp.wrapper import MCPSDKToolProvider
        
        # Create mock server
        server = MockMCPServer(port=8888)
        server.start()
        
        logger.info(f"Mock MCP server started at {server.base_url}")
        logger.info(f"Available tools: {len(server.get_tools())}")
        
        # Stop the server
        server.stop()
        logger.info("Mock MCP server stopped")
        
        print("\nTest passed: imports and basic functionality working!")
        return True
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 